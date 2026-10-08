"""
MediVoice Edge — Pipeline Orchestrator
Central coordinator for the complete speech-to-speech translation pipeline.

Pipeline Flow:
    Microphone → Audio Frontend (VAD + Denoise)
                → ASR (Speech → Text)
                → Flash Cache Check (< 50ms if hit)
                → MT (Text → Translated Text)
                → TTS (Text → Speech)
                → Speaker / Headphone

This orchestrator manages:
- Sequential execution of pipeline stages
- Latency tracking per stage and end-to-end
- Flash cache bypass for emergency phrases
- Language direction management (VI↔EN)
- Error handling and graceful degradation
"""

import logging
import math
import time
import yaml
import numpy as np
from collections import deque
from functools import wraps
from threading import Lock, RLock
from typing import Callable, Optional, Dict
from dataclasses import dataclass, field
from pathlib import Path

from .audio_frontend import AudioFrontend, AudioConfig
from .asr_engine import ASREngine, ASRResult, validate_asr_audio_window
from .mt_engine import MTEngine, MTResult, validate_mt_source_text
from .tts_engine import TTSEngine, TTSResult, validate_tts_audio
from .flash_cache import CachedPhrase, FlashCache
from .safety_guard import safety_issue_codes, validate_translation
from ..utils.bounded_file import read_stable_regular_file

logger = logging.getLogger(__name__)


def _serialized_runtime(method):
    """Serialize operations that share models, counters, or the audio sink."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._runtime_lock:
            return method(self, *args, **kwargs)

    return wrapped


def _exclusive_interactive_session(method):
    """Allow only one microphone session without blocking status queries."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if not self._interactive_lock.acquire(blocking=False):
            raise RuntimeError("Interactive microphone session is already active")
        try:
            return method(self, *args, **kwargs)
        finally:
            self._interactive_lock.release()

    return wrapped

MAX_PIPELINE_CONFIG_BYTES = 1_000_000
MAX_PIPELINE_CONFIG_DEPTH = 32
MAX_PIPELINE_CONFIG_NODES = 10_000
MAX_LATENCY_HISTORY_SAMPLES = 10_000
ALLOWED_PIPELINE_CONFIG_SECTIONS = {
    "pipeline",
    "runtime",
    "audio",
    "asr",
    "mt",
    "tts",
    "flash_cache",
    "performance_targets",
}
REQUIRED_PIPELINE_CONFIG_SECTIONS = ALLOWED_PIPELINE_CONFIG_SECTIONS - {
    "performance_targets"
}
PIPELINE_CONFIG_KEYS = {
    "pipeline": {"mode", "streaming", "language_pairs"},
    "runtime": {
        "allow_base_model_fallback",
        "allow_text_only_tts_fallback",
    },
    "audio": {
        "sample_rate",
        "bit_depth",
        "channels",
        "chunk_duration_ms",
        "vad_threshold",
        "vad_backend",
        "vad_aggressiveness",
        "noise_suppression_enabled",
        "silence_duration_ms",
    },
    "asr": {"max_input_duration_seconds", "max_new_tokens", "vi", "en"},
    "mt": {
        "model_name",
        "model_path",
        "quantization",
        "max_source_tokens",
        "max_source_characters",
        "max_new_tokens",
        "num_beams",
        "speculative_decoding",
        "medical_lexicon_path",
        "temperature",
    },
    "tts": {
        "output_sample_rate",
        "max_text_characters",
        "max_duration_seconds",
        "vi",
        "en",
    },
    "flash_cache": {
        "enabled",
        "cache_path",
        "fuzzy_matching",
        "max_response_ms",
    },
    "performance_targets": {
        "status",
        "total_latency_ms",
        "rtf",
        "asr_wer_vi",
        "mt_bleu",
        "mt_comet",
        "tts_mos",
        "required_reporting",
    },
}
REQUIRED_PIPELINE_CONFIG_KEYS = {
    "pipeline": PIPELINE_CONFIG_KEYS["pipeline"],
    "runtime": PIPELINE_CONFIG_KEYS["runtime"],
    "audio": PIPELINE_CONFIG_KEYS["audio"],
    "asr": PIPELINE_CONFIG_KEYS["asr"],
    "mt": {
        "model_path",
        "max_source_tokens",
        "max_source_characters",
        "max_new_tokens",
        "num_beams",
        "speculative_decoding",
        "medical_lexicon_path",
        "temperature",
    },
    "tts": PIPELINE_CONFIG_KEYS["tts"],
    "flash_cache": {"enabled", "cache_path", "fuzzy_matching"},
}
ASR_LANGUAGE_CONFIG_KEYS = {
    "model_name",
    "model_path",
    "quantization",
    "max_length",
    "language",
    "task",
}
TTS_LANGUAGE_CONFIG_KEYS = {"engine", "model_path", "speaker_id", "sample_rate"}


class _UniqueKeySafeLoader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self._composition_depth = 0
        self._composed_nodes = 0

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "pipeline config aliases are not allowed",
                self.peek_event().start_mark,
            )
        if self._composition_depth >= MAX_PIPELINE_CONFIG_DEPTH:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "pipeline config nesting is too deep",
                self.peek_event().start_mark,
            )
        if self._composed_nodes >= MAX_PIPELINE_CONFIG_NODES:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "pipeline config contains too many nodes",
                self.peek_event().start_mark,
            )
        self._composition_depth += 1
        self._composed_nodes += 1
        try:
            return super().compose_node(parent, index)
        finally:
            self._composition_depth -= 1


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _read_pipeline_config(path: Path) -> str:
    """Read a stable, bounded regular config file without exposing its content."""
    payload = read_stable_regular_file(
        path,
        maximum_bytes=MAX_PIPELINE_CONFIG_BYTES,
        label="Pipeline config",
    )
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("Pipeline config must be valid UTF-8") from None


@dataclass
class PipelineResult:
    """Complete result from end-to-end pipeline execution."""
    # Input
    source_audio_duration_s: float
    # ASR
    asr_text: str
    asr_language: str
    asr_confidence: float
    asr_latency_ms: float
    # Translation
    translated_text: str
    target_language: str
    mt_latency_ms: float
    from_cache: bool
    # TTS
    output_audio: Optional[np.ndarray]
    output_sample_rate: int
    tts_latency_ms: float
    tts_rtf: float
    # Overall
    total_latency_ms: float
    overall_rtf: float
    # Breakdown
    latency_breakdown: Dict[str, float] = field(default_factory=dict)
    safety_passed: bool = True
    safety_issues: list[str] = field(default_factory=list)
    requires_confirmation: bool = False
    degraded_mode: Optional[str] = None
    degradation_code: Optional[str] = None


def tts_degradation_code(error: Exception) -> str:
    """Return a stable public code without exposing backend error details."""
    if isinstance(error, FileNotFoundError):
        return "tts_artifact_unavailable"
    if isinstance(error, OSError):
        return "tts_device_io_error"
    if isinstance(error, ValueError):
        return "tts_output_validation_error"
    return "tts_runtime_error"


class MediVoicePipeline:
    """
    Main pipeline orchestrator for MediVoice Edge.

    Connects all pipeline stages and manages the translation flow:
    Audio → ASR → (Cache Check) → MT → TTS → Audio Output

    Usage:
        pipeline = MediVoicePipeline()
        pipeline.load()
        result = pipeline.translate_speech(audio_array)
        pipeline.play(result)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pipeline with configuration.

        Args:
            config_path: Path to pipeline_config.yaml
        """
        self._runtime_lock = RLock()
        self._interactive_lock = Lock()
        self.config = self._load_config(config_path)

        # Initialize pipeline stages
        audio_settings = self.config.get("audio", {})
        asr_cfg = self.config.get("asr", {})
        max_input_duration_seconds = asr_cfg.get(
            "max_input_duration_seconds",
            30.0,
        )
        audio_cfg = AudioConfig(
            sample_rate=audio_settings.get("sample_rate", 16000),
            bit_depth=audio_settings.get("bit_depth", 16),
            channels=audio_settings.get("channels", 1),
            chunk_duration_ms=audio_settings.get("chunk_duration_ms", 500),
            vad_threshold=audio_settings.get("vad_threshold", 0.5),
            vad_backend=audio_settings.get("vad_backend", "webrtc"),
            vad_aggressiveness=audio_settings.get("vad_aggressiveness", 2),
            noise_suppression_enabled=audio_settings.get(
                "noise_suppression_enabled",
                False,
            ),
            silence_duration_ms=audio_settings.get("silence_duration_ms", 800),
            max_segment_duration_seconds=max_input_duration_seconds,
        )

        self.audio_frontend = AudioFrontend(config=audio_cfg)

        runtime_cfg = self.config.get("runtime", {})
        allow_base_fallback = runtime_cfg.get("allow_base_model_fallback", False)
        self.allow_text_only_tts_fallback = runtime_cfg.get(
            "allow_text_only_tts_fallback",
            False,
        )

        self.asr_engine = ASREngine(
            vi_model_path=asr_cfg.get("vi", {}).get("model_path", "models/asr/phowhisper-small-medical"),
            en_model_path=asr_cfg.get("en", {}).get("model_path", "models/asr/distil-whisper-en"),
            allow_base_fallback=allow_base_fallback,
            max_input_duration_seconds=max_input_duration_seconds,
            max_new_tokens=asr_cfg.get("max_new_tokens", 225),
        )

        mt_cfg = self.config.get("mt", {})
        self.mt_engine = MTEngine(
            model_path=mt_cfg.get("model_path", "models/mt/nllb-medical"),
            lexicon_path=mt_cfg.get("medical_lexicon_path"),
            max_new_tokens=mt_cfg.get("max_new_tokens", 256),
            max_source_tokens=mt_cfg.get("max_source_tokens", 256),
            max_source_characters=mt_cfg.get("max_source_characters", 4096),
            num_beams=mt_cfg.get("num_beams", 1),
            temperature=mt_cfg.get("temperature", 0.1),
            allow_base_fallback=allow_base_fallback,
        )

        tts_cfg = self.config.get("tts", {})
        self.tts_engine = TTSEngine(
            vi_model_path=tts_cfg.get("vi", {}).get(
                "model_path",
                "models/tts/vi_VN-vivos-x_low.onnx",
            ),
            en_model_path=tts_cfg.get("en", {}).get(
                "model_path",
                "models/tts/en_US-lessac-medium.onnx",
            ),
            output_sample_rate=tts_cfg.get("output_sample_rate"),
            max_text_characters=tts_cfg.get("max_text_characters", 4096),
            max_duration_seconds=tts_cfg.get("max_duration_seconds", 120.0),
        )

        flash_cfg = self.config.get("flash_cache", {})
        self.flash_cache_enabled = flash_cfg.get("enabled", True)
        self.flash_cache = FlashCache(
            cache_path=flash_cfg.get("cache_path"),
            allow_fuzzy=flash_cfg.get("fuzzy_matching", False),
        )

        self._is_loaded = False

        # Performance tracking
        self._total_translations = 0
        self._cache_hits = 0
        self._text_only_tts_fallbacks = 0
        self._tts_startup_degradation_code: Optional[str] = None
        self._latency_history = deque(
            maxlen=MAX_LATENCY_HISTORY_SAMPLES
        )

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load pipeline configuration from YAML file."""
        if config_path is None:
            logger.warning("No config file found. Using defaults.")
            return {}
        path = Path(config_path)
        content = _read_pipeline_config(path)
        try:
            config = yaml.load(content, Loader=_UniqueKeySafeLoader)
        except (yaml.YAMLError, RecursionError):
            raise ValueError("Invalid pipeline config YAML") from None
        if not isinstance(config, dict):
            raise ValueError("Pipeline config root must be a mapping")
        self._validate_config(config)
        logger.info("Pipeline config loaded from: %s", path)
        return config

    @staticmethod
    def _require_mapping(config: dict, section: str) -> dict:
        value = config.get(section)
        if not isinstance(value, dict):
            raise ValueError(f"Pipeline config section {section} must be a mapping")
        return value

    @staticmethod
    def _validate_keys(
        value: dict,
        context: str,
        allowed: set[str],
        required: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"{context} has unknown keys: {sorted(unknown)}")
        missing = required - set(value)
        if missing:
            raise ValueError(f"{context} is missing required keys: {sorted(missing)}")

    @classmethod
    def _validate_config(cls, config: dict) -> None:
        sections = set(config)
        unknown = sections - ALLOWED_PIPELINE_CONFIG_SECTIONS
        if unknown:
            raise ValueError(
                f"Pipeline config has unknown top-level sections: {sorted(unknown)}"
            )
        missing = REQUIRED_PIPELINE_CONFIG_SECTIONS - sections
        if missing:
            raise ValueError(
                f"Pipeline config is missing required sections: {sorted(missing)}"
            )
        for section in sections:
            section_config = cls._require_mapping(config, section)
            cls._validate_keys(
                section_config,
                section,
                PIPELINE_CONFIG_KEYS[section],
                REQUIRED_PIPELINE_CONFIG_KEYS.get(section, frozenset()),
            )

        pipeline_cfg = cls._require_mapping(config, "pipeline")
        if pipeline_cfg.get("mode") != "cascaded":
            raise ValueError("pipeline.mode must be 'cascaded'")
        streaming = pipeline_cfg.get("streaming")
        if type(streaming) is not bool:
            raise ValueError("pipeline.streaming must be a boolean")
        if streaming:
            raise ValueError("pipeline.streaming=true is not implemented")
        language_pairs = pipeline_cfg.get("language_pairs")
        if not isinstance(language_pairs, list) or any(
            not isinstance(pair, dict)
            or set(pair) != {"source", "target"}
            for pair in language_pairs or []
        ):
            raise ValueError("pipeline.language_pairs must contain source/target mappings")
        directions = {
            (pair["source"], pair["target"])
            for pair in language_pairs
        }
        if directions != {("vi", "en"), ("en", "vi")} or len(language_pairs) != 2:
            raise ValueError(
                "pipeline.language_pairs must contain exactly vi_to_en and en_to_vi"
            )

        runtime_cfg = cls._require_mapping(config, "runtime")
        flash_cfg = cls._require_mapping(config, "flash_cache")
        boolean_fields = (
            (runtime_cfg, "allow_base_model_fallback", "runtime"),
            (runtime_cfg, "allow_text_only_tts_fallback", "runtime"),
            (
                cls._require_mapping(config, "audio"),
                "noise_suppression_enabled",
                "audio",
            ),
            (flash_cfg, "enabled", "flash_cache"),
            (flash_cfg, "fuzzy_matching", "flash_cache"),
        )
        for section_config, key, section in boolean_fields:
            if type(section_config.get(key)) is not bool:
                raise ValueError(f"{section}.{key} must be a boolean")
        if flash_cfg["fuzzy_matching"]:
            raise ValueError("flash_cache.fuzzy_matching=true is not supported")

        cache_path = flash_cfg.get("cache_path")
        if cache_path is not None and (
            not isinstance(cache_path, str) or not cache_path.strip()
        ):
            raise ValueError("flash_cache.cache_path must be null or a non-empty string")

        asr_cfg = cls._require_mapping(config, "asr")
        tts_cfg = cls._require_mapping(config, "tts")
        for section, section_config in (("asr", asr_cfg), ("tts", tts_cfg)):
            for language in ("vi", "en"):
                language_config = section_config.get(language)
                if not isinstance(language_config, dict):
                    raise ValueError(f"{section}.{language} must be a mapping")
                cls._validate_keys(
                    language_config,
                    f"{section}.{language}",
                    (
                        ASR_LANGUAGE_CONFIG_KEYS
                        if section == "asr"
                        else TTS_LANGUAGE_CONFIG_KEYS
                    ),
                    {"model_path"},
                )
                model_path = language_config.get("model_path")
                if not isinstance(model_path, str) or not model_path.strip():
                    raise ValueError(
                        f"{section}.{language}.model_path must be a non-empty string"
                    )
        mt_cfg = cls._require_mapping(config, "mt")
        model_path = mt_cfg.get("model_path")
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("mt.model_path must be a non-empty string")
        lexicon_path = mt_cfg.get("medical_lexicon_path")
        if lexicon_path is not None and (
            not isinstance(lexicon_path, str) or not lexicon_path.strip()
        ):
            raise ValueError(
                "mt.medical_lexicon_path must be null or a non-empty string"
            )
        speculative_decoding = mt_cfg["speculative_decoding"]
        if type(speculative_decoding) is not bool:
            raise ValueError("mt.speculative_decoding must be a boolean")
        if speculative_decoding:
            raise ValueError("mt.speculative_decoding=true is not implemented")

    @staticmethod
    def _resolve_target_language(source_lang: str, target_lang: Optional[str]) -> str:
        """Validate the bilingual contract before cache or model dispatch."""
        supported = {"vi", "en"}
        resolved = target_lang or ("en" if source_lang == "vi" else "vi")
        if source_lang not in supported or resolved not in supported or source_lang == resolved:
            raise ValueError(f"Unsupported translation direction: {source_lang}_to_{resolved}")
        return resolved

    @staticmethod
    def _validate_cached_direction(
        cached: CachedPhrase,
        source_lang: str,
        target_lang: str,
    ) -> None:
        if (
            cached.source_lang != source_lang
            or cached.target_lang != target_lang
        ):
            raise RuntimeError(
                "Flash cache direction mismatch: "
                f"cached={cached.source_lang}_to_{cached.target_lang}, "
                f"requested={source_lang}_to_{target_lang}"
            )

    def _translation_safety(
        self,
        source: str,
        target: str,
        source_lang: str,
        target_lang: str,
    ):
        terminology = (
            self.mt_engine.lexicon.vi_to_en
            if source_lang == "vi"
            else self.mt_engine.lexicon.en_to_vi
        )
        return validate_translation(
            source,
            target,
            source_lang,
            target_lang,
            terminology,
        )

    @_serialized_runtime
    def load(self):
        """
        Load all pipeline components.
        This should be called once during initialization.
        """
        if self.get_status()["ready"]:
            logger.info("Pipeline already ready; skipping reload")
            return
        self._is_loaded = False
        logger.info("=" * 60)
        logger.info("MediVoice Edge — Loading Pipeline Components")
        logger.info("=" * 60)

        start = time.perf_counter()

        # Load each stage
        logger.info("[1/5] Loading Audio Frontend (VAD + Noise Suppression)...")
        self.audio_frontend.load()

        logger.info("[2/5] Loading ASR Engines (VI + EN)...")
        self.asr_engine.load()

        logger.info("[3/5] Loading Machine Translation Engine...")
        self.mt_engine.load()

        logger.info("[4/5] Loading TTS Engine...")
        self._tts_startup_degradation_code = None
        try:
            self.tts_engine.load()
        except (OSError, RuntimeError, ValueError) as exc:
            if not self.allow_text_only_tts_fallback:
                raise
            self._tts_startup_degradation_code = tts_degradation_code(exc)
            logger.error(
                "TTS startup unavailable; pipeline will operate text-only "
                "degradation_code=%s",
                self._tts_startup_degradation_code,
            )

        if self.flash_cache_enabled:
            logger.info("[5/5] Loading Flash Cache (Emergency Phrases)...")
            self.flash_cache.load()
        else:
            logger.info("[5/5] Flash Cache disabled by configuration")

        component_status = self._component_readiness()
        failed_components = self._failed_required_components(
            component_status
        )
        if failed_components:
            raise RuntimeError(
                "Pipeline component readiness check failed: "
                + ", ".join(failed_components)
            )
        self._is_loaded = True
        load_time = time.perf_counter() - start

        cache_stats = self.flash_cache.get_stats()
        logger.info("=" * 60)
        logger.info(f"Pipeline ready in {load_time:.1f}s")
        logger.info(f"Flash Cache: {cache_stats['total_phrases']} phrases loaded")
        logger.info("=" * 60)

    def _component_readiness(self) -> dict[str, bool]:
        return {
            "audio_frontend": self.audio_frontend.is_ready is True,
            "asr": self.asr_engine.is_ready is True,
            "mt": self.mt_engine.is_ready is True,
            "tts": self.tts_engine.is_ready is True,
            "flash_cache": self.flash_cache.is_ready is True,
        }

    def _failed_required_components(
        self,
        components: dict[str, bool],
    ) -> list[str]:
        return [
            name
            for name, ready in components.items()
            if not ready
            and not (name == "flash_cache" and not self.flash_cache_enabled)
            and not (name == "tts" and self.allow_text_only_tts_fallback)
        ]

    def get_status(self) -> dict:
        """Return verified startup readiness for every pipeline component."""
        components = self._component_readiness()
        if self._is_loaded is True and components["tts"]:
            operational_mode = "full"
            degradation_code = None
        elif self._is_loaded is True and self.allow_text_only_tts_fallback:
            operational_mode = "text_only"
            degradation_code = (
                self._tts_startup_degradation_code
                or "tts_runtime_error"
            )
        else:
            operational_mode = "unavailable"
            degradation_code = None
        required_ready = (
            components["audio_frontend"]
            and components["asr"]
            and components["mt"]
            and (components["tts"] or self.allow_text_only_tts_fallback)
            and (
                components["flash_cache"]
                or not self.flash_cache_enabled
            )
        )
        return {
            "ready": self._is_loaded is True and required_ready,
            "loaded": self._is_loaded is True,
            "flash_cache_enabled": self.flash_cache_enabled,
            "text_only_tts_fallback_enabled": self.allow_text_only_tts_fallback,
            "operational_mode": operational_mode,
            "degradation_code": degradation_code,
            "components": components,
        }

    def _require_ready(self) -> None:
        status = self.get_status()
        if status["ready"]:
            return
        failed_components = self._failed_required_components(
            status["components"]
        )
        if not status["loaded"] and not failed_components:
            failed_components = ["pipeline"]
        raise RuntimeError(
            "Pipeline is not ready; unavailable components: "
            + ", ".join(failed_components)
        )

    @_serialized_runtime
    def translate_speech(
        self,
        audio: np.ndarray,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        sample_rate: int = 16000,
        skip_tts: bool = False,
        audio_frontend_applied: bool = False,
    ) -> PipelineResult:
        """
        Complete speech-to-speech translation pipeline.

        Flow: Audio → ASR → Cache/MT → TTS → Audio

        Args:
            audio: Input audio (float32, normalized to [-1, 1])
            source_lang: Source language ("vi"/"en"). Auto-detect if None.
            target_lang: Target language. Auto-determined if None.
            sample_rate: Input audio sample rate
            skip_tts: If True, skip TTS synthesis (text-only output)
            audio_frontend_applied: True only for segments emitted by this
                pipeline's AudioFrontend, which have already been denoised.

        Returns:
            PipelineResult with all stage outputs and latency breakdown
        """
        self._require_ready()
        if type(audio_frontend_applied) is not bool:
            raise ValueError("audio_frontend_applied must be a boolean")
        if (
            audio_frontend_applied
            and sample_rate != self.audio_frontend.config.sample_rate
        ):
            raise ValueError(
                "Preprocessed audio sample_rate must match AudioFrontend"
            )

        if source_lang is not None:
            self._resolve_target_language(source_lang, target_lang)

        audio = validate_asr_audio_window(
            audio,
            sample_rate,
            self.asr_engine.max_input_duration_seconds,
        )

        pipeline_start = time.perf_counter()
        latency_breakdown = {}

        # ===== STAGE 1: Audio Frontend (Denoise) =====
        if audio_frontend_applied:
            clean_audio = audio
            latency_breakdown["audio_frontend_ms"] = 0.0
        else:
            stage_start = time.perf_counter()
            clean_audio = self.audio_frontend.denoiser.suppress(
                audio,
                sample_rate,
            )
            latency_breakdown["audio_frontend_ms"] = (
                time.perf_counter() - stage_start
            ) * 1000

        source_audio_duration = len(audio) / sample_rate

        # ===== STAGE 2: ASR (Speech → Text) =====
        stage_start = time.perf_counter()
        asr_result = self.asr_engine.transcribe(
            clean_audio,
            language=source_lang,
            sample_rate=sample_rate,
        )
        latency_breakdown["asr_ms"] = (time.perf_counter() - stage_start) * 1000

        if not asr_result.text.strip():
            logger.warning("ASR returned empty text. Skipping translation.")
            total_latency = (time.perf_counter() - pipeline_start) * 1000
            return PipelineResult(
                source_audio_duration_s=source_audio_duration,
                asr_text="",
                asr_language=asr_result.language,
                asr_confidence=0.0,
                asr_latency_ms=asr_result.latency_ms,
                translated_text="",
                target_language=target_lang or ("en" if asr_result.language == "vi" else "vi"),
                mt_latency_ms=0,
                from_cache=False,
                output_audio=None,
                output_sample_rate=22050,
                tts_latency_ms=0,
                tts_rtf=0,
                total_latency_ms=total_latency,
                overall_rtf=0,
                latency_breakdown=latency_breakdown,
                safety_passed=False,
                safety_issues=["empty_asr_transcript"],
            )

        # Determine target language
        target_lang = self._resolve_target_language(asr_result.language, target_lang)
        validate_mt_source_text(
            asr_result.text,
            max_source_characters=self.mt_engine.max_source_characters,
        )

        # ===== STAGE 2.5: Flash Cache Check =====
        stage_start = time.perf_counter()
        cached = (
            self.flash_cache.lookup(asr_result.text, asr_result.language)
            if self.flash_cache_enabled
            else None
        )
        latency_breakdown["cache_lookup_ms"] = (time.perf_counter() - stage_start) * 1000

        if cached:
            self._validate_cached_direction(
                cached,
                asr_result.language,
                target_lang,
            )
            # Cache HIT — bypass MT engine
            translated_text = cached.translated_text
            mt_latency = latency_breakdown["cache_lookup_ms"]
            from_cache = True
            requires_confirmation = cached.requires_confirmation
            logger.info(
                "Flash cache route selected source_lang=%s target_lang=%s "
                "source_chars=%d output_chars=%d",
                asr_result.language,
                target_lang,
                len(asr_result.text),
                len(translated_text),
            )

            # Use pre-synthesized audio if available
            if cached.audio is not None and not skip_tts:
                safety = self._translation_safety(
                    asr_result.text,
                    translated_text,
                    asr_result.language,
                    target_lang,
                )
                if not safety.safe:
                    logger.error(
                        "Translation safety guard blocked cached output "
                        "issue_codes=%s",
                        safety_issue_codes(safety.issues),
                    )
                    cached_audio = None
                elif requires_confirmation:
                    logger.warning("Cached clinical action requires confirmation; automatic TTS is suppressed.")
                    cached_audio = None
                else:
                    cached_audio = validate_tts_audio(
                        cached.audio,
                        cached.audio_sample_rate,
                    )
                total_latency = (time.perf_counter() - pipeline_start) * 1000
                self._record_translation_latency(
                    total_latency,
                    cache_hit=True,
                )
                return PipelineResult(
                    source_audio_duration_s=source_audio_duration,
                    asr_text=asr_result.text,
                    asr_language=asr_result.language,
                    asr_confidence=asr_result.confidence,
                    asr_latency_ms=asr_result.latency_ms,
                    translated_text=(translated_text if safety.safe else ""),
                    target_language=target_lang,
                    mt_latency_ms=mt_latency,
                    from_cache=True,
                    output_audio=cached_audio,
                    output_sample_rate=cached.audio_sample_rate,
                    tts_latency_ms=0,
                    tts_rtf=0,
                    total_latency_ms=total_latency,
                    overall_rtf=total_latency / 1000 / source_audio_duration if source_audio_duration > 0 else 0,
                    latency_breakdown=latency_breakdown,
                    safety_passed=safety.safe,
                    safety_issues=safety.issues,
                    requires_confirmation=requires_confirmation,
                )
        else:
            # Cache MISS — use MT engine
            # ===== STAGE 3: Machine Translation =====
            stage_start = time.perf_counter()
            mt_result = self.mt_engine.translate(
                asr_result.text,
                source_lang=asr_result.language,
                target_lang=target_lang,
            )
            latency_breakdown["mt_ms"] = (time.perf_counter() - stage_start) * 1000
            translated_text = mt_result.translated_text
            mt_latency = mt_result.latency_ms
            from_cache = False
            requires_confirmation = False

        # ===== STAGE 4: TTS (Text → Speech) =====
        output_audio = None
        output_sample_rate = self.tts_engine.output_sample_rate
        tts_latency = 0.0
        tts_rtf = 0.0
        degraded_mode = None
        degradation_code = None

        safety = self._translation_safety(
            asr_result.text,
            translated_text,
            asr_result.language,
            target_lang,
        )
        if not safety.safe:
            logger.error(
                "Translation safety guard blocked output issue_codes=%s",
                safety_issue_codes(safety.issues),
            )

        if requires_confirmation:
            logger.warning("Cached clinical action requires confirmation; automatic TTS is suppressed.")
        if not skip_tts and safety.safe and not requires_confirmation:
            if self.tts_engine.is_ready is not True:
                if not self.allow_text_only_tts_fallback:
                    raise RuntimeError("TTS engine is not ready")
                degraded_mode = "text_only"
                degradation_code = (
                    self._tts_startup_degradation_code
                    or "tts_runtime_error"
                )
                logger.error(
                    "TTS not ready; preserving safety-passed text-only result "
                    "degradation_code=%s",
                    degradation_code,
                )
            else:
                stage_start = time.perf_counter()
                try:
                    tts_result = self.tts_engine.synthesize(
                        translated_text,
                        language=target_lang,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    latency_breakdown["tts_ms"] = (
                        time.perf_counter() - stage_start
                    ) * 1000
                    if not self.allow_text_only_tts_fallback:
                        raise
                    degraded_mode = "text_only"
                    degradation_code = tts_degradation_code(exc)
                    logger.error(
                        "TTS unavailable; preserving safety-passed text-only result "
                        "degradation_code=%s",
                        degradation_code,
                    )
                else:
                    latency_breakdown["tts_ms"] = (
                        time.perf_counter() - stage_start
                    ) * 1000
                    output_audio = tts_result.audio
                    output_sample_rate = tts_result.sample_rate
                    tts_latency = tts_result.latency_ms
                    tts_rtf = tts_result.rtf

        # ===== FINAL: Compute totals =====
        total_latency = (time.perf_counter() - pipeline_start) * 1000
        overall_rtf = (total_latency / 1000) / source_audio_duration if source_audio_duration > 0 else 0

        self._record_translation_latency(
            total_latency,
            cache_hit=from_cache,
        )
        if degraded_mode == "text_only":
            self._text_only_tts_fallbacks += 1

        result = PipelineResult(
            source_audio_duration_s=source_audio_duration,
            asr_text=asr_result.text,
            asr_language=asr_result.language,
            asr_confidence=asr_result.confidence,
            asr_latency_ms=asr_result.latency_ms,
            translated_text=(translated_text if safety.safe else ""),
            target_language=target_lang,
            mt_latency_ms=mt_latency,
            from_cache=from_cache,
            output_audio=output_audio,
            output_sample_rate=output_sample_rate,
            tts_latency_ms=tts_latency,
            tts_rtf=tts_rtf,
            total_latency_ms=total_latency,
            overall_rtf=overall_rtf,
            latency_breakdown=latency_breakdown,
            safety_passed=safety.safe,
            safety_issues=safety.issues,
            requires_confirmation=requires_confirmation,
            degraded_mode=degraded_mode,
            degradation_code=degradation_code,
        )

        # Log performance summary
        logger.info(
            "Translation complete route=%s source_lang=%s target_lang=%s "
            "source_chars=%d output_chars=%d safety_passed=%s "
            "requires_confirmation=%s degraded_mode=%s degradation_code=%s "
            "total_latency_ms=%.0f rtf=%.2f "
            "frontend_ms=%.0f asr_ms=%.0f mt_ms=%.0f tts_ms=%.0f",
            "cache" if from_cache else "model",
            asr_result.language,
            target_lang,
            len(asr_result.text),
            len(translated_text),
            safety.safe,
            requires_confirmation,
            degraded_mode,
            degradation_code,
            total_latency,
            overall_rtf,
            latency_breakdown.get("audio_frontend_ms", 0),
            asr_result.latency_ms,
            mt_latency,
            tts_latency,
        )

        return result

    def _validate_playback_result(
        self,
        result: PipelineResult,
        *,
        confirmation_required: bool,
    ) -> None:
        """Revalidate text safety and confirmation state at the audio sink."""
        if result.safety_passed is not True:
            raise RuntimeError(
                "Refusing playback: translation failed clinical safety gate"
            )
        expected_confirmation = True if confirmation_required else False
        if result.requires_confirmation is not expected_confirmation:
            raise RuntimeError(
                "Refusing playback: explicit confirmation state is invalid"
            )
        try:
            source_text = result.asr_text
            translated_text = result.translated_text
            source_language = result.asr_language
            target_language = result.target_language
            recorded_issues = result.safety_issues
        except AttributeError as exc:
            raise TypeError("Playback requires a complete PipelineResult") from exc
        if (
            not isinstance(source_text, str)
            or not isinstance(translated_text, str)
            or not isinstance(source_language, str)
            or not isinstance(target_language, str)
            or not isinstance(recorded_issues, list)
        ):
            raise TypeError("Playback requires a complete PipelineResult")
        safety = self._translation_safety(
            source_text,
            translated_text,
            source_language,
            target_language,
        )
        if not safety.safe or recorded_issues != safety.issues:
            raise RuntimeError(
                "Refusing playback: translation no longer matches clinical safety evidence"
            )

    @_serialized_runtime
    def play(self, result: PipelineResult):
        """Play an automatically approved translated audio output."""
        self._validate_playback_result(result, confirmation_required=False)
        if result.output_audio is not None:
            self.tts_engine.play_audio(result.output_audio, result.output_sample_rate)
        else:
            logger.warning("No audio to play.")

    @_serialized_runtime
    def confirm_and_play(
        self,
        result: PipelineResult,
        *,
        confirmed: bool,
    ) -> TTSResult:
        """Synthesize and play a reviewed cache action after explicit consent."""
        if confirmed is not True:
            raise RuntimeError("Refusing playback: explicit confirmation is required")
        self._validate_playback_result(result, confirmation_required=True)
        if self.tts_engine.is_ready is not True:
            raise RuntimeError("Refusing playback: TTS engine is not ready")
        if result.output_audio is not None:
            raise RuntimeError(
                "Refusing playback: confirmation-gated result already contains audio"
            )
        synthesized = self.tts_engine.synthesize(
            result.translated_text,
            language=result.target_language,
        )
        audio = validate_tts_audio(synthesized.audio, synthesized.sample_rate)
        self.tts_engine.play_audio(audio, synthesized.sample_rate)
        return synthesized

    @_serialized_runtime
    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: Optional[str] = None,
    ) -> MTResult:
        """
        Text-only translation (no ASR/TTS).
        Useful for testing the MT engine directly.
        """
        self._require_ready()

        target_lang = self._resolve_target_language(source_lang, target_lang)
        validate_mt_source_text(
            text,
            max_source_characters=self.mt_engine.max_source_characters,
        )

        # Check flash cache first
        cached = (
            self.flash_cache.lookup(text, source_lang)
            if self.flash_cache_enabled
            else None
        )
        if cached:
            self._validate_cached_direction(cached, source_lang, target_lang)
            result = MTResult(
                source_text=text,
                translated_text=cached.translated_text,
                source_lang=source_lang,
                target_lang=cached.target_lang,
                latency_ms=0.1,
                first_token_ms=None,
                tokens_generated=0,
                from_cache=True,
                requires_confirmation=cached.requires_confirmation,
            )
        else:
            result = self.mt_engine.translate(text, source_lang, target_lang)

        safety = self._translation_safety(
            text,
            result.translated_text,
            source_lang,
            target_lang,
        )
        result.safety_passed = safety.safe
        result.safety_issues = safety.issues
        if not safety.safe:
            result.translated_text = ""
        return result

    @_exclusive_interactive_session
    def run_interactive(
        self,
        on_result: Optional[Callable[[PipelineResult], None]] = None,
    ) -> None:
        """
        Run the pipeline in interactive mode with microphone input.
        Press Ctrl+C to stop.
        """
        self._require_ready()
        if on_result is not None and not callable(on_result):
            raise ValueError("on_result must be callable")

        logger.info("\n🎤 MediVoice Edge — Interactive Mode")
        logger.info("Speak into the microphone. Press Ctrl+C to stop.\n")

        for speech_segment in self.audio_frontend.stream_from_microphone():
            result = self.translate_speech(
                speech_segment,
                sample_rate=self.audio_frontend.config.sample_rate,
                audio_frontend_applied=True,
            )
            if on_result is not None:
                on_result(result)
            if result.output_audio is not None:
                self.play(result)

    @_serialized_runtime
    def get_performance_stats(self) -> Dict:
        """Get pipeline performance statistics."""
        latency_samples = len(self._latency_history)
        stats = {
            "total_translations": self._total_translations,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._total_translations),
            "text_only_tts_fallbacks": self._text_only_tts_fallbacks,
            "text_only_tts_fallback_rate": (
                self._text_only_tts_fallbacks
                / max(1, self._total_translations)
            ),
            "latency_window_samples": latency_samples,
            "latency_window_capacity": MAX_LATENCY_HISTORY_SAMPLES,
            "latency_samples_dropped": max(
                0,
                self._total_translations - latency_samples,
            ),
        }
        if self._latency_history:
            ordered = sorted(self._latency_history)
            stats["avg_latency_ms"] = sum(ordered) / latency_samples
            stats["min_latency_ms"] = ordered[0]
            stats["max_latency_ms"] = ordered[-1]
            stats["p95_latency_ms"] = ordered[
                math.ceil(0.95 * latency_samples) - 1
            ]
        return stats

    def _record_translation_latency(
        self,
        latency_ms: float,
        *,
        cache_hit: bool = False,
    ) -> None:
        if (
            isinstance(latency_ms, bool)
            or not isinstance(latency_ms, (int, float))
            or not math.isfinite(float(latency_ms))
            or float(latency_ms) < 0.0
        ):
            raise RuntimeError("Translation latency must be finite and non-negative")
        if type(cache_hit) is not bool:
            raise RuntimeError("cache_hit telemetry flag must be a boolean")
        self._total_translations += 1
        self._cache_hits += int(cache_hit)
        self._latency_history.append(float(latency_ms))

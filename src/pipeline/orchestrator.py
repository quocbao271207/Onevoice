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
import time
import yaml
import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass, field
from pathlib import Path

from .audio_frontend import AudioFrontend, AudioConfig
from .asr_engine import ASREngine, ASRResult, validate_asr_audio_window
from .mt_engine import MTEngine, MTResult, validate_mt_source_text
from .tts_engine import TTSEngine, TTSResult, validate_tts_audio
from .flash_cache import CachedPhrase, FlashCache
from .safety_guard import validate_translation

logger = logging.getLogger(__name__)


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
        self.config = self._load_config(config_path)

        # Initialize pipeline stages
        audio_cfg = AudioConfig(
            sample_rate=self.config.get("audio", {}).get("sample_rate", 16000),
            chunk_duration_ms=self.config.get("audio", {}).get("chunk_duration_ms", 500),
            vad_threshold=self.config.get("audio", {}).get("vad_threshold", 0.5),
            silence_duration_ms=self.config.get("audio", {}).get("silence_duration_ms", 800),
        )

        self.audio_frontend = AudioFrontend(config=audio_cfg)

        runtime_cfg = self.config.get("runtime", {})
        allow_base_fallback = bool(runtime_cfg.get("allow_base_model_fallback", False))

        asr_cfg = self.config.get("asr", {})
        self.asr_engine = ASREngine(
            vi_model_path=asr_cfg.get("vi", {}).get("model_path", "models/asr/phowhisper-small-medical"),
            en_model_path=asr_cfg.get("en", {}).get("model_path", "models/asr/distil-whisper-en"),
            allow_base_fallback=allow_base_fallback,
            max_input_duration_seconds=asr_cfg.get(
                "max_input_duration_seconds",
                30.0,
            ),
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
        self.flash_cache = FlashCache(
            cache_path=flash_cfg.get("cache_path"),
            allow_fuzzy=bool(flash_cfg.get("fuzzy_matching", False)),
        )

        self._is_loaded = False

        # Performance tracking
        self._total_translations = 0
        self._cache_hits = 0
        self._latency_history = []

    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load pipeline configuration from YAML file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Pipeline config loaded from: {config_path}")
            return config
        else:
            logger.warning("No config file found. Using defaults.")
            return {}

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

    def load(self):
        """
        Load all pipeline components.
        This should be called once during initialization.
        """
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
        self.tts_engine.load()

        logger.info("[5/5] Loading Flash Cache (Emergency Phrases)...")
        self.flash_cache.load()

        self._is_loaded = True
        load_time = time.perf_counter() - start

        cache_stats = self.flash_cache.get_stats()
        logger.info("=" * 60)
        logger.info(f"Pipeline ready in {load_time:.1f}s")
        logger.info(f"Flash Cache: {cache_stats['total_phrases']} phrases loaded")
        logger.info("=" * 60)

    def translate_speech(
        self,
        audio: np.ndarray,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        sample_rate: int = 16000,
        skip_tts: bool = False,
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

        Returns:
            PipelineResult with all stage outputs and latency breakdown
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

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
        stage_start = time.perf_counter()
        # In streaming mode, audio_frontend would chunk and denoise
        # For batch mode, we just denoise the full segment
        clean_audio = self.audio_frontend.denoiser.suppress(audio, sample_rate)
        latency_breakdown["audio_frontend_ms"] = (time.perf_counter() - stage_start) * 1000

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
            )

        # Determine target language
        target_lang = self._resolve_target_language(asr_result.language, target_lang)
        validate_mt_source_text(
            asr_result.text,
            max_source_characters=self.mt_engine.max_source_characters,
        )

        # ===== STAGE 2.5: Flash Cache Check =====
        stage_start = time.perf_counter()
        cached = self.flash_cache.lookup(asr_result.text, asr_result.language)
        latency_breakdown["cache_lookup_ms"] = (time.perf_counter() - stage_start) * 1000

        if cached:
            self._validate_cached_direction(
                cached,
                asr_result.language,
                target_lang,
            )
            # Cache HIT — bypass MT engine
            self._cache_hits += 1
            translated_text = cached.translated_text
            mt_latency = latency_breakdown["cache_lookup_ms"]
            from_cache = True
            requires_confirmation = cached.requires_confirmation
            logger.info(f"⚡ Flash Cache HIT: \"{asr_result.text}\" → \"{translated_text}\"")

            # Use pre-synthesized audio if available
            if cached.audio is not None and not skip_tts:
                safety = self._translation_safety(
                    asr_result.text,
                    translated_text,
                    asr_result.language,
                    target_lang,
                )
                if not safety.safe:
                    logger.error("Translation safety guard blocked cached TTS: %s", safety.issues)
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
                self._total_translations += 1
                return PipelineResult(
                    source_audio_duration_s=source_audio_duration,
                    asr_text=asr_result.text,
                    asr_language=asr_result.language,
                    asr_confidence=asr_result.confidence,
                    asr_latency_ms=asr_result.latency_ms,
                    translated_text=translated_text,
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

        safety = self._translation_safety(
            asr_result.text,
            translated_text,
            asr_result.language,
            target_lang,
        )
        if not safety.safe:
            logger.error("Translation safety guard blocked TTS: %s", safety.issues)

        if requires_confirmation:
            logger.warning("Cached clinical action requires confirmation; automatic TTS is suppressed.")
        if not skip_tts and safety.safe and not requires_confirmation:
            stage_start = time.perf_counter()
            tts_result = self.tts_engine.synthesize(
                translated_text,
                language=target_lang,
            )
            latency_breakdown["tts_ms"] = (time.perf_counter() - stage_start) * 1000
            output_audio = tts_result.audio
            output_sample_rate = tts_result.sample_rate
            tts_latency = tts_result.latency_ms
            tts_rtf = tts_result.rtf

        # ===== FINAL: Compute totals =====
        total_latency = (time.perf_counter() - pipeline_start) * 1000
        overall_rtf = (total_latency / 1000) / source_audio_duration if source_audio_duration > 0 else 0

        self._total_translations += 1
        self._latency_history.append(total_latency)

        result = PipelineResult(
            source_audio_duration_s=source_audio_duration,
            asr_text=asr_result.text,
            asr_language=asr_result.language,
            asr_confidence=asr_result.confidence,
            asr_latency_ms=asr_result.latency_ms,
            translated_text=translated_text,
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
        )

        # Log performance summary
        cache_label = "⚡CACHE" if from_cache else "🧠AI"
        logger.info(
            f"\n{'='*60}\n"
            f"  [{cache_label}] Translation Complete\n"
            f"  {asr_result.language.upper()} → {target_lang.upper()}\n"
            f"  Input:  \"{asr_result.text}\"\n"
            f"  Output: \"{translated_text}\"\n"
            f"  Latency: {total_latency:.0f}ms (RTF={overall_rtf:.2f})\n"
            f"  Breakdown: Frontend={latency_breakdown.get('audio_frontend_ms', 0):.0f}ms | "
            f"ASR={asr_result.latency_ms:.0f}ms | "
            f"MT={mt_latency:.0f}ms | "
            f"TTS={tts_latency:.0f}ms\n"
            f"{'='*60}"
        )

        return result

    def play(self, result: PipelineResult):
        """Play the translated audio output."""
        if result.output_audio is not None:
            self.tts_engine.play_audio(result.output_audio, result.output_sample_rate)
        else:
            logger.warning("No audio to play.")

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
        if not self._is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        target_lang = self._resolve_target_language(source_lang, target_lang)
        validate_mt_source_text(
            text,
            max_source_characters=self.mt_engine.max_source_characters,
        )

        # Check flash cache first
        cached = self.flash_cache.lookup(text, source_lang)
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
        return result

    def run_interactive(self):
        """
        Run the pipeline in interactive mode with microphone input.
        Press Ctrl+C to stop.
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        logger.info("\n🎤 MediVoice Edge — Interactive Mode")
        logger.info("Speak into the microphone. Press Ctrl+C to stop.\n")

        try:
            for speech_segment in self.audio_frontend.stream_from_microphone():
                result = self.translate_speech(speech_segment)
                if result.output_audio is not None:
                    self.play(result)
        except KeyboardInterrupt:
            logger.info("\nInteractive mode stopped.")

    def get_performance_stats(self) -> Dict:
        """Get pipeline performance statistics."""
        stats = {
            "total_translations": self._total_translations,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._total_translations),
        }
        if self._latency_history:
            stats["avg_latency_ms"] = sum(self._latency_history) / len(self._latency_history)
            stats["min_latency_ms"] = min(self._latency_history)
            stats["max_latency_ms"] = max(self._latency_history)
            stats["p95_latency_ms"] = sorted(self._latency_history)[
                int(0.95 * len(self._latency_history))
            ]
        return stats

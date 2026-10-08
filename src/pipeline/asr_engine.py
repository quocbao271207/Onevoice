"""
MediVoice Edge — ASR Engine Module
Automatic Speech Recognition for Vietnamese and English medical speech.

Pipeline Stage 2: Clean Audio → Transcribed Text

Models:
- Vietnamese: PhoWhisper-small fine-tuned on audited VietMed + ViMedCSS train data
- English: Distil-Whisper-Small.en; Eka remains evaluation-only

This module currently performs utterance-level PyTorch inference. Streaming and
QNN deployment remain separate, measured milestones.
"""

import logging
import math
import time
import numpy as np
from math import gcd
from numbers import Integral, Real
from typing import Optional, Dict
from dataclasses import dataclass

from .audio_frontend import validate_audio_input
from .generation_guard import require_completed_generation

logger = logging.getLogger(__name__)

BASE_MODEL_REVISIONS = {
    "vinai/PhoWhisper-small": "a86b604c346caf7148c37512eafe783a16420adb",
    "distil-whisper/distil-small.en": "9e4a67ca4569c30be43a3fe7fba1621e504f0093",
}
SUPPORTED_ASR_LANGUAGES = {"vi", "en"}
WHISPER_MAX_INPUT_DURATION_SECONDS = 30.0
DEFAULT_ASR_MAX_NEW_TOKENS = 225


def validate_asr_audio_window(
    audio: np.ndarray,
    sample_rate: int,
    max_duration_seconds: float,
) -> np.ndarray:
    """Validate audio and reject input beyond one Whisper feature window."""
    value = validate_audio_input(audio, sample_rate)
    duration_seconds = len(value) / sample_rate
    if duration_seconds > max_duration_seconds:
        raise ValueError(
            f"ASR audio has {len(value)} samples ({duration_seconds:.3f}s); "
            f"limit is {max_duration_seconds:.3f}s. "
            "Refusing silent Whisper truncation."
        )
    return value


@dataclass
class ASRResult:
    """Result from ASR inference."""
    text: str
    language: str           # "vi" or "en"
    confidence: float       # uncalibrated sequence token probability proxy
    latency_ms: float       # Inference time in milliseconds
    is_code_switched: bool  # Contains mixed VI/EN


class ASREngine:
    """
    Automatic Speech Recognition engine with dual-language support.

    Vietnamese uses multilingual PhoWhisper; English uses Distil-Whisper.en.
    The returned confidence is explicitly not a calibrated clinical score.
    """

    def __init__(
        self,
        vi_model_path: str = "models/asr/phowhisper-small-medical",
        en_model_path: str = "models/asr/distil-whisper-en",
        device: str = "auto",
        use_onnx: bool = False,
        languages: tuple[str, ...] = ("vi", "en"),
        allow_base_fallback: bool = False,
        max_input_duration_seconds: float = WHISPER_MAX_INPUT_DURATION_SECONDS,
        max_new_tokens: int = DEFAULT_ASR_MAX_NEW_TOKENS,
    ):
        if (
            isinstance(max_input_duration_seconds, bool)
            or not isinstance(max_input_duration_seconds, Real)
            or not math.isfinite(float(max_input_duration_seconds))
            or not (
                0.0
                < float(max_input_duration_seconds)
                <= WHISPER_MAX_INPUT_DURATION_SECONDS
            )
        ):
            raise ValueError(
                "max_input_duration_seconds must be finite and in "
                f"(0, {WHISPER_MAX_INPUT_DURATION_SECONDS}]"
            )
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, Integral)
            or int(max_new_tokens) <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if (
            not isinstance(languages, (tuple, list))
            or not languages
            or any(
                not isinstance(language, str)
                or language not in SUPPORTED_ASR_LANGUAGES
                for language in languages
            )
            or len(set(languages)) != len(languages)
        ):
            raise ValueError("languages must be unique values from {'vi', 'en'}")
        self.vi_model_path = vi_model_path
        self.en_model_path = en_model_path
        self.device = device
        self.use_onnx = use_onnx
        self.languages = tuple(languages)
        self.allow_base_fallback = allow_base_fallback
        self.max_input_duration_seconds = float(max_input_duration_seconds)
        self.max_new_tokens = int(max_new_tokens)
        self.models: Dict[str, object] = {}
        self.processors: Dict[str, object] = {}
        self.loaded_model_paths: Dict[str, str] = {}
        self._is_loaded = False

    @staticmethod
    def _to_whisper_rate(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Whisper feature extractors require mono 16 kHz input."""
        value = validate_audio_input(audio, sample_rate)
        if sample_rate == 16000:
            return value, sample_rate
        from scipy.signal import resample_poly

        common = gcd(int(sample_rate), 16000)
        return resample_poly(value, 16000 // common, int(sample_rate) // common).astype(np.float32), 16000

    def _validated_whisper_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, int]:
        value = validate_asr_audio_window(
            audio,
            sample_rate,
            self.max_input_duration_seconds,
        )
        if sample_rate == 16000:
            return value, sample_rate
        from scipy.signal import resample_poly

        common = gcd(int(sample_rate), 16000)
        resampled = resample_poly(
            value,
            16000 // common,
            int(sample_rate) // common,
        ).astype(np.float32)
        return resampled, 16000

    def load(self):
        """Load ASR models for both languages."""
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading ASR models on device: {self.device}")

        # Try loading fine-tuned models, fall back to base models
        for lang, model_path, fallback in [
            ("vi", self.vi_model_path, "vinai/PhoWhisper-small"),
            ("en", self.en_model_path, "distil-whisper/distil-small.en"),
        ]:
            if lang not in self.languages:
                continue
            try:
                logger.info(f"Loading ASR [{lang}] from: {model_path}")
                revision = BASE_MODEL_REVISIONS.get(model_path)
                self.processors[lang] = AutoProcessor.from_pretrained(model_path, revision=revision)
                self.models[lang] = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_path,
                    revision=revision,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                ).to(self.device)
                self.models[lang].eval()
                self.loaded_model_paths[lang] = model_path
            except Exception as e:
                if not self.allow_base_fallback:
                    raise RuntimeError(
                        f"Failed to load configured ASR checkpoint for {lang}: {model_path}. "
                        "Base fallback is disabled to prevent silently evaluating the wrong model."
                    ) from e
                logger.warning(
                    f"Failed to load fine-tuned ASR [{lang}]: {e}. "
                    f"Falling back to: {fallback}"
                )
                revision = BASE_MODEL_REVISIONS[fallback]
                self.processors[lang] = AutoProcessor.from_pretrained(fallback, revision=revision)
                self.models[lang] = AutoModelForSpeechSeq2Seq.from_pretrained(
                    fallback,
                    revision=revision,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                ).to(self.device)
                self.models[lang].eval()
                self.loaded_model_paths[lang] = fallback

        self._is_loaded = True
        logger.info("ASR engines loaded successfully (VI + EN)")

    def detect_language(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Detect whether the audio contains Vietnamese or English speech.

        Uses Whisper's built-in language detection capability.
        Fails closed if detection cannot be completed reliably.

        Args:
            audio: Audio signal (float32, normalized)
            sample_rate: Sample rate in Hz

        Returns:
            Language code: "vi" or "en"
        """
        audio, sample_rate = self._validated_whisper_audio(audio, sample_rate)
        if not self._is_loaded:
            raise RuntimeError("ASR engine not loaded. Call load() first.")

        try:
            import torch
            # Language detection requires a multilingual Whisper checkpoint;
            # the English-only Distil-Whisper model cannot provide VI logits.
            processor = self.processors.get("vi", list(self.processors.values())[0])
            model = self.models.get("vi", list(self.models.values())[0])

            processed = processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = processed.input_features.to(self.device)

            with torch.no_grad():
                # Use Whisper's language detection
                decoder_input_ids = torch.tensor(
                    [[model.config.decoder_start_token_id]]
                ).to(self.device)
                outputs = model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids
                )
                # Get language token probabilities
                logits = outputs.logits[0, 0]

            # Check if Vietnamese or English has higher probability
            # Whisper language tokens: <|vi|> and <|en|>
            tokenizer = processor.tokenizer
            vi_token = tokenizer.convert_tokens_to_ids("<|vi|>")
            en_token = tokenizer.convert_tokens_to_ids("<|en|>")

            if vi_token is None or en_token is None or vi_token == en_token:
                raise RuntimeError(
                    "Whisper tokenizer lacks distinct VI/EN language tokens"
                )
            vi_logit = float(logits[vi_token].item())
            en_logit = float(logits[en_token].item())
            if not math.isfinite(vi_logit) or not math.isfinite(en_logit):
                raise RuntimeError("Whisper returned non-finite language logits")
            if vi_logit == en_logit:
                raise RuntimeError("Whisper language detection is ambiguous")
            detected = "vi" if vi_logit > en_logit else "en"
            logger.debug(
                "Language detection: VI=%.3f, EN=%.3f → %s",
                vi_logit,
                en_logit,
                detected,
            )
            return detected
        except Exception as exc:
            raise RuntimeError(
                "Language detection failed; refusing implicit fallback"
            ) from exc

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        sample_rate: int = 16000
    ) -> ASRResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio signal (float32, normalized to [-1, 1])
            language: Force language ("vi" or "en"). Auto-detect if None.
            sample_rate: Sample rate in Hz

        Returns:
            ASRResult with transcribed text and metadata
        """
        if not self._is_loaded:
            raise RuntimeError("ASR engine not loaded. Call load() first.")

        start_time = time.perf_counter()

        audio, sample_rate = self._validated_whisper_audio(audio, sample_rate)

        # Step 1: Detect language if not specified
        if language is None:
            language = self.detect_language(audio, sample_rate)
        if (
            language not in self.languages
            or language not in self.models
            or language not in self.processors
        ):
            raise ValueError(f"Unsupported ASR language: {language}")

        # Step 2: Select appropriate model
        model = self.models[language]
        processor = self.processors[language]

        # Step 3: Preprocess audio
        import torch
        processed = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = processed.input_features.to(self.device)

        # Step 4: Generate transcription
        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "num_beams": 1,           # Greedy for speed
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            if "attention_mask" in processed:
                generate_kwargs["attention_mask"] = processed.attention_mask.to(self.device)

            # Set language for Whisper
            if hasattr(processor, 'tokenizer'):
                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language=language, task="transcribe"
                )
                if forced_decoder_ids:
                    generate_kwargs["forced_decoder_ids"] = forced_decoder_ids

            outputs = model.generate(input_features, **generate_kwargs)

        # Step 5: Decode tokens to text
        require_completed_generation(
            outputs.sequences[0],
            processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            context="ASR generation",
        )
        transcription = processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )[0].strip()

        # Step 6: Calculate confidence (average token probability)
        confidence = 0.0
        if hasattr(outputs, 'scores') and outputs.scores:
            transitions = model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True,
            )
            if transitions.numel():
                confidence = float(transitions.mean().exp().item())

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Step 7: Detect code-switching
        is_code_switched = self._detect_code_switching(transcription, language)

        result = ASRResult(
            text=transcription,
            language=language,
            confidence=confidence,
            latency_ms=latency_ms,
            is_code_switched=is_code_switched
        )

        logger.info(
            f"ASR [{language}]: \"{transcription[:80]}...\" "
            f"(conf={confidence:.2f}, latency={latency_ms:.0f}ms)"
        )
        return result

    def _detect_code_switching(self, text: str, primary_language: str) -> bool:
        """
        Detect if transcribed text contains code-switching.
        E.g., Vietnamese text with English medical terms mixed in.
        """
        if primary_language == "vi":
            # Check for English words in Vietnamese text
            english_medical_terms = [
                "test", "PCR", "CT", "MRI", "ECG", "ICU", "shock",
                "virus", "COVID", "vaccine", "paracetamol", "aspirin",
                "intubation", "ventilator", "monitor", "catheter",
                "stent", "bypass", "dialysis", "biopsy"
            ]
            text_lower = text.lower()
            return any(term.lower() in text_lower for term in english_medical_terms)
        return False

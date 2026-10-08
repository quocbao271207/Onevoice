"""Offline Piper TTS with explicit, fail-closed model loading."""

from __future__ import annotations

import logging
import math
import time
import unicodedata
from dataclasses import dataclass
from math import gcd
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from .audio_frontend import MAX_SAMPLE_RATE, MIN_SAMPLE_RATE, validate_audio_input

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    audio: np.ndarray
    sample_rate: int
    duration_s: float
    latency_ms: float
    rtf: float
    language: str


DRUG_G2P_VI = {
    "paracetamol": "pa ra xê ta môn",
    "panadol": "pa na đôn",
    "aspirin": "a xpi rin",
    "amoxicillin": "a mốc xi xi lin",
    "ibuprofen": "ai biu prô fen",
    "metformin": "mét pho min",
    "omeprazole": "ô mê pra dôn",
    "atorvastatin": "a to va xta tin",
    "amlodipine": "am lô đi pin",
    "losartan": "lô xa tan",
    "clopidogrel": "clô pi đô grel",
    "warfarin": "oa pha rin",
    "insulin": "in xu lin",
    "morphine": "moóc phin",
    "fentanyl": "phen ta nil",
    "lidocaine": "li đô ca in",
    "epinephrine": "ê pi nê phrin",
    "norepinephrine": "no ê pi nê phrin",
    "dopamine": "đô pa min",
    "dobutamine": "đô bu ta min",
}

DEFAULT_MAX_TTS_TEXT_CHARACTERS = 4096
DEFAULT_MAX_TTS_DURATION_SECONDS = 120.0


def validate_tts_sample_rate(sample_rate: int, *, name: str = "sample_rate") -> int:
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, Integral)
        or not MIN_SAMPLE_RATE <= int(sample_rate) <= MAX_SAMPLE_RATE
    ):
        raise ValueError(
            f"{name} must be an integer in [{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}]"
        )
    return int(sample_rate)


def validate_tts_text(text: str, max_text_characters: int) -> None:
    if not isinstance(text, str):
        raise TypeError("TTS text must be a string")
    if not text.strip():
        raise ValueError("TTS text must not be blank")
    if any(unicodedata.category(character) == "Cc" for character in text):
        raise ValueError("TTS text must not contain control characters")
    if len(text) > max_text_characters:
        raise ValueError(
            f"TTS text has {len(text)} characters; limit is {max_text_characters}"
        )


def validate_tts_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Validate synthesized or cached audio before it can reach a sink."""
    try:
        value = np.asarray(audio, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("TTS audio must be a numeric array") from exc
    if value.ndim != 1:
        raise ValueError("TTS audio must be one-dimensional mono audio")
    return validate_audio_input(value, validate_tts_sample_rate(sample_rate))


class TTSEngine:
    """Load one Vietnamese and one English Piper voice and cache both sessions."""

    def __init__(
        self,
        vi_model_path: str = "models/tts/vi_VN-vivos-x_low.onnx",
        en_model_path: str = "models/tts/en_US-lessac-medium.onnx",
        output_sample_rate: int | None = None,
        apply_medical_aliases: bool = False,
        max_text_characters: int = DEFAULT_MAX_TTS_TEXT_CHARACTERS,
        max_duration_seconds: float = DEFAULT_MAX_TTS_DURATION_SECONDS,
    ):
        if output_sample_rate is not None:
            output_sample_rate = validate_tts_sample_rate(
                output_sample_rate,
                name="output_sample_rate",
            )
        if (
            isinstance(max_text_characters, bool)
            or not isinstance(max_text_characters, Integral)
        ):
            raise ValueError("max_text_characters must be a positive integer")
        if int(max_text_characters) <= 0:
            raise ValueError("max_text_characters must be a positive integer")
        if (
            isinstance(max_duration_seconds, bool)
            or not isinstance(max_duration_seconds, Real)
            or not math.isfinite(float(max_duration_seconds))
            or float(max_duration_seconds) <= 0.0
        ):
            raise ValueError("max_duration_seconds must be finite and positive")
        self.model_paths = {"vi": Path(vi_model_path), "en": Path(en_model_path)}
        self.requested_sample_rate = output_sample_rate
        self.output_sample_rate = output_sample_rate or 22050
        self.apply_medical_aliases = apply_medical_aliases
        self.max_text_characters = int(max_text_characters)
        self.max_duration_seconds = float(max_duration_seconds)
        self._voices: dict[str, object] = {}
        self._is_loaded = False

    def load(self) -> None:
        """Fail immediately if Piper or either checked-in model artifact is absent."""
        try:
            from piper.voice import PiperVoice
        except ImportError as exc:
            raise RuntimeError("Install piper-tts from requirements-local.txt") from exc

        for language, model_path in self.model_paths.items():
            config_path = Path(f"{model_path}.json")
            missing = [str(path) for path in (model_path, config_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Missing Piper {language} artifact(s): {missing}. "
                    "Run scripts/download_tts_models.py first."
                )
            self._voices[language] = PiperVoice.load(model_path, config_path=config_path)

        self._is_loaded = True
        logger.info("Loaded Piper voices: %s", {key: str(value) for key, value in self.model_paths.items()})

    @staticmethod
    def _apply_g2p(text: str, language: str) -> str:
        if language != "vi":
            return text
        result = text
        lower = result.casefold()
        for drug, phonetic in DRUG_G2P_VI.items():
            if drug in lower:
                result = result.replace(drug, phonetic).replace(drug.capitalize(), phonetic)
                lower = result.casefold()
        return result

    @staticmethod
    def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return audio.astype(np.float32, copy=False)
        from scipy.signal import resample_poly

        common = gcd(source_rate, target_rate)
        return resample_poly(audio, target_rate // common, source_rate // common).astype(np.float32)

    def synthesize(self, text: str, language: str = "vi") -> TTSResult:
        if not self._is_loaded:
            raise RuntimeError("TTS engine is not loaded; call load() first")
        if language not in self._voices:
            raise ValueError(f"Unsupported TTS language: {language}")
        validate_tts_text(text, self.max_text_characters)

        start = time.perf_counter()
        processed_text = self._apply_g2p(text, language) if self.apply_medical_aliases else text
        audio_chunks = []
        source_rate = None
        source_samples = 0
        chunks = self._voices[language].synthesize(processed_text)
        for index, chunk in enumerate(chunks):
            if not hasattr(chunk, "sample_rate") or not hasattr(
                chunk,
                "audio_float_array",
            ):
                raise RuntimeError(f"Piper returned malformed audio chunk {index}")
            chunk_rate = validate_tts_sample_rate(
                chunk.sample_rate,
                name=f"Piper chunk {index} sample_rate",
            )
            if source_rate is None:
                source_rate = chunk_rate
            elif chunk_rate != source_rate:
                raise RuntimeError(
                    "Piper returned inconsistent sample rates across chunks"
                )
            chunk_audio = validate_tts_audio(chunk.audio_float_array, chunk_rate)
            source_samples += len(chunk_audio)
            if source_samples / source_rate > self.max_duration_seconds:
                raise RuntimeError(
                    "TTS audio duration exceeds limit while consuming Piper chunks"
                )
            audio_chunks.append(chunk_audio)
        if source_rate is None:
            raise RuntimeError(f"Piper returned no audio for language={language}")
        audio = np.concatenate(audio_chunks).astype(np.float32, copy=False)
        target_rate = self.requested_sample_rate or source_rate
        audio = self._resample(audio, source_rate, target_rate)
        audio = validate_tts_audio(audio, target_rate)

        latency_ms = (time.perf_counter() - start) * 1000
        duration_s = len(audio) / target_rate
        if duration_s > self.max_duration_seconds:
            raise RuntimeError(
                f"TTS audio duration {duration_s:.3f}s exceeds limit "
                f"{self.max_duration_seconds:.3f}s"
            )
        self.output_sample_rate = target_rate
        return TTSResult(
            audio=audio,
            sample_rate=target_rate,
            duration_s=duration_s,
            latency_ms=latency_ms,
            rtf=(latency_ms / 1000) / duration_s if duration_s else float("inf"),
            language=language,
        )

    @staticmethod
    def play_audio(audio: np.ndarray, sample_rate: int = 22050) -> None:
        audio = validate_tts_audio(audio, sample_rate)
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Install sounddevice to play audio interactively") from exc
        try:
            sd.play(audio, sample_rate)
            sd.wait()
        except BaseException:
            try:
                sd.stop()
            except Exception as stop_exc:
                logger.error(
                    "Failed to stop audio output after interruption "
                    "error_type=%s",
                    type(stop_exc).__name__,
                )
            raise

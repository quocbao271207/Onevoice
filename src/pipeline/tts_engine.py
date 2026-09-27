"""Offline Piper TTS with explicit, fail-closed model loading."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np

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


class TTSEngine:
    """Load one Vietnamese and one English Piper voice and cache both sessions."""

    def __init__(
        self,
        vi_model_path: str = "models/tts/vi_VN-vivos-x_low.onnx",
        en_model_path: str = "models/tts/en_US-lessac-medium.onnx",
        output_sample_rate: int | None = None,
        apply_medical_aliases: bool = False,
    ):
        self.model_paths = {"vi": Path(vi_model_path), "en": Path(en_model_path)}
        self.requested_sample_rate = output_sample_rate
        self.output_sample_rate = output_sample_rate or 22050
        self.apply_medical_aliases = apply_medical_aliases
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
        if not text.strip():
            raise ValueError("Cannot synthesize empty text")

        start = time.perf_counter()
        processed_text = self._apply_g2p(text, language) if self.apply_medical_aliases else text
        chunks = list(self._voices[language].synthesize(processed_text))
        if not chunks:
            raise RuntimeError(f"Piper returned no audio for language={language}")
        source_rate = int(chunks[0].sample_rate)
        audio = np.concatenate([chunk.audio_float_array for chunk in chunks]).astype(np.float32)
        target_rate = self.requested_sample_rate or source_rate
        audio = self._resample(audio, source_rate, target_rate)
        self.output_sample_rate = target_rate

        latency_ms = (time.perf_counter() - start) * 1000
        duration_s = len(audio) / target_rate
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
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Install sounddevice to play audio interactively") from exc
        sd.play(audio, sample_rate)
        sd.wait()

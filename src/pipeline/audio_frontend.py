"""
MediVoice Edge — Audio Frontend Module
Handles microphone input, noise suppression, and Voice Activity Detection (VAD).

Pipeline Stage 1: Raw Audio → Clean Audio Chunks with speech boundaries
"""

import numpy as np
import logging
import math
from numbers import Integral, Real
from typing import Optional, Generator, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


MAX_AUDIO_CHANNELS = 8
MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 384_000


def validate_audio_input(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Validate and normalize an inference audio boundary to mono float32."""
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, Integral)
        or not MIN_SAMPLE_RATE <= int(sample_rate) <= MAX_SAMPLE_RATE
    ):
        raise ValueError(
            "sample_rate must be an integer in "
            f"[{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}]"
        )
    try:
        value = np.asarray(audio, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("audio must be a numeric array") from exc
    if value.ndim == 2:
        channels = value.shape[1]
        if not 1 <= channels <= MAX_AUDIO_CHANNELS:
            raise ValueError(
                "two-dimensional audio must be channel-last with 1 to "
                f"{MAX_AUDIO_CHANNELS} channels"
            )
    elif value.ndim != 1:
        raise ValueError("audio must be one- or two-dimensional")
    if value.size == 0:
        raise ValueError("audio must contain at least one sample")
    if not np.isfinite(value).all():
        raise ValueError("audio must contain only finite samples")
    peak = float(np.max(np.abs(value)))
    if peak > 1.0 + 1e-6:
        raise ValueError("audio must be normalized to [-1, 1]")
    if value.ndim == 2:
        value = value.mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(value, dtype=np.float32)


@dataclass
class AudioConfig:
    """Configuration for audio frontend processing."""
    sample_rate: int = 16000
    bit_depth: int = 16
    channels: int = 1
    chunk_duration_ms: int = 500
    vad_threshold: float = 0.5
    silence_duration_ms: int = 800
    # Derived
    chunk_size: int = field(init=False)
    silence_chunks: int = field(init=False)

    def __post_init__(self):
        integer_fields = {
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "channels": self.channels,
            "chunk_duration_ms": self.chunk_duration_ms,
            "silence_duration_ms": self.silence_duration_ms,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if not MIN_SAMPLE_RATE <= self.sample_rate <= MAX_SAMPLE_RATE:
            raise ValueError(
                f"sample_rate must be in [{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}]"
            )
        if self.bit_depth not in {8, 16, 24, 32}:
            raise ValueError("bit_depth must be one of 8, 16, 24, or 32")
        if not 1 <= self.channels <= MAX_AUDIO_CHANNELS:
            raise ValueError(
                f"channels must be in [1, {MAX_AUDIO_CHANNELS}]"
            )
        if self.chunk_duration_ms <= 0:
            raise ValueError("chunk_duration_ms must be positive")
        if self.silence_duration_ms < 0:
            raise ValueError("silence_duration_ms cannot be negative")
        if (
            isinstance(self.vad_threshold, bool)
            or not isinstance(self.vad_threshold, Real)
            or not math.isfinite(self.vad_threshold)
            or not 0.0 <= self.vad_threshold <= 1.0
        ):
            raise ValueError("vad_threshold must be finite and in [0, 1]")
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        # A partial chunk still requires one full observation. Rounding down can
        # terminate speech earlier than the configured silence duration.
        self.silence_chunks = max(1, math.ceil(self.silence_duration_ms / self.chunk_duration_ms))


class VoiceActivityDetector:
    """
    Voice Activity Detection using Silero-VAD v4.
    Detects speech boundaries in streaming audio chunks.

    Silero-VAD is chosen because:
    - Ultra-lightweight (~1.5 MB)
    - < 10ms per frame processing
    - High accuracy for Vietnamese speech
    - Works well with code-switching (VI↔EN)
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = None
        self._is_loaded = False

    def load(self):
        """Load the Silero-VAD model."""
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True  # Use ONNX for faster inference
            )
            self.model = model
            self._get_speech_timestamps = utils[0]
            self._is_loaded = True
            logger.info("Silero-VAD v4 loaded successfully (ONNX mode)")
        except Exception as e:
            logger.warning(f"Failed to load Silero-VAD: {e}. Using energy-based fallback.")
            self._is_loaded = False

    def detect(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> float:
        """
        Returns speech probability [0.0 - 1.0] for an audio chunk.

        Args:
            audio_chunk: Audio samples as numpy array (float32, normalized to [-1, 1])
            sample_rate: Sample rate in Hz

        Returns:
            Speech probability between 0.0 and 1.0
        """
        audio_chunk = validate_audio_input(audio_chunk, sample_rate)
        if self._is_loaded and self.model is not None:
            import torch
            tensor = torch.from_numpy(audio_chunk).float()
            prob = self.model(tensor, sample_rate).item()
            return prob
        else:
            # Fallback: simple energy-based VAD
            energy = np.sqrt(np.mean(audio_chunk ** 2))
            return min(1.0, energy / 0.02)  # Normalize against typical speech energy

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
        """Check if chunk contains speech."""
        return self.detect(audio_chunk, sample_rate) >= self.threshold


class NoiseSuppressor:
    """
    Noise suppression for hospital environments.
    Uses RNNoise / WebRTC-based noise suppression.

    Designed to handle:
    - Ambulance sirens (75+ dB)
    - Ventilator noise
    - Patient monitor beeps
    - Background chatter in ER
    """

    def __init__(self):
        self._is_loaded = False

    def load(self):
        """Load noise suppression model."""
        try:
            # Try to use noisereduce as a Python-native alternative
            import noisereduce
            self._is_loaded = True
            logger.info("Noise suppression engine loaded (noisereduce)")
        except ImportError:
            logger.warning("noisereduce not available. Noise suppression disabled.")
            self._is_loaded = False

    def suppress(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Apply noise suppression to audio signal.

        Args:
            audio: Input audio signal (float32)
            sample_rate: Sample rate in Hz

        Returns:
            Denoised audio signal
        """
        audio = validate_audio_input(audio, sample_rate)
        if not self._is_loaded:
            return audio

        try:
            import noisereduce as nr
            # Stationary noise reduction — suitable for continuous background noise
            reduced = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                prop_decrease=0.8,  # Aggressive noise reduction for hospital env
                stationary=True
            )
            reduced = reduced.astype(np.float32)
        except Exception as e:
            logger.warning(f"Noise suppression failed: {e}. Returning original audio.")
            return audio
        return validate_audio_input(reduced, sample_rate)


class AudioFrontend:
    """
    Complete Audio Frontend pipeline:
    1. Receive raw PCM audio from microphone
    2. Apply noise suppression (RNNoise)
    3. Run VAD (Silero-VAD)
    4. Yield clean speech chunks with boundaries

    This module connects the 3-mic MEMS array to the ASR engine.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.vad = VoiceActivityDetector(threshold=self.config.vad_threshold)
        self.denoiser = NoiseSuppressor()
        self._buffer = deque(maxlen=100)  # Rolling buffer of chunks
        self._speech_active = False
        self._silence_count = 0
        self._speech_chunks = []

    def load(self):
        """Initialize all audio frontend components."""
        logger.info("Loading Audio Frontend components...")
        self.vad.load()
        self.denoiser.load()
        logger.info("Audio Frontend ready.")

    def process_chunk(self, raw_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single audio chunk through the frontend pipeline.

        Returns:
            Complete speech segment when end-of-speech is detected,
            None if speech is still ongoing or no speech detected.
        """
        # Step 1: Noise suppression
        clean_chunk = self.denoiser.suppress(raw_chunk, self.config.sample_rate)

        # Step 2: Voice Activity Detection
        is_speech = self.vad.is_speech(clean_chunk, self.config.sample_rate)

        if is_speech:
            self._speech_active = True
            self._silence_count = 0
            self._speech_chunks.append(clean_chunk)
            return None  # Still speaking, wait for more

        elif self._speech_active:
            self._silence_count += 1
            if self._silence_count >= self.config.silence_chunks:
                # End of speech detected — return complete segment
                speech_segment = np.concatenate(self._speech_chunks)
                self._speech_chunks = []
                self._speech_active = False
                self._silence_count = 0
                logger.debug(
                    f"Speech segment captured: {len(speech_segment) / self.config.sample_rate:.2f}s"
                )
                return speech_segment
            else:
                # Short pause within speech — keep buffering
                self._speech_chunks.append(clean_chunk)
                return None

        return None  # No speech

    def stream_from_microphone(self) -> Generator[np.ndarray, None, None]:
        """
        Stream audio from the default microphone and yield speech segments.

        Yields:
            Complete speech segments (numpy arrays) as they are detected.
        """
        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not installed. Cannot stream from microphone.")
            return

        logger.info(
            f"Starting microphone stream "
            f"(SR={self.config.sample_rate}, chunk={self.config.chunk_duration_ms}ms)"
        )

        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio stream status: {status}")
            chunk = indata[:, 0].astype(np.float32)  # Take first channel
            self._buffer.append(chunk)

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            blocksize=self.config.chunk_size,
            dtype='float32',
            callback=audio_callback
        ):
            logger.info("Microphone stream active. Listening...")
            while True:
                if self._buffer:
                    chunk = self._buffer.popleft()
                    segment = self.process_chunk(chunk)
                    if segment is not None:
                        yield segment
                else:
                    import time
                    time.sleep(0.01)  # Avoid busy-waiting

"""
MediVoice Edge — Audio Frontend Module
Handles microphone input, noise suppression, and Voice Activity Detection (VAD).

Pipeline Stage 1: Raw Audio → Clean Audio Chunks with speech boundaries
"""

import numpy as np
import logging
import math
from numbers import Integral, Real
from typing import Optional, Generator
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from threading import Lock

logger = logging.getLogger(__name__)


MAX_AUDIO_CHANNELS = 8
MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 384_000
WEBRTC_SAMPLE_RATES = {8_000, 16_000, 32_000, 48_000}
WEBRTC_FRAME_DURATION_MS = 10
MICROPHONE_MAX_BACKLOG_MS = 2_000
MICROPHONE_QUEUE_POLL_SECONDS = 0.1


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
    vad_backend: str = "webrtc"
    vad_aggressiveness: int = 2
    noise_suppression_enabled: bool = False
    silence_duration_ms: int = 800
    max_segment_duration_seconds: float = 30.0
    # Derived
    chunk_size: int = field(init=False)
    silence_chunks: int = field(init=False)
    max_segment_samples: int = field(init=False)

    def __post_init__(self):
        integer_fields = {
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "channels": self.channels,
            "chunk_duration_ms": self.chunk_duration_ms,
            "vad_aggressiveness": self.vad_aggressiveness,
            "silence_duration_ms": self.silence_duration_ms,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if not MIN_SAMPLE_RATE <= self.sample_rate <= MAX_SAMPLE_RATE:
            raise ValueError(
                f"sample_rate must be in [{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}]"
            )
        if self.vad_backend != "webrtc":
            raise ValueError("vad_backend must be 'webrtc'")
        if self.sample_rate not in WEBRTC_SAMPLE_RATES:
            raise ValueError(
                "WebRTC VAD sample_rate must be one of "
                f"{sorted(WEBRTC_SAMPLE_RATES)}"
            )
        if not 0 <= self.vad_aggressiveness <= 3:
            raise ValueError("vad_aggressiveness must be in [0, 3]")
        if type(self.noise_suppression_enabled) is not bool:
            raise ValueError("noise_suppression_enabled must be a boolean")
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
        if (
            isinstance(self.max_segment_duration_seconds, bool)
            or not isinstance(self.max_segment_duration_seconds, Real)
            or not math.isfinite(float(self.max_segment_duration_seconds))
            or float(self.max_segment_duration_seconds) <= 0.0
        ):
            raise ValueError(
                "max_segment_duration_seconds must be finite and positive"
            )
        self.max_segment_duration_seconds = float(
            self.max_segment_duration_seconds
        )
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        self.max_segment_samples = int(
            self.sample_rate * self.max_segment_duration_seconds
        )
        if self.max_segment_samples < self.chunk_size:
            raise ValueError(
                "max_segment_duration_seconds must cover at least one "
                "configured audio chunk"
            )
        # A partial chunk still requires one full observation. Rounding down can
        # terminate speech earlier than the configured silence duration.
        self.silence_chunks = max(1, math.ceil(self.silence_duration_ms / self.chunk_duration_ms))


class VoiceActivityDetector:
    """
    Offline voice activity detection using WebRTC VAD.
    Detects speech boundaries in streaming audio chunks.

    The backend accepts local PCM frames only and performs no runtime network
    access. Chunk probability is the fraction of 10 ms frames marked voiced.
    """

    def __init__(self, threshold: float = 0.5, aggressiveness: int = 2):
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not math.isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError("threshold must be finite and in [0, 1]")
        if (
            isinstance(aggressiveness, bool)
            or not isinstance(aggressiveness, Integral)
            or not 0 <= int(aggressiveness) <= 3
        ):
            raise ValueError("aggressiveness must be an integer in [0, 3]")
        self.threshold = threshold
        self.aggressiveness = int(aggressiveness)
        self.model = None
        self._is_loaded = False

    def load(self):
        """Load the local WebRTC VAD extension without network fallback."""
        try:
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError(
                "Install webrtcvad-wheels from requirements-local.txt; "
                "energy fallback is disabled"
            ) from exc
        try:
            self.model = webrtcvad.Vad(self.aggressiveness)
        except Exception as exc:
            raise RuntimeError("Failed to initialize WebRTC VAD") from exc
        self._is_loaded = True
        logger.info(
            "WebRTC VAD loaded backend=local aggressiveness=%d frame_ms=%d",
            self.aggressiveness,
            WEBRTC_FRAME_DURATION_MS,
        )

    @property
    def is_ready(self) -> bool:
        return self._is_loaded and self.model is not None

    def detect(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> float:
        """
        Returns speech probability [0.0 - 1.0] for an audio chunk.

        Args:
            audio_chunk: Audio samples as numpy array (float32, normalized to [-1, 1])
            sample_rate: Sample rate in Hz

        Returns:
            Speech probability between 0.0 and 1.0
        """
        if not self.is_ready:
            raise RuntimeError("WebRTC VAD is not loaded")
        audio_chunk = validate_audio_input(audio_chunk, sample_rate)
        if sample_rate not in WEBRTC_SAMPLE_RATES:
            raise ValueError(
                "WebRTC VAD sample_rate must be one of "
                f"{sorted(WEBRTC_SAMPLE_RATES)}"
            )
        frame_samples = int(sample_rate * WEBRTC_FRAME_DURATION_MS / 1000)
        frame_count = math.ceil(len(audio_chunk) / frame_samples)
        padded_samples = frame_count * frame_samples
        if padded_samples != len(audio_chunk):
            audio_chunk = np.pad(
                audio_chunk,
                (0, padded_samples - len(audio_chunk)),
            )
        pcm16 = np.clip(
            np.rint(audio_chunk * 32767.0),
            -32768,
            32767,
        ).astype("<i2")
        voiced_frames = 0
        try:
            for start in range(0, padded_samples, frame_samples):
                frame = pcm16[start : start + frame_samples].tobytes()
                voiced_frames += int(self.model.is_speech(frame, sample_rate))
        except Exception as exc:
            raise RuntimeError("WebRTC VAD inference failed") from exc
        return voiced_frames / frame_count

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> bool:
        """Check if chunk contains speech."""
        return self.detect(audio_chunk, sample_rate) >= self.threshold


class NoiseSuppressor:
    """
    Optional stationary spectral-gating backend.

    The backend remains disabled until it has clinical speech-retention
    evidence; enabled mode never degrades silently to pass-through.
    """

    def __init__(self, enabled: bool = False):
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        self.enabled = enabled
        self._backend = None
        self._is_loaded = False

    def load(self):
        """Load the configured backend or mark explicit pass-through ready."""
        if not self.enabled:
            self._backend = None
            self._is_loaded = False
            logger.info("Noise suppression disabled by configuration")
            return
        try:
            import noisereduce
        except ImportError as exc:
            raise RuntimeError(
                "Noise suppression is enabled but noisereduce is not installed"
            ) from exc
        self._backend = noisereduce
        self._is_loaded = True
        logger.info("Noise suppression engine loaded backend=noisereduce")

    @property
    def is_ready(self) -> bool:
        return not self.enabled or (
            self._is_loaded and self._backend is not None
        )

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
        if not self.enabled:
            return audio
        if not self.is_ready:
            raise RuntimeError("Noise suppression backend is not loaded")

        try:
            # Deterministic settings; clinical validation is required before enablement.
            reduced = self._backend.reduce_noise(
                y=audio,
                sr=sample_rate,
                prop_decrease=0.8,
                stationary=True,
            )
            reduced = np.asarray(reduced, dtype=np.float32)
        except Exception as exc:
            raise RuntimeError("Noise suppression failed") from exc
        return validate_audio_input(reduced, sample_rate)


class AudioFrontend:
    """
    Complete Audio Frontend pipeline:
    1. Receive raw PCM audio from microphone
    2. Apply explicitly configured noise suppression
    3. Run local WebRTC VAD
    4. Yield clean speech chunks with boundaries

    This module connects the 3-mic MEMS array to the ASR engine.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.vad = VoiceActivityDetector(
            threshold=self.config.vad_threshold,
            aggressiveness=self.config.vad_aggressiveness,
        )
        self.denoiser = NoiseSuppressor(
            enabled=self.config.noise_suppression_enabled,
        )
        self._microphone_buffer_capacity_chunks = max(
            2,
            math.ceil(
                MICROPHONE_MAX_BACKLOG_MS / self.config.chunk_duration_ms
            ),
        )
        self._buffer: Queue[np.ndarray] = Queue(
            maxsize=self._microphone_buffer_capacity_chunks
        )
        self._stream_error: Optional[str] = None
        self._stream_error_lock = Lock()
        self._speech_active = False
        self._silence_count = 0
        self._speech_chunks = []
        self._speech_sample_count = 0
        self._is_loaded = False

    def load(self):
        """Initialize all audio frontend components."""
        if self.is_ready:
            logger.info("Audio Frontend already ready; skipping reload")
            return
        self._is_loaded = False
        logger.info("Loading Audio Frontend components...")
        self.vad.load()
        self.denoiser.load()
        self._is_loaded = True
        logger.info(
            "Audio Frontend ready vad=webrtc noise_suppression=%s",
            "enabled" if self.denoiser.enabled else "disabled",
        )

    @property
    def is_ready(self) -> bool:
        return (
            self._is_loaded
            and self.vad.is_ready
            and self.denoiser.is_ready
        )

    def get_status(self) -> dict:
        return {
            "ready": self.is_ready,
            "vad_backend": self.config.vad_backend,
            "vad_loaded": self.vad.is_ready,
            "noise_suppression_enabled": self.denoiser.enabled,
            "noise_suppression_loaded": self.denoiser._is_loaded,
            "microphone_buffer_capacity_chunks": (
                self._microphone_buffer_capacity_chunks
            ),
            "microphone_stream_error": self._stream_error,
        }

    def _record_stream_error(self, message: str) -> None:
        with self._stream_error_lock:
            if self._stream_error is None:
                self._stream_error = message

    def _raise_stream_error(self) -> None:
        with self._stream_error_lock:
            message = self._stream_error
        if message is not None:
            raise RuntimeError(message)

    def _reset_microphone_stream_state(self) -> None:
        self._buffer = Queue(maxsize=self._microphone_buffer_capacity_chunks)
        with self._stream_error_lock:
            self._stream_error = None
        self._speech_active = False
        self._silence_count = 0
        self._speech_chunks = []
        self._speech_sample_count = 0

    def _append_speech_chunk(self, chunk: np.ndarray) -> None:
        next_sample_count = self._speech_sample_count + len(chunk)
        if next_sample_count > self.config.max_segment_samples:
            self._speech_active = False
            self._silence_count = 0
            self._speech_chunks = []
            self._speech_sample_count = 0
            raise RuntimeError(
                "Speech segment exceeded "
                f"{self.config.max_segment_duration_seconds:.3f}s; "
                "refusing unbounded accumulation"
            )
        self._speech_chunks.append(chunk)
        self._speech_sample_count = next_sample_count

    def _enqueue_microphone_chunk(
        self,
        indata: np.ndarray,
        frames: int,
        status,
    ) -> None:
        """Validate a callback chunk and enqueue it without silent eviction."""
        if status:
            self._record_stream_error(
                f"Microphone input reported status: {status}"
            )
            return
        if (
            isinstance(frames, bool)
            or not isinstance(frames, Integral)
            or int(frames) != self.config.chunk_size
        ):
            self._record_stream_error(
                "Microphone frame count does not match configured chunk size"
            )
            return
        try:
            chunk = np.asarray(indata, dtype=np.float32)
        except (TypeError, ValueError):
            self._record_stream_error("Microphone chunk must be numeric")
            return
        expected_shape = (self.config.chunk_size, self.config.channels)
        if chunk.shape != expected_shape:
            self._record_stream_error(
                "Microphone chunk shape does not match configured frames/channels"
            )
            return
        if not np.isfinite(chunk).all():
            self._record_stream_error(
                "Microphone chunk must contain only finite samples"
            )
            return
        if float(np.max(np.abs(chunk))) > 1.0 + 1e-6:
            self._record_stream_error(
                "Microphone chunk must be normalized to [-1, 1]"
            )
            return
        try:
            self._buffer.put_nowait(np.array(chunk, dtype=np.float32, copy=True))
        except Full:
            self._record_stream_error(
                "Microphone input buffer overflow; refusing silent audio loss"
            )

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
            self._append_speech_chunk(clean_chunk)
            return None  # Still speaking, wait for more

        elif self._speech_active:
            self._silence_count += 1
            if self._silence_count >= self.config.silence_chunks:
                # End of speech detected — return complete segment
                speech_segment = np.concatenate(self._speech_chunks)
                self._speech_chunks = []
                self._speech_sample_count = 0
                self._speech_active = False
                self._silence_count = 0
                logger.debug(
                    f"Speech segment captured: {len(speech_segment) / self.config.sample_rate:.2f}s"
                )
                return speech_segment
            else:
                # Short pause within speech — keep buffering
                self._append_speech_chunk(clean_chunk)
                return None

        return None  # No speech

    def stream_from_microphone(self) -> Generator[np.ndarray, None, None]:
        """
        Stream audio from the default microphone and yield speech segments.

        Yields:
            Complete speech segments (numpy arrays) as they are detected.
        """
        if not self.get_status()["ready"]:
            raise RuntimeError("Audio frontend is not ready; call load() first")
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is required for microphone streaming"
            ) from exc

        logger.info(
            f"Starting microphone stream "
            f"(SR={self.config.sample_rate}, chunk={self.config.chunk_duration_ms}ms)"
        )

        self._reset_microphone_stream_state()

        def audio_callback(indata, frames, _time_info, status):
            self._enqueue_microphone_chunk(indata, frames, status)

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            blocksize=self.config.chunk_size,
            dtype='float32',
            callback=audio_callback
        ):
            logger.info("Microphone stream active. Listening...")
            while True:
                self._raise_stream_error()
                try:
                    chunk = self._buffer.get(
                        timeout=MICROPHONE_QUEUE_POLL_SECONDS
                    )
                except Empty:
                    self._raise_stream_error()
                    continue
                self._raise_stream_error()
                segment = self.process_chunk(chunk)
                if segment is not None:
                    yield segment

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.pipeline.audio_frontend import (
    AudioConfig,
    AudioFrontend,
    NoiseSuppressor,
    VoiceActivityDetector,
)
from src.pipeline.orchestrator import MediVoicePipeline


def block_import(monkeypatch, blocked_name):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == blocked_name:
            raise ImportError(f"blocked {blocked_name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, blocked_name, raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_webrtc_vad_requires_local_dependency_without_energy_fallback(monkeypatch):
    block_import(monkeypatch, "webrtcvad")
    vad = VoiceActivityDetector(threshold=0.5, aggressiveness=2)

    with pytest.raises(RuntimeError, match="webrtcvad-wheels"):
        vad.load()
    with pytest.raises(RuntimeError, match="not loaded"):
        vad.detect(np.zeros(160, dtype=np.float32), 16_000)


def test_webrtc_vad_frames_pcm16_and_reports_speech_ratio(monkeypatch):
    class FakeVad:
        def __init__(self, mode):
            self.mode = mode
            self.calls = []

        def is_speech(self, frame, sample_rate):
            self.calls.append((frame, sample_rate))
            return len(self.calls) == 1

    fake_vad = FakeVad(mode=3)
    monkeypatch.setitem(
        sys.modules,
        "webrtcvad",
        SimpleNamespace(Vad=lambda mode: fake_vad),
    )
    vad = VoiceActivityDetector(threshold=0.5, aggressiveness=3)
    vad.load()

    probability = vad.detect(
        np.concatenate(
            (
                np.full(160, 0.25, dtype=np.float32),
                np.zeros(160, dtype=np.float32),
            )
        ),
        16_000,
    )

    assert probability == 0.5
    assert vad.is_speech(np.zeros(160, dtype=np.float32), 16_000) is False
    assert fake_vad.mode == 3
    assert len(fake_vad.calls) == 3
    assert all(len(frame) == 320 for frame, _ in fake_vad.calls)
    assert all(sample_rate == 16_000 for _, sample_rate in fake_vad.calls)


def test_audio_frontend_load_is_offline_and_reports_explicit_status(monkeypatch):
    block_import(monkeypatch, "torch")
    frontend = AudioFrontend()

    frontend.load()
    status = frontend.get_status()

    assert status == {
        "ready": True,
        "vad_backend": "webrtc",
        "vad_loaded": True,
        "noise_suppression_enabled": False,
        "noise_suppression_loaded": False,
        "microphone_buffer_capacity_chunks": 4,
        "microphone_stream_error": None,
    }
    assert frontend.vad.detect(np.zeros(160, dtype=np.float32), 16_000) == 0.0


def test_audio_config_restricts_webrtc_contract():
    with pytest.raises(ValueError, match="vad_backend"):
        AudioConfig(vad_backend="silero_hub")
    with pytest.raises(ValueError, match="vad_aggressiveness"):
        AudioConfig(vad_aggressiveness=4)
    with pytest.raises(ValueError, match="noise_suppression_enabled"):
        AudioConfig(noise_suppression_enabled="false")
    with pytest.raises(ValueError, match="WebRTC VAD sample_rate"):
        AudioConfig(sample_rate=44_100)
    with pytest.raises(ValueError, match="max_segment_duration_seconds"):
        AudioConfig(max_segment_duration_seconds=float("inf"))
    with pytest.raises(ValueError, match="at least one configured audio chunk"):
        AudioConfig(
            chunk_duration_ms=500,
            max_segment_duration_seconds=0.25,
        )


def test_disabled_noise_suppression_is_explicit_pass_through(monkeypatch):
    block_import(monkeypatch, "noisereduce")
    denoiser = NoiseSuppressor(enabled=False)
    denoiser.load()
    audio = np.array([0.1, -0.2], dtype=np.float32)

    output = denoiser.suppress(audio, 16_000)

    assert np.array_equal(output, audio)
    assert denoiser.enabled is False
    assert denoiser.is_ready


def test_enabled_noise_suppression_fails_closed_when_unavailable(monkeypatch):
    block_import(monkeypatch, "noisereduce")
    denoiser = NoiseSuppressor(enabled=True)

    with pytest.raises(RuntimeError, match="noisereduce"):
        denoiser.load()
    with pytest.raises(RuntimeError, match="not loaded"):
        denoiser.suppress(np.zeros(160, dtype=np.float32), 16_000)


def test_noise_suppression_runtime_failure_never_returns_raw_audio(monkeypatch):
    def fail_reduce_noise(**_kwargs):
        raise ValueError("backend failed")

    monkeypatch.setitem(
        sys.modules,
        "noisereduce",
        SimpleNamespace(reduce_noise=fail_reduce_noise),
    )
    denoiser = NoiseSuppressor(enabled=True)
    denoiser.load()

    with pytest.raises(RuntimeError, match="Noise suppression failed"):
        denoiser.suppress(np.zeros(160, dtype=np.float32), 16_000)


def test_pipeline_applies_explicit_audio_frontend_backends():
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")

    assert pipeline.audio_frontend.config.vad_backend == "webrtc"
    assert pipeline.audio_frontend.config.vad_aggressiveness == 2
    assert pipeline.audio_frontend.config.noise_suppression_enabled is False
    assert pipeline.audio_frontend.denoiser.enabled is False
    assert (
        pipeline.audio_frontend.config.max_segment_duration_seconds
        == pipeline.asr_engine.max_input_duration_seconds
        == 30.0
    )


def test_frontend_bounds_continuous_speech_before_unbounded_accumulation(
    monkeypatch,
):
    frontend = AudioFrontend(
        AudioConfig(
            chunk_duration_ms=1_000,
            max_segment_duration_seconds=2.0,
        )
    )
    chunk = np.zeros(frontend.config.chunk_size, dtype=np.float32)
    monkeypatch.setattr(frontend.denoiser, "suppress", lambda audio, _rate: audio)
    monkeypatch.setattr(frontend.vad, "is_speech", lambda *_args: True)

    assert frontend.process_chunk(chunk) is None
    assert frontend.process_chunk(chunk) is None
    with pytest.raises(RuntimeError, match="segment exceeded 2.000s"):
        frontend.process_chunk(chunk)

    assert frontend._speech_chunks == []
    assert frontend._speech_sample_count == 0


def test_frontend_releases_segment_at_exact_duration_limit(monkeypatch):
    frontend = AudioFrontend(
        AudioConfig(
            chunk_duration_ms=1_000,
            silence_duration_ms=800,
            max_segment_duration_seconds=2.0,
        )
    )
    chunk = np.zeros(frontend.config.chunk_size, dtype=np.float32)
    speech_decisions = iter((True, True, False))
    monkeypatch.setattr(frontend.denoiser, "suppress", lambda audio, _rate: audio)
    monkeypatch.setattr(
        frontend.vad,
        "is_speech",
        lambda *_args: next(speech_decisions),
    )

    assert frontend.process_chunk(chunk) is None
    assert frontend.process_chunk(chunk) is None
    segment = frontend.process_chunk(chunk)

    assert segment is not None
    assert len(segment) == 2 * frontend.config.sample_rate


def test_microphone_queue_overflow_and_status_fail_closed_without_dropping():
    frontend = AudioFrontend(AudioConfig(chunk_duration_ms=1_000))
    chunk = np.zeros(
        (frontend.config.chunk_size, frontend.config.channels),
        dtype=np.float32,
    )

    frontend._enqueue_microphone_chunk(
        chunk,
        frontend.config.chunk_size,
        status=None,
    )
    frontend._enqueue_microphone_chunk(
        chunk,
        frontend.config.chunk_size,
        status=None,
    )
    assert frontend._buffer.qsize() == 2

    frontend._enqueue_microphone_chunk(
        chunk,
        frontend.config.chunk_size,
        status=None,
    )
    assert frontend._buffer.qsize() == 2
    with pytest.raises(RuntimeError, match="buffer overflow"):
        frontend._raise_stream_error()

    frontend._reset_microphone_stream_state()
    frontend._enqueue_microphone_chunk(
        chunk,
        frontend.config.chunk_size,
        status="input overflow",
    )
    assert frontend._buffer.empty()
    with pytest.raises(RuntimeError, match="reported status"):
        frontend._raise_stream_error()


@pytest.mark.parametrize(
    ("frames", "shape", "fill_value", "message"),
    [
        (8_000 - 1, (8_000, 1), 0.0, "frame count"),
        (8_000, (8_000, 2), 0.0, "shape"),
        (8_000, (8_000, 1), float("nan"), "finite"),
        (8_000, (8_000, 1), 1.1, "normalized"),
    ],
)
def test_microphone_callback_rejects_malformed_chunks(
    frames,
    shape,
    fill_value,
    message,
):
    frontend = AudioFrontend()
    chunk = np.full(shape, fill_value, dtype=np.float32)

    frontend._enqueue_microphone_chunk(chunk, frames, status=None)

    assert frontend._buffer.empty()
    with pytest.raises(RuntimeError, match=message):
        frontend._raise_stream_error()


def test_microphone_stream_preserves_channels_and_uses_bounded_queue(monkeypatch):
    frontend = AudioFrontend(AudioConfig(channels=2))
    frontend.load()
    frames = frontend.config.chunk_size
    stereo = np.column_stack(
        (
            np.zeros(frames, dtype=np.float32),
            np.full(frames, 0.25, dtype=np.float32),
        )
    )
    observed = {}

    class FakeInputStream:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def __enter__(self):
            observed["callback"](stereo, frames, None, None)
            return self

        def __exit__(self, *_args):
            observed["closed"] = True

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(InputStream=FakeInputStream),
    )
    monkeypatch.setattr(frontend, "process_chunk", lambda chunk: chunk)

    stream = frontend.stream_from_microphone()
    segment = next(stream)
    stream.close()

    assert np.array_equal(segment, stereo)
    assert observed["channels"] == 2
    assert observed["blocksize"] == frames
    assert observed["closed"] is True


def test_microphone_stream_requires_ready_frontend_and_sounddevice(monkeypatch):
    frontend = AudioFrontend()
    with pytest.raises(RuntimeError, match="not ready"):
        next(frontend.stream_from_microphone())

    frontend.load()
    block_import(monkeypatch, "sounddevice")
    with pytest.raises(RuntimeError, match="sounddevice"):
        next(frontend.stream_from_microphone())

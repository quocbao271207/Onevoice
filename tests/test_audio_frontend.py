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

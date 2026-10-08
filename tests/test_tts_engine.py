from types import SimpleNamespace

import numpy as np
import pytest

from src.pipeline.orchestrator import MediVoicePipeline
from src.pipeline.tts_engine import TTSEngine


class FakeVoice:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls: list[str] = []

    def synthesize(self, text: str):
        self.calls.append(text)
        return iter(self.chunks)


def loaded_engine(chunks, **kwargs) -> tuple[TTSEngine, FakeVoice]:
    engine = TTSEngine(**kwargs)
    voice = FakeVoice(chunks)
    engine._voices = {"vi": voice, "en": voice}
    engine._is_loaded = True
    return engine, voice


def audio_chunk(values, sample_rate=22050):
    return SimpleNamespace(
        audio_float_array=np.asarray(values, dtype=np.float32),
        sample_rate=sample_rate,
    )


def test_tts_rejects_invalid_text_before_voice_dispatch():
    engine, voice = loaded_engine(
        [audio_chunk([0.0, 0.1])],
        max_text_characters=10,
    )

    with pytest.raises(TypeError, match="must be a string"):
        engine.synthesize(None, "vi")
    with pytest.raises(ValueError, match="must not be blank"):
        engine.synthesize("   ", "vi")
    with pytest.raises(ValueError, match="11 characters.*limit is 10"):
        engine.synthesize("a" * 11, "vi")
    with pytest.raises(ValueError, match="control characters"):
        engine.synthesize("Give aspirin.\x00 Do not give warfarin.", "en")

    assert voice.calls == []


def test_tts_validates_and_returns_finite_normalized_audio():
    engine, voice = loaded_engine(
        [audio_chunk([0.0, 0.25]), audio_chunk([-0.25, 0.0])]
    )

    result = engine.synthesize("Bệnh nhân ổn định.", "vi")

    assert voice.calls == ["Bệnh nhân ổn định."]
    assert result.audio.dtype == np.float32
    assert result.audio.flags.c_contiguous
    assert np.array_equal(
        result.audio,
        np.array([0.0, 0.25, -0.25, 0.0], dtype=np.float32),
    )
    assert result.sample_rate == 22050
    assert result.duration_s == pytest.approx(4 / 22050)
    assert np.isfinite(result.rtf)


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        ([audio_chunk([])], "at least one sample"),
        ([audio_chunk([0.0, np.nan])], "finite"),
        ([audio_chunk([0.0, 1.1])], "normalized"),
        ([audio_chunk([[0.0], [0.1]])], "one-dimensional"),
        (
            [audio_chunk([0.0], 22050), audio_chunk([0.0], 16000)],
            "inconsistent sample rates",
        ),
    ],
)
def test_tts_rejects_malformed_piper_audio(chunks, message):
    engine, _voice = loaded_engine(chunks)

    with pytest.raises((ValueError, RuntimeError), match=message):
        engine.synthesize("Stable.", "en")


def test_tts_rejects_unbounded_duration_and_invalid_configuration():
    engine, _voice = loaded_engine(
        [audio_chunk(np.zeros(100, dtype=np.float32), sample_rate=10000)],
        max_duration_seconds=0.005,
    )
    with pytest.raises(RuntimeError, match="duration.*limit"):
        engine.synthesize("Stable.", "en")

    with pytest.raises(ValueError, match="output_sample_rate"):
        TTSEngine(output_sample_rate=0)
    with pytest.raises(ValueError, match="max_text_characters"):
        TTSEngine(max_text_characters=True)
    with pytest.raises(ValueError, match="max_duration_seconds"):
        TTSEngine(max_duration_seconds=float("inf"))


def test_tts_stops_consuming_chunks_as_soon_as_duration_limit_is_crossed():
    class UnboundedVoice:
        def synthesize(self, _text):
            yield audio_chunk(np.zeros(6, dtype=np.float32), sample_rate=8000)
            yield audio_chunk(np.zeros(6, dtype=np.float32), sample_rate=8000)
            raise AssertionError("TTS must stop before requesting another chunk")

    engine = TTSEngine(max_duration_seconds=0.001)
    engine._voices = {"en": UnboundedVoice()}
    engine._is_loaded = True

    with pytest.raises(RuntimeError, match="duration exceeds limit"):
        engine.synthesize("Stable.", "en")


def test_play_audio_validates_before_device_dispatch():
    with pytest.raises(ValueError, match="finite"):
        TTSEngine.play_audio(np.array([np.nan], dtype=np.float32), 22050)


def test_pipeline_revalidates_cached_audio_before_return(monkeypatch):
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")
    pipeline._is_loaded = True
    monkeypatch.setattr(
        pipeline.audio_frontend.denoiser,
        "suppress",
        lambda audio, _sample_rate: audio,
    )
    monkeypatch.setattr(
        pipeline.asr_engine,
        "transcribe",
        lambda *_args, **_kwargs: SimpleNamespace(
            text="Check the pulse",
            language="en",
            confidence=1.0,
            latency_ms=1.0,
        ),
    )
    monkeypatch.setattr(
        pipeline.flash_cache,
        "lookup",
        lambda *_args: SimpleNamespace(
            source_lang="en",
            translated_text="Kiểm tra mạch",
            target_lang="vi",
            audio=np.array([np.nan], dtype=np.float32),
            audio_sample_rate=22050,
            requires_confirmation=False,
        ),
    )

    with pytest.raises(ValueError, match="finite"):
        pipeline.translate_speech(
            np.zeros(1600, dtype=np.float32),
            source_lang="en",
            target_lang="vi",
        )


def test_pipeline_applies_tts_boundary_configuration():
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")

    assert pipeline.tts_engine.requested_sample_rate == 22050
    assert pipeline.tts_engine.max_text_characters == 4096
    assert pipeline.tts_engine.max_duration_seconds == 120.0

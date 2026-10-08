import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.pipeline.mt_engine import MTResult
from src.pipeline.orchestrator import MediVoicePipeline, tts_degradation_code
from src.pipeline.tts_engine import TTSEngine, TTSResult


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
    voice = UnboundedVoice()
    engine._voices = {"vi": voice, "en": voice}
    engine._is_loaded = True

    with pytest.raises(RuntimeError, match="duration exceeds limit"):
        engine.synthesize("Stable.", "en")


def test_play_audio_validates_before_device_dispatch():
    with pytest.raises(ValueError, match="finite"):
        TTSEngine.play_audio(np.array([np.nan], dtype=np.float32), 22050)


def test_play_audio_stops_device_when_wait_is_interrupted(monkeypatch):
    events = []

    def interrupted_wait():
        events.append("wait")
        raise KeyboardInterrupt

    fake_sounddevice = SimpleNamespace(
        play=lambda audio, rate: events.append(("play", audio.copy(), rate)),
        wait=interrupted_wait,
        stop=lambda: events.append("stop"),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    audio = np.array([0.0, 0.25], dtype=np.float32)

    with pytest.raises(KeyboardInterrupt):
        TTSEngine.play_audio(audio, 22_050)

    assert events[0][0] == "play"
    assert np.array_equal(events[0][1], audio)
    assert events[0][2] == 22_050
    assert events[1:] == ["wait", "stop"]


def test_pipeline_audio_sink_rechecks_safety_before_device_dispatch(monkeypatch):
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")
    dispatched = []
    monkeypatch.setattr(
        pipeline.tts_engine,
        "play_audio",
        lambda audio, rate: dispatched.append((audio, rate)),
    )
    audio = np.zeros(8, dtype=np.float32)

    def playback_result(**overrides):
        values = {
            "asr_text": "Check the pulse",
            "translated_text": "Kiểm tra mạch",
            "asr_language": "en",
            "target_language": "vi",
            "output_audio": audio,
            "output_sample_rate": 22_050,
            "safety_passed": True,
            "safety_issues": [],
            "requires_confirmation": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    with pytest.raises(RuntimeError, match="clinical safety gate"):
        pipeline.play(playback_result(safety_passed=False))
    with pytest.raises(RuntimeError, match="confirmation state"):
        pipeline.play(playback_result(requires_confirmation=True))
    with pytest.raises(RuntimeError, match="clinical safety gate"):
        pipeline.play(playback_result(safety_passed="true"))
    with pytest.raises(RuntimeError, match="confirmation state"):
        pipeline.play(playback_result(requires_confirmation="false"))
    with pytest.raises(RuntimeError, match="no longer matches"):
        pipeline.play(
            playback_result(
                asr_text="Give aspirin 5 mg",
                translated_text="Cho aspirin 50 mg",
            )
        )

    assert dispatched == []
    pipeline.play(playback_result())
    assert len(dispatched) == 1
    assert np.array_equal(dispatched[0][0], audio)
    assert dispatched[0][1] == 22_050


def test_confirmed_clinical_action_is_synthesized_only_after_explicit_consent(
    monkeypatch,
):
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")
    audio = np.array([0.0, 0.25], dtype=np.float32)
    synthesized = TTSResult(
        audio=audio,
        sample_rate=22_050,
        duration_s=len(audio) / 22_050,
        latency_ms=2.0,
        rtf=0.1,
        language="vi",
    )
    synthesize_calls = []
    dispatched = []
    monkeypatch.setattr(
        pipeline.tts_engine,
        "synthesize",
        lambda text, language: (
            synthesize_calls.append((text, language)) or synthesized
        ),
    )
    monkeypatch.setattr(
        pipeline.tts_engine,
        "play_audio",
        lambda value, rate: dispatched.append((value.copy(), rate)),
    )
    result = SimpleNamespace(
        asr_text="Check the pulse",
        translated_text="Kiểm tra mạch",
        asr_language="en",
        target_language="vi",
        output_audio=None,
        output_sample_rate=22_050,
        safety_passed=True,
        safety_issues=[],
        requires_confirmation=True,
    )

    with pytest.raises(RuntimeError, match="explicit confirmation is required"):
        pipeline.confirm_and_play(result, confirmed=False)
    assert synthesize_calls == []
    assert dispatched == []

    returned = pipeline.confirm_and_play(result, confirmed=True)

    assert returned is synthesized
    assert synthesize_calls == [("Kiểm tra mạch", "vi")]
    assert len(dispatched) == 1
    assert np.array_equal(dispatched[0][0], audio)
    assert dispatched[0][1] == 22_050


def test_pipeline_revalidates_cached_audio_before_return(
    monkeypatch,
    mark_pipeline_ready,
):
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")
    mark_pipeline_ready(pipeline)
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

    stats = pipeline.get_performance_stats()
    assert stats["total_translations"] == 0
    assert stats["cache_hits"] == 0
    assert stats["latency_window_samples"] == 0


def test_pipeline_records_pre_synthesized_cache_latency(
    monkeypatch,
    mark_pipeline_ready,
):
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")
    mark_pipeline_ready(pipeline)
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
            audio=np.zeros(8, dtype=np.float32),
            audio_sample_rate=22050,
            requires_confirmation=False,
        ),
    )

    result = pipeline.translate_speech(
        np.zeros(1600, dtype=np.float32),
        source_lang="en",
        target_lang="vi",
    )
    stats = pipeline.get_performance_stats()

    assert result.from_cache
    assert stats["total_translations"] == 1
    assert stats["cache_hits"] == 1
    assert stats["latency_window_samples"] == 1
    assert len(pipeline._latency_history) == 1


def test_pipeline_applies_tts_boundary_configuration():
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")

    assert pipeline.tts_engine.requested_sample_rate == 22050
    assert pipeline.tts_engine.max_text_characters == 4096
    assert pipeline.tts_engine.max_duration_seconds == 120.0


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (FileNotFoundError("secret path"), "tts_artifact_unavailable"),
        (OSError("private device detail"), "tts_device_io_error"),
        (ValueError("private tensor detail"), "tts_output_validation_error"),
        (RuntimeError("private backend detail"), "tts_runtime_error"),
    ],
)
def test_tts_degradation_codes_are_stable_and_redacted(error, expected_code):
    assert tts_degradation_code(error) == expected_code
    assert str(error) not in tts_degradation_code(error)


def test_pipeline_preserves_safe_text_when_configured_tts_fallback_fires(
    monkeypatch,
    mark_pipeline_ready,
    caplog,
):
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")
    mark_pipeline_ready(pipeline)
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
    monkeypatch.setattr(pipeline.flash_cache, "lookup", lambda *_args: None)
    monkeypatch.setattr(
        pipeline.mt_engine,
        "translate",
        lambda text, source_lang, target_lang: MTResult(
            source_text=text,
            translated_text="Kiểm tra mạch",
            source_lang=source_lang,
            target_lang=target_lang,
            latency_ms=1.0,
            first_token_ms=None,
            tokens_generated=3,
            from_cache=False,
        ),
    )

    def fail_tts(*_args, **_kwargs):
        raise RuntimeError("PATIENT_SECRET_BACKEND_DETAIL")

    monkeypatch.setattr(pipeline.tts_engine, "synthesize", fail_tts)
    caplog.set_level("INFO", logger="src.pipeline.orchestrator")

    result = pipeline.translate_speech(
        np.zeros(1600, dtype=np.float32),
        source_lang="en",
        target_lang="vi",
    )

    assert result.safety_passed
    assert result.translated_text == "Kiểm tra mạch"
    assert result.output_audio is None
    assert result.degraded_mode == "text_only"
    assert result.degradation_code == "tts_runtime_error"
    assert result.latency_breakdown["tts_ms"] >= 0.0
    stats = pipeline.get_performance_stats()
    assert stats["total_translations"] == 1
    assert stats["text_only_tts_fallbacks"] == 1
    assert stats["text_only_tts_fallback_rate"] == 1.0
    assert "PATIENT_SECRET_BACKEND_DETAIL" not in caplog.text
    assert "degradation_code=tts_runtime_error" in caplog.text


def test_pipeline_propagates_tts_failure_when_fallback_is_disabled(
    monkeypatch,
    mark_pipeline_ready,
):
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")
    pipeline.allow_text_only_tts_fallback = False
    mark_pipeline_ready(pipeline)
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
    monkeypatch.setattr(pipeline.flash_cache, "lookup", lambda *_args: None)
    monkeypatch.setattr(
        pipeline.mt_engine,
        "translate",
        lambda text, source_lang, target_lang: MTResult(
            source_text=text,
            translated_text="Kiểm tra mạch",
            source_lang=source_lang,
            target_lang=target_lang,
            latency_ms=1.0,
            first_token_ms=None,
            tokens_generated=3,
            from_cache=False,
        ),
    )

    def fail_tts(*_args, **_kwargs):
        raise RuntimeError("tts failed")

    monkeypatch.setattr(pipeline.tts_engine, "synthesize", fail_tts)

    with pytest.raises(RuntimeError, match="tts failed"):
        pipeline.translate_speech(
            np.zeros(1600, dtype=np.float32),
            source_lang="en",
            target_lang="vi",
        )

    assert pipeline.get_performance_stats()["total_translations"] == 0

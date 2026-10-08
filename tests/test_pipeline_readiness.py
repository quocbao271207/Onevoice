import sys
from types import SimpleNamespace

import pytest

from src.pipeline.asr_engine import ASREngine
from src.pipeline.audio_frontend import AudioFrontend
from src.pipeline.flash_cache import FlashCache
from src.pipeline.mt_engine import MTEngine
from src.pipeline.orchestrator import MediVoicePipeline
from src.pipeline.tts_engine import TTSEngine


class FakeStage:
    def __init__(self, *, becomes_ready=True):
        self.is_ready = False
        self.becomes_ready = becomes_ready
        self.load_count = 0

    def load(self):
        self.load_count += 1
        self.is_ready = self.becomes_ready


class FakeCache(FakeStage):
    def get_stats(self):
        return {"total_phrases": 1 if self.is_ready else 0}


def fake_pipeline(*, tts_ready=True, cache_enabled=True):
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")
    pipeline.audio_frontend = FakeStage()
    pipeline.asr_engine = FakeStage()
    pipeline.mt_engine = FakeStage()
    pipeline.tts_engine = FakeStage(becomes_ready=tts_ready)
    pipeline.flash_cache = FakeCache()
    pipeline.flash_cache_enabled = cache_enabled
    return pipeline


def test_pipeline_load_fails_closed_when_stage_does_not_become_ready():
    pipeline = fake_pipeline(tts_ready=False)

    with pytest.raises(RuntimeError, match="tts"):
        pipeline.load()

    status = pipeline.get_status()
    assert status["ready"] is False
    assert status["loaded"] is False
    assert status["components"]["tts"] is False


def test_pipeline_load_is_idempotent_after_verified_readiness():
    pipeline = fake_pipeline()

    pipeline.load()
    first_status = pipeline.get_status()
    pipeline.load()

    assert first_status["ready"] is True
    assert pipeline.get_status()["ready"] is True
    for stage in (
        pipeline.audio_frontend,
        pipeline.asr_engine,
        pipeline.mt_engine,
        pipeline.tts_engine,
        pipeline.flash_cache,
    ):
        assert stage.load_count == 1


def test_disabled_cache_is_not_required_for_pipeline_readiness():
    pipeline = fake_pipeline(cache_enabled=False)

    pipeline.load()

    status = pipeline.get_status()
    assert status["ready"] is True
    assert status["flash_cache_enabled"] is False
    assert status["components"]["flash_cache"] is False
    assert pipeline.flash_cache.load_count == 0


def test_engine_readiness_requires_complete_runtime_objects():
    frontend = AudioFrontend()
    frontend._is_loaded = True
    frontend.vad._is_loaded = True
    frontend.vad.model = object()
    assert frontend.is_ready is True

    asr = ASREngine()
    asr._is_loaded = True
    asr.models = {"vi": object()}
    asr.processors = {"vi": object()}
    asr.loaded_model_paths = {"vi": "vi-model"}
    assert asr.is_ready is False
    asr.models["en"] = object()
    asr.processors["en"] = object()
    asr.loaded_model_paths["en"] = "en-model"
    assert asr.is_ready is True

    mt = MTEngine()
    mt._is_loaded = True
    mt.model = object()
    assert mt.is_ready is False
    mt.tokenizer = object()
    mt.loaded_model_path = "mt-model"
    assert mt.is_ready is True

    tts = TTSEngine()
    tts._is_loaded = True
    tts._voices = {"vi": object()}
    assert tts.is_ready is False
    tts._voices["en"] = object()
    assert tts.is_ready is True

    cache = FlashCache()
    cache._is_loaded = True
    assert cache.is_ready is False
    cache.cache = {"vi:test": object()}
    assert cache.is_ready is True


def test_tts_load_does_not_publish_one_language_when_second_fails(
    tmp_path,
    monkeypatch,
):
    vi_path = tmp_path / "vi.onnx"
    en_path = tmp_path / "en.onnx"
    for path in (vi_path, en_path):
        path.write_bytes(b"model")
        path.with_suffix(f"{path.suffix}.json").write_text(
            "{}",
            encoding="utf-8",
        )

    calls = []

    class FakePiperVoice:
        @staticmethod
        def load(model_path, *, config_path):
            calls.append((model_path, config_path))
            if len(calls) == 2:
                raise RuntimeError("English voice failed")
            return object()

    monkeypatch.setitem(
        sys.modules,
        "piper.voice",
        SimpleNamespace(PiperVoice=FakePiperVoice),
    )
    engine = TTSEngine(
        vi_model_path=str(vi_path),
        en_model_path=str(en_path),
    )

    with pytest.raises(RuntimeError, match="English voice failed"):
        engine.load()

    assert engine._voices == {}
    assert engine.is_ready is False

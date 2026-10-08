import sys
from types import SimpleNamespace

import numpy as np
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


def test_runtime_entrypoints_recheck_component_readiness(monkeypatch):
    pipeline = fake_pipeline(tts_ready=False)
    pipeline._is_loaded = True
    pipeline.audio_frontend.is_ready = True
    pipeline.asr_engine.is_ready = True
    pipeline.mt_engine.is_ready = True
    pipeline.flash_cache.is_ready = True

    def unexpected_runtime_call(*_args, **_kwargs):
        raise AssertionError("Unready pipeline must stop before runtime work")

    pipeline.flash_cache.lookup = unexpected_runtime_call
    pipeline.audio_frontend.stream_from_microphone = unexpected_runtime_call
    monkeypatch.setattr(
        pipeline.audio_frontend,
        "get_status",
        unexpected_runtime_call,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="tts"):
        pipeline.translate_text("Stable.", "en", "vi")
    with pytest.raises(RuntimeError, match="tts"):
        pipeline.translate_speech(
            np.zeros(160, dtype=np.float32),
            source_lang="en",
            target_lang="vi",
        )
    with pytest.raises(RuntimeError, match="tts"):
        pipeline.run_interactive()


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


def test_component_entrypoints_recheck_complete_readiness():
    asr = ASREngine()
    asr._is_loaded = True
    with pytest.raises(RuntimeError, match="not ready"):
        asr.detect_language(np.zeros(160, dtype=np.float32), 16000)
    with pytest.raises(RuntimeError, match="not ready"):
        asr.transcribe(
            np.zeros(160, dtype=np.float32),
            language="vi",
            sample_rate=16000,
        )

    mt = MTEngine()
    mt._is_loaded = True
    with pytest.raises(RuntimeError, match="not ready"):
        mt.translate("Stable.", "en", "vi")

    tts = TTSEngine()
    tts._is_loaded = True
    tts._voices = {"vi": object()}
    with pytest.raises(RuntimeError, match="not ready"):
        tts.synthesize("Stable.", "en")

    cache = FlashCache()
    cache._is_loaded = True
    with pytest.raises(RuntimeError, match="not ready"):
        cache.lookup("Stable.", "en")


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


def test_asr_load_does_not_publish_vietnamese_when_english_fails(
    monkeypatch,
):
    class FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return None

    class FakeProcessorFactory:
        @staticmethod
        def from_pretrained(_path, **_kwargs):
            return object()

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(path, **_kwargs):
            if path == "en-configured":
                raise RuntimeError("English ASR failed")
            return FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            float16=object(),
            float32=object(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForSpeechSeq2Seq=FakeModelFactory,
            AutoProcessor=FakeProcessorFactory,
        ),
    )
    engine = ASREngine(
        vi_model_path="vi-configured",
        en_model_path="en-configured",
        allow_base_fallback=False,
    )

    with pytest.raises(RuntimeError, match="configured ASR checkpoint for en"):
        engine.load()

    assert engine.device == "auto"
    assert engine.models == {}
    assert engine.processors == {}
    assert engine.loaded_model_paths == {}
    assert engine.is_ready is False


def test_mt_load_does_not_publish_model_before_eval_succeeds(monkeypatch):
    class FailingEvalModel:
        def eval(self):
            raise RuntimeError("MT eval failed")

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(_path, **_kwargs):
            return object()

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(_path, **_kwargs):
            return FailingEvalModel()

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            float16=object(),
            float32=object(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForSeq2SeqLM=FakeModelFactory,
            AutoTokenizer=FakeTokenizerFactory,
        ),
    )
    engine = MTEngine(
        model_path="mt-configured",
        allow_base_fallback=False,
    )

    with pytest.raises(RuntimeError, match="configured MT checkpoint"):
        engine.load()

    assert engine.device == "auto"
    assert engine.model is None
    assert engine.tokenizer is None
    assert engine.loaded_model_path is None
    assert engine.is_ready is False

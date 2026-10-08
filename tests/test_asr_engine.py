from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.pipeline.asr_engine import ASREngine
from src.pipeline.orchestrator import MediVoicePipeline


class FakeProcessed(dict):
    def __init__(self):
        input_features = torch.zeros((1, 80, 3000), dtype=torch.float32)
        attention_mask = torch.ones((1, 3000), dtype=torch.long)
        super().__init__(attention_mask=attention_mask)
        self.input_features = input_features
        self.attention_mask = attention_mask


class FakeTokenizer:
    eos_token_id = 1
    pad_token_id = 0


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __init__(self, *, decoded_text="Dùng aspirin."):
        self.decoded_text = decoded_text
        self.decode_calls = 0

    def __call__(self, *_args, **_kwargs):
        return FakeProcessed()

    def get_decoder_prompt_ids(self, **_kwargs):
        return None

    def batch_decode(self, _sequences, **_kwargs):
        self.decode_calls += 1
        return [self.decoded_text]


class FakeModel:
    def __init__(self, sequence):
        self.sequence = torch.tensor([sequence], dtype=torch.long)

    def generate(self, *_args, **_kwargs):
        return SimpleNamespace(sequences=self.sequence, scores=[])


def loaded_engine(sequence) -> tuple[ASREngine, FakeProcessor]:
    engine = ASREngine(device="cpu", languages=("vi",))
    processor = FakeProcessor()
    engine.models = {"vi": FakeModel(sequence)}
    engine.processors = {"vi": processor}
    engine.loaded_model_paths = {"vi": "vi-model"}
    engine._is_loaded = True
    return engine, processor


def test_asr_rejects_audio_beyond_whisper_window_before_model_dispatch():
    engine, _processor = loaded_engine([2, 1, 0])
    engine.max_input_duration_seconds = 1.0

    with pytest.raises(ValueError, match="1.000s.*limit is 1.000s"):
        engine.transcribe(
            np.zeros(16001, dtype=np.float32),
            language="vi",
            sample_rate=16000,
        )
    with pytest.raises(ValueError, match="1.000s.*limit is 1.000s"):
        engine.detect_language(np.zeros(16001, dtype=np.float32), 16000)


def test_asr_rejects_generation_without_terminal_eos_before_decode():
    engine, processor = loaded_engine([2, 10, 11])

    with pytest.raises(RuntimeError, match="did not produce EOS"):
        engine.transcribe(
            np.zeros(1600, dtype=np.float32),
            language="vi",
            sample_rate=16000,
        )

    assert processor.decode_calls == 0


def test_asr_accepts_completed_generation_without_logging_transcript(
    caplog,
):
    engine, processor = loaded_engine([2, 10, 1, 0])
    processor.decoded_text = "PATIENT_SECRET_ASR aspirin."
    caplog.set_level("INFO", logger="src.pipeline.asr_engine")

    result = engine.transcribe(
        np.zeros(1600, dtype=np.float32),
        language="vi",
        sample_rate=16000,
    )

    assert result.text == "PATIENT_SECRET_ASR aspirin."
    assert processor.decode_calls == 1
    assert "PATIENT_SECRET_ASR" not in caplog.text
    with pytest.raises(ValueError, match="Unsupported ASR language"):
        engine.transcribe(
            np.zeros(1600, dtype=np.float32),
            language="fr",
            sample_rate=16000,
        )


def test_asr_rejects_nonfinite_confidence_proxy():
    class NonfiniteConfidenceModel(FakeModel):
        def generate(self, *_args, **_kwargs):
            return SimpleNamespace(
                sequences=self.sequence,
                scores=[torch.zeros((1, 1), dtype=torch.float32)],
            )

        def compute_transition_scores(self, *_args, **_kwargs):
            return torch.tensor([[float("nan")]], dtype=torch.float32)

    engine = ASREngine(device="cpu", languages=("vi",))
    engine.models = {"vi": NonfiniteConfidenceModel([2, 10, 1, 0])}
    engine.processors = {"vi": FakeProcessor()}
    engine.loaded_model_paths = {"vi": "vi-model"}
    engine._is_loaded = True

    with pytest.raises(RuntimeError, match="confidence proxy"):
        engine.transcribe(
            np.zeros(1600, dtype=np.float32),
            language="vi",
            sample_rate=16000,
        )


def test_asr_constructor_rejects_unsafe_generation_contract():
    with pytest.raises(ValueError, match="max_input_duration_seconds"):
        ASREngine(max_input_duration_seconds=30.1)
    with pytest.raises(ValueError, match="max_new_tokens"):
        ASREngine(max_new_tokens=0)
    with pytest.raises(ValueError, match="languages"):
        ASREngine(languages=("vi", "fr"))


def test_asr_language_detection_never_defaults_after_runtime_failure():
    engine = ASREngine()
    with pytest.raises(RuntimeError, match="not ready"):
        engine.detect_language(np.zeros(1600, dtype=np.float32), 16000)

    engine = ASREngine(languages=("vi",))
    engine._is_loaded = True
    engine.processors = {"vi": FakeProcessor()}
    engine.models = {"vi": object()}
    engine.loaded_model_paths = {"vi": "vi-model"}
    with pytest.raises(RuntimeError, match="Language detection failed"):
        engine.detect_language(np.zeros(1600, dtype=np.float32), 16000)


def test_pipeline_rejects_overlength_audio_before_frontend_dispatch(
    monkeypatch,
    mark_pipeline_ready,
):
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")
    mark_pipeline_ready(pipeline)
    pipeline.asr_engine.max_input_duration_seconds = 0.01

    def unexpected_frontend(*_args, **_kwargs):
        raise AssertionError("Overlength audio must stop before denoising")

    monkeypatch.setattr(
        pipeline.audio_frontend.denoiser,
        "suppress",
        unexpected_frontend,
    )

    with pytest.raises(ValueError, match="limit is 0.010s"):
        pipeline.translate_speech(
            np.zeros(161, dtype=np.float32),
            source_lang="vi",
            sample_rate=16000,
        )


def test_pipeline_marks_empty_asr_transcript_as_safety_blocked(
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
            text="   ",
            language="en",
            confidence=0.0,
            latency_ms=1.0,
        ),
    )

    def unexpected_downstream(*_args, **_kwargs):
        raise AssertionError("Empty ASR transcript must stop before cache/MT/TTS")

    monkeypatch.setattr(pipeline.flash_cache, "lookup", unexpected_downstream)
    monkeypatch.setattr(pipeline.mt_engine, "translate", unexpected_downstream)
    monkeypatch.setattr(pipeline.tts_engine, "synthesize", unexpected_downstream)

    result = pipeline.translate_speech(
        np.zeros(1600, dtype=np.float32),
        source_lang="en",
        target_lang="vi",
    )

    assert not result.safety_passed
    assert result.safety_issues == ["empty_asr_transcript"]
    assert result.translated_text == ""
    assert result.output_audio is None


def test_pipeline_applies_asr_generation_contract_configuration():
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")

    assert pipeline.asr_engine.max_input_duration_seconds == 30.0
    assert pipeline.asr_engine.max_new_tokens == 225

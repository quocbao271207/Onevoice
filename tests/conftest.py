import pytest


@pytest.fixture
def mark_pipeline_ready():
    """Publish a complete deterministic runtime contract for pipeline tests."""

    def mark(pipeline):
        pipeline.audio_frontend._is_loaded = True
        pipeline.audio_frontend.vad._is_loaded = True
        pipeline.audio_frontend.vad.model = object()

        languages = pipeline.asr_engine.languages
        pipeline.asr_engine.models = {
            language: object() for language in languages
        }
        pipeline.asr_engine.processors = {
            language: object() for language in languages
        }
        pipeline.asr_engine.loaded_model_paths = {
            language: f"{language}-model" for language in languages
        }
        pipeline.asr_engine._is_loaded = True

        pipeline.mt_engine.model = object()
        pipeline.mt_engine.tokenizer = object()
        pipeline.mt_engine.loaded_model_path = "mt-model"
        pipeline.mt_engine._is_loaded = True

        pipeline.tts_engine._voices = {
            language: object()
            for language in pipeline.tts_engine.model_paths
        }
        pipeline.tts_engine._is_loaded = True

        if pipeline.flash_cache_enabled:
            pipeline.flash_cache.load()

        pipeline._is_loaded = True
        assert pipeline.get_status()["ready"] is True
        return pipeline

    return mark

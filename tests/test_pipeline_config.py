from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.pipeline.mt_engine import MTResult, MedicalLexicon
from src.pipeline.orchestrator import MediVoicePipeline


ROOT = Path(__file__).resolve().parents[1]


def canonical_config() -> dict:
    return deepcopy(
        yaml.safe_load(
            (ROOT / "configs" / "pipeline_config.yaml").read_text(encoding="utf-8")
        )
    )


def write_config(path: Path, config: dict) -> Path:
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_explicit_pipeline_config_must_exist_and_be_a_unique_mapping(tmp_path):
    with pytest.raises(FileNotFoundError, match="Pipeline config does not exist"):
        MediVoicePipeline(config_path=str(tmp_path / "missing.yaml"))

    sequence = tmp_path / "sequence.yaml"
    sequence.write_text("- runtime\n- audio\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        MediVoicePipeline(config_path=str(sequence))

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "runtime:\n  allow_base_model_fallback: false\n"
        "runtime:\n  allow_base_model_fallback: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key.*runtime"):
        MediVoicePipeline(config_path=str(duplicate))


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("runtime", "allow_base_model_fallback"),
        ("audio", "noise_suppression_enabled"),
        ("flash_cache", "enabled"),
        ("flash_cache", "fuzzy_matching"),
    ],
)
def test_pipeline_config_rejects_string_booleans(tmp_path, section, key):
    config = canonical_config()
    config[section][key] = "false"
    path = write_config(tmp_path / f"{section}-{key}.yaml", config)

    with pytest.raises(ValueError, match=f"{section}.{key} must be a boolean"):
        MediVoicePipeline(config_path=str(path))


def test_pipeline_config_rejects_unknown_sections_and_wrong_topology(tmp_path):
    config = canonical_config()
    config["runtiem"] = {}
    with pytest.raises(ValueError, match="unknown top-level sections.*runtiem"):
        MediVoicePipeline(config_path=str(write_config(tmp_path / "unknown.yaml", config)))

    config = canonical_config()
    config["pipeline"]["language_pairs"] = [{"source": "vi", "target": "en"}]
    with pytest.raises(ValueError, match="exactly vi_to_en and en_to_vi"):
        MediVoicePipeline(config_path=str(write_config(tmp_path / "pairs.yaml", config)))

    config = canonical_config()
    config["pipeline"]["streaming"] = True
    with pytest.raises(ValueError, match="streaming=true is not implemented"):
        MediVoicePipeline(config_path=str(write_config(tmp_path / "streaming.yaml", config)))


def test_pipeline_config_rejects_unknown_or_missing_nested_keys(tmp_path):
    config = canonical_config()
    config["audio"]["sample_rae"] = config["audio"].pop("sample_rate")
    with pytest.raises(ValueError, match="audio has unknown keys.*sample_rae"):
        MediVoicePipeline(config_path=str(write_config(tmp_path / "typo.yaml", config)))

    config = canonical_config()
    del config["asr"]["max_new_tokens"]
    with pytest.raises(ValueError, match="asr is missing required keys.*max_new_tokens"):
        MediVoicePipeline(config_path=str(write_config(tmp_path / "missing-key.yaml", config)))

    config = canonical_config()
    config["tts"]["vi"]["model_pth"] = config["tts"]["vi"].pop("model_path")
    with pytest.raises(ValueError, match=r"tts\.vi has unknown keys.*model_pth"):
        MediVoicePipeline(config_path=str(write_config(tmp_path / "nested-typo.yaml", config)))


def test_pipeline_config_rejects_unimplemented_decoding_modes(tmp_path):
    config = canonical_config()
    config["flash_cache"]["fuzzy_matching"] = True
    with pytest.raises(ValueError, match="fuzzy_matching=true is not supported"):
        MediVoicePipeline(config_path=str(write_config(tmp_path / "fuzzy.yaml", config)))

    config = canonical_config()
    config["mt"]["speculative_decoding"] = True
    with pytest.raises(ValueError, match="speculative_decoding=true is not implemented"):
        MediVoicePipeline(config_path=str(write_config(tmp_path / "speculative.yaml", config)))

    config = canonical_config()
    config["mt"]["temperature"] = 0.1
    with pytest.raises(ValueError, match="temperature must be 0"):
        MediVoicePipeline(config_path=str(write_config(tmp_path / "sampling.yaml", config)))


def test_disabled_flash_cache_is_neither_loaded_nor_queried(tmp_path, monkeypatch):
    config = canonical_config()
    config["flash_cache"]["enabled"] = False
    path = write_config(tmp_path / "cache-disabled.yaml", config)
    pipeline = MediVoicePipeline(config_path=str(path))

    monkeypatch.setattr(pipeline.audio_frontend, "load", lambda: None)
    monkeypatch.setattr(pipeline.asr_engine, "load", lambda: None)
    monkeypatch.setattr(pipeline.mt_engine, "load", lambda: None)
    monkeypatch.setattr(pipeline.tts_engine, "load", lambda: None)

    def unexpected_cache_call(*_args, **_kwargs):
        raise AssertionError("Disabled cache must not be loaded or queried")

    monkeypatch.setattr(pipeline.flash_cache, "load", unexpected_cache_call)
    monkeypatch.setattr(pipeline.flash_cache, "lookup", unexpected_cache_call)
    pipeline.load()
    monkeypatch.setattr(
        pipeline.mt_engine,
        "translate",
        lambda text, source_lang, target_lang: MTResult(
            source_text=text,
            translated_text="Dùng aspirin 5 mg.",
            source_lang=source_lang,
            target_lang=target_lang,
            latency_ms=1.0,
            first_token_ms=None,
            tokens_generated=5,
            from_cache=False,
        ),
    )

    result = pipeline.translate_text("Give aspirin 5 mg.", "en", "vi")

    assert not result.from_cache
    assert pipeline.flash_cache.get_stats()["total_phrases"] == 0


def test_pipeline_applies_audio_shape_configuration(tmp_path):
    config = canonical_config()
    config["audio"]["bit_depth"] = 24
    config["audio"]["channels"] = 2
    pipeline = MediVoicePipeline(
        config_path=str(write_config(tmp_path / "audio.yaml", config))
    )

    assert pipeline.audio_frontend.config.bit_depth == 24
    assert pipeline.audio_frontend.config.channels == 2


def test_explicit_medical_lexicon_path_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError, match="medical lexicon"):
        MedicalLexicon(str(tmp_path / "missing.json"))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"vi_to_en": {}, "vi_to_en": {}}', "duplicate key.*vi_to_en"),
        ('{"wrong_direction": {"a": "b"}}', "unknown sections.*wrong_direction"),
        ('{"vi_to_en": []}', "vi_to_en must be an object"),
        ('{"vi_to_en": {"aspirin": ""}}', "values must be non-empty strings"),
        ("{}", "must contain at least one term"),
    ],
)
def test_medical_lexicon_rejects_ambiguous_or_invalid_json(
    tmp_path,
    payload,
    message,
):
    path = tmp_path / "invalid-lexicon.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        MedicalLexicon(str(path))


def test_medical_lexicon_validates_before_publishing_updates(tmp_path):
    lexicon = MedicalLexicon()
    original_vi_to_en = dict(lexicon.vi_to_en)
    original_en_to_vi = dict(lexicon.en_to_vi)
    invalid = tmp_path / "partially-invalid.json"
    invalid.write_text(
        '{"vi_to_en": {"đau ngực": "chest pain"}, '
        '"en_to_vi": {"dyspnea": null}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="values must be non-empty strings"):
        lexicon._load_from_file(str(invalid))

    assert lexicon.vi_to_en == original_vi_to_en
    assert lexicon.en_to_vi == original_en_to_vi


def test_medical_lexicon_merges_valid_bidirectional_terms(tmp_path):
    path = tmp_path / "valid-lexicon.json"
    path.write_text(
        '{"vi_to_en": {"đau ngực": "chest pain"}, '
        '"en_to_vi": {"dyspnea": "khó thở"}}',
        encoding="utf-8",
    )

    lexicon = MedicalLexicon(str(path))

    assert lexicon.vi_to_en["đau ngực"] == "chest pain"
    assert lexicon.en_to_vi["dyspnea"] == "khó thở"

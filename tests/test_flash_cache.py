import json

import pytest

from src.pipeline.flash_cache import CachedPhrase, EMERGENCY_PHRASES, FlashCache
from src.pipeline.orchestrator import MediVoicePipeline
from src.pipeline.safety_guard import validate_translation


def write_cache(path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def valid_custom_phrase(**overrides):
    phrase = {
        "source": "Give aspirin 5 mg.",
        "source_lang": "en",
        "translation": "Dùng aspirin 5 mg.",
        "target_lang": "vi",
        "category": "medication",
        "requires_confirmation": True,
    }
    phrase.update(overrides)
    return phrase


def test_default_emergency_cache_is_structurally_safe():
    cache = FlashCache()
    cache.load()

    assert cache.get_stats()["total_phrases"] == sum(
        len(phrases) for phrases in EMERGENCY_PHRASES.values()
    )
    for phrase in cache.cache.values():
        check = validate_translation(
            phrase.source_text,
            phrase.translated_text,
            phrase.source_lang,
            phrase.target_lang,
        )
        assert check.safe, (phrase.source_text, check.issues)


def test_custom_cache_load_is_atomic_and_missing_configured_file_fails(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="custom flash cache"):
        FlashCache(str(missing)).load()

    invalid = tmp_path / "invalid.json"
    write_cache(
        invalid,
        [
            valid_custom_phrase(source="Give aspirin 5 mg."),
            valid_custom_phrase(source="Give warfarin 5 mg.", target_lang="en"),
        ],
    )
    cache = FlashCache(str(invalid))
    with pytest.raises(ValueError, match="opposite bilingual direction"):
        cache.load()
    assert cache.cache == {}
    assert not cache._is_loaded


def test_custom_cache_requires_confirmation_and_safe_content(tmp_path):
    unsafe_confirmation = tmp_path / "unsafe-confirmation.json"
    write_cache(
        unsafe_confirmation,
        [valid_custom_phrase(requires_confirmation=False)],
    )
    with pytest.raises(ValueError, match="requires_confirmation=true"):
        FlashCache(str(unsafe_confirmation)).load()

    unsafe_translation = tmp_path / "unsafe-translation.json"
    write_cache(
        unsafe_translation,
        [valid_custom_phrase(translation="Dùng aspirin 50 mg.")],
    )
    with pytest.raises(ValueError, match="failed safety validation"):
        FlashCache(str(unsafe_translation)).load()


def test_custom_cache_rejects_unknown_fields_and_normalized_collisions(tmp_path):
    unknown = tmp_path / "unknown.json"
    write_cache(unknown, [valid_custom_phrase(typo_translation="unsafe")])
    with pytest.raises(ValueError, match="unknown fields"):
        FlashCache(str(unknown)).load()

    collision = tmp_path / "collision.json"
    write_cache(
        collision,
        [
            valid_custom_phrase(source="Give aspirin 5 mg."),
            valid_custom_phrase(source="  GIVE aspirin 5 mg!  "),
        ],
    )
    with pytest.raises(ValueError, match="duplicate normalized source"):
        FlashCache(str(collision)).load()


def test_cache_normalization_preserves_decimal_identity_and_disables_fuzzy():
    cache = FlashCache()
    assert cache._make_key("Give 5.0 mg.", "en") != cache._make_key(
        "Give 50 mg.",
        "en",
    )

    with pytest.raises(ValueError, match="Fuzzy flash-cache matching is disabled"):
        FlashCache(allow_fuzzy=True)


def test_valid_custom_cache_is_loaded_without_logging_raw_phrase(tmp_path, caplog):
    custom = tmp_path / "custom.json"
    write_cache(custom, [valid_custom_phrase()])
    cache = FlashCache(str(custom))

    cache.load()
    caplog.set_level("INFO", logger="src.pipeline.flash_cache")
    phrase = cache.lookup("Give aspirin 5 mg!", "en")

    assert phrase is not None
    assert phrase.target_lang == "vi"
    assert phrase.requires_confirmation
    assert "Give aspirin 5 mg" not in caplog.text
    assert "Dùng aspirin 5 mg" not in caplog.text


def test_pipeline_rejects_cache_hit_for_wrong_target_language(
    monkeypatch,
    mark_pipeline_ready,
):
    pipeline = MediVoicePipeline(config_path="configs/pipeline_config.yaml")
    mark_pipeline_ready(pipeline)
    monkeypatch.setattr(
        pipeline.flash_cache,
        "lookup",
        lambda *_args: CachedPhrase(
            source_text="Give aspirin 5 mg.",
            source_lang="en",
            translated_text="Give aspirin 5 mg.",
            target_lang="en",
            requires_confirmation=True,
        ),
    )

    with pytest.raises(RuntimeError, match="direction mismatch"):
        pipeline.translate_text("Give aspirin 5 mg.", "en", "vi")

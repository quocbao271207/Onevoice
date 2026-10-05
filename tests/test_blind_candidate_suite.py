from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_blind_candidate_suite import (
    blind_quality_failures,
    sha256,
    validate_content_integrity,
    validate_identifiers,
    verify_lock,
)
from src.data.quality import fingerprint_text


def write_lock(path: Path, mt: Path, asr: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "locked_unopened",
                "manifests": {
                    "mt": {"path": str(mt), "sha256": sha256(mt)},
                    "asr": {"path": str(asr), "sha256": sha256(asr)},
                },
            }
        ),
        encoding="utf-8",
    )


def test_blind_lock_verifies_checksums_and_detects_tampering(tmp_path: Path):
    mt = tmp_path / "mt.jsonl"
    asr = tmp_path / "asr.jsonl"
    mt.write_text('{"id":"mt"}\n', encoding="utf-8")
    asr.write_text('{"id":"asr"}\n', encoding="utf-8")
    lock = tmp_path / "lock.json"
    write_lock(lock, mt, asr)
    assert verify_lock(lock)["status"] == "locked_unopened"
    mt.write_text('{"id":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_lock(lock)


def test_placeholder_blind_lock_cannot_be_opened():
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="not locked"):
        verify_lock(root / "data/eval/blind_test_v2.lock.json")


@pytest.mark.parametrize(
    ("task", "rows", "message"),
    [
        ("mt", [{"pair_fingerprint": "pair"}], "missing id"),
        ("mt", [{"id": "mt-1"}], "missing pair_fingerprint"),
        (
            "asr",
            [{"id": "asr-1", "audio_sha256": "audio"}],
            "missing text_fingerprint",
        ),
        (
            "asr",
            [{"id": "asr-1", "text_fingerprint": "text-1"}],
            "missing audio_sha256",
        ),
    ],
)
def test_blind_identifiers_require_leakage_fingerprints(task, rows, message):
    with pytest.raises(ValueError, match=message):
        validate_identifiers(task, rows)


@pytest.mark.parametrize(
    ("task", "rows", "message"),
    [
        (
            "mt",
            [
                {"id": "mt", "pair_fingerprint": "pair-1"},
                {"id": "mt", "pair_fingerprint": "pair-2"},
            ],
            "duplicate id",
        ),
        (
            "mt",
            [
                {"id": "mt-1", "pair_fingerprint": "pair"},
                {"id": "mt-2", "pair_fingerprint": "pair"},
            ],
            "duplicate pair_fingerprint",
        ),
        (
            "asr",
            [
                {
                    "id": "asr-1",
                    "text_fingerprint": "text-1",
                    "audio_sha256": "audio",
                },
                {
                    "id": "asr-2",
                    "text_fingerprint": "text-2",
                    "audio_sha256": "audio",
                },
            ],
            "duplicate audio_sha256",
        ),
        (
            "asr",
            [
                {
                    "id": "asr",
                    "text_fingerprint": "text-1",
                    "audio_sha256": "audio-1",
                },
                {
                    "id": "asr",
                    "text_fingerprint": "text-2",
                    "audio_sha256": "audio-2",
                },
            ],
            "duplicate id",
        ),
    ],
)
def test_blind_identifiers_reject_duplicate_samples(task, rows, message):
    with pytest.raises(ValueError, match=message):
        validate_identifiers(task, rows)


def test_blind_mt_fingerprint_is_recomputed_from_text():
    source = "Take aspirin 81 mg daily."
    target = "Uống aspirin 81 mg mỗi ngày."
    row = {
        "id": "mt-1",
        "source_text": source,
        "target_text": target,
        "pair_fingerprint": fingerprint_text(source + "\x1f" + target),
    }
    validate_content_integrity("mt", [row])

    row["target_text"] = "Uống aspirin 810 mg mỗi ngày."
    with pytest.raises(ValueError, match="pair_fingerprint does not match"):
        validate_content_integrity("mt", [row])


@pytest.mark.parametrize("missing", ["source_text", "target_text"])
def test_blind_mt_requires_both_text_sides(missing):
    row = {
        "source_text": "No penicillin.",
        "target_text": "Không dùng penicillin.",
        "pair_fingerprint": "declared",
    }
    row.pop(missing)
    with pytest.raises(ValueError, match="missing source_text or target_text"):
        validate_content_integrity("mt", [row])


def test_blind_asr_fingerprints_are_recomputed_from_text_and_audio(tmp_path: Path):
    audio = tmp_path / "blind.flac"
    audio.write_bytes(b"blind-audio-payload")
    text = "Không dùng penicillin 500 mg."
    row = {
        "id": "asr-1",
        "text": text,
        "text_fingerprint": fingerprint_text(text),
        "audio_path": str(audio),
        "audio_sha256": sha256(audio),
    }
    validate_content_integrity("asr", [row])

    row["text"] = "Dùng penicillin 500 mg."
    with pytest.raises(ValueError, match="text_fingerprint does not match"):
        validate_content_integrity("asr", [row])

    row["text"] = text
    audio.write_bytes(b"tampered-audio-payload")
    with pytest.raises(ValueError, match="audio_sha256 does not match"):
        validate_content_integrity("asr", [row])


def test_blind_asr_requires_existing_audio(tmp_path: Path):
    text = "Không dùng penicillin."
    row = {
        "text": text,
        "text_fingerprint": fingerprint_text(text),
        "audio_path": str(tmp_path / "missing.flac"),
        "audio_sha256": "declared",
    }
    with pytest.raises(FileNotFoundError, match="audio is missing"):
        validate_content_integrity("asr", [row])


def accuracy_config():
    return {
        "release_gates": {
            "aggregate": {
                "asr_vi_wer_max": 0.19,
                "mt_sacrebleu_min": 25.0,
                "mt_chrf2_min": 46.0,
            }
        }
    }


def bakeoff_config():
    return {"promotion_gate": {"asr_code_switch_wer_max": 0.21}}


def test_blind_mt_quality_requires_lower_ci_bounds_above_policy():
    report = {
        "directions": {
            "en_to_vi": {
                "samples": 100,
                "sacrebleu": 30.0,
                "sacrebleu_bootstrap_95ci": [26.0, 34.0],
                "chrf2": 50.0,
                "chrf2_bootstrap_95ci": [47.0, 53.0],
            }
        }
    }
    assert blind_quality_failures(
        report, "mt", "en_to_vi", accuracy_config(), bakeoff_config()
    ) == []

    report["directions"]["en_to_vi"]["chrf2_bootstrap_95ci"] = [45.0, 53.0]
    assert blind_quality_failures(
        report, "mt", "en_to_vi", accuracy_config(), bakeoff_config()
    ) == ["chrf2_bootstrap_95ci:lower_bound_below_policy"]


def test_blind_asr_quality_requires_upper_ci_and_code_switch_policy():
    report = {
        "samples": 100,
        "wer": 0.15,
        "cer": 0.10,
        "wer_bootstrap_95ci": [0.13, 0.18],
        "slices": {"code_switch": {"True": {"wer": 0.20}}},
    }
    assert blind_quality_failures(
        report, "asr", None, accuracy_config(), bakeoff_config()
    ) == []

    report["wer_bootstrap_95ci"] = [0.13, 0.20]
    report["slices"]["code_switch"]["True"]["wer"] = float("nan")
    failures = blind_quality_failures(
        report, "asr", None, accuracy_config(), bakeoff_config()
    )
    assert failures == [
        "wer_bootstrap_95ci:upper_bound_above_policy",
        "code_switch_wer:missing_or_invalid",
    ]

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_blind_candidate_suite import sha256, validate_identifiers, verify_lock


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

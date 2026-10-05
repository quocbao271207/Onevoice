from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_blind_candidate_suite import sha256, verify_lock


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

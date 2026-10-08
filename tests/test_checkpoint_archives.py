from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.candidate_evidence import verify_evidence_archive
from scripts.watch_checkpoints import (
    archive_ready_checkpoints,
    completed_checkpoint,
    sha256,
    verified_existing_archive,
)


def make_checkpoint(root: Path, step: int, *, complete: bool = True) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": step,
                "log_history": [{"step": step, "eval_loss": 1.25}],
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    if complete:
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            (checkpoint / name).write_bytes(name.encode())
    return checkpoint


def test_incomplete_checkpoint_is_not_ready(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path, 500, complete=False)
    assert completed_checkpoint(checkpoint) is None


def test_checkpoint_archive_is_atomic_checksummed_and_idempotent(tmp_path: Path):
    model_dir = tmp_path / "model"
    checkpoint = make_checkpoint(model_dir, 500)
    archive_dir = tmp_path / "archives"

    first = archive_ready_checkpoints(model_dir, archive_dir)
    assert first[0]["status"] == "archived"
    assert first[0]["eval_loss"] == 1.25
    archive_path = archive_dir / "checkpoint-500.tar.gz"
    checksum_path = archive_dir / "checkpoint-500.tar.gz.sha256"
    content_manifest_path = archive_dir / "checkpoint-500.tar.gz.manifest.json"
    assert verified_existing_archive(archive_path, checksum_path)
    assert checksum_path.read_text(encoding="utf-8").split()[0] == sha256(archive_path)
    content_manifest = verify_evidence_archive(archive_path)
    assert content_manifest_path.is_file()
    assert {item["path"] for item in content_manifest["files"]} == {
        "checkpoint-500/adapter_model.safetensors",
        "checkpoint-500/optimizer.pt",
        "checkpoint-500/rng_state.pth",
        "checkpoint-500/scheduler.pt",
        "checkpoint-500/trainer_state.json",
    }
    assert first[0]["manifest"] == str(content_manifest_path.resolve())
    assert first[0]["manifest_sha256"] == sha256(content_manifest_path)
    assert first[0]["file_count"] == 5
    assert not list(archive_dir.glob("*.part"))
    assert not list(archive_dir.glob("*.tmp"))

    second = archive_ready_checkpoints(model_dir, archive_dir)
    assert second[0]["status"] == "already_verified"
    manifest = json.loads((archive_dir / "checkpoint_archives.json").read_text(encoding="utf-8"))
    assert [item["step"] for item in manifest["checkpoints"]] == [500]
    assert manifest["checkpoints"][0]["manifest_sha256"] == sha256(content_manifest_path)
    assert checkpoint.is_dir()


def test_existing_legacy_checkpoint_archive_gets_a_verified_content_manifest(tmp_path: Path):
    model_dir = tmp_path / "model"
    make_checkpoint(model_dir, 500)
    archive_dir = tmp_path / "archives"

    archive_ready_checkpoints(model_dir, archive_dir)
    content_manifest_path = archive_dir / "checkpoint-500.tar.gz.manifest.json"
    content_manifest_path.unlink()

    records = archive_ready_checkpoints(model_dir, archive_dir)

    assert records[0]["status"] == "already_verified"
    assert records[0]["manifest"] == str(content_manifest_path.resolve())
    assert verify_evidence_archive(archive_dir / "checkpoint-500.tar.gz")["file_count"] == 5


def test_corrupt_checkpoint_content_manifest_is_rejected(tmp_path: Path):
    model_dir = tmp_path / "model"
    make_checkpoint(model_dir, 500)
    archive_dir = tmp_path / "archives"
    archive_ready_checkpoints(model_dir, archive_dir)
    manifest_path = archive_dir / "checkpoint-500.tar.gz.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="member verification failed"):
        archive_ready_checkpoints(model_dir, archive_dir)


def test_mismatched_trainer_step_is_not_archived(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path, 1000)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 999, "log_history": []}), encoding="utf-8"
    )
    assert completed_checkpoint(checkpoint) is None


def test_non_integer_or_duplicate_trainer_step_is_not_archived(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path, 1000)
    trainer_state = checkpoint / "trainer_state.json"
    trainer_state.write_text(
        '{"global_step":"1000","log_history":[]}',
        encoding="utf-8",
    )
    assert completed_checkpoint(checkpoint) is None

    trainer_state.write_text(
        '{"global_step":1000,"global_step":1000,"log_history":[]}',
        encoding="utf-8",
    )
    assert completed_checkpoint(checkpoint) is None


def test_corrupt_checkpoint_index_is_not_silently_replaced(tmp_path: Path):
    model_dir = tmp_path / "model"
    make_checkpoint(model_dir, 500)
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    index = archive_dir / "checkpoint_archives.json"
    index.write_text('{"checkpoints":NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        archive_ready_checkpoints(model_dir, archive_dir)

    assert index.read_text(encoding="utf-8") == '{"checkpoints":NaN}'

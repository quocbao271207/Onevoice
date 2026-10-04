from __future__ import annotations

import json
import tarfile
from pathlib import Path

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
    assert verified_existing_archive(archive_path, checksum_path)
    assert checksum_path.read_text(encoding="utf-8").split()[0] == sha256(archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        assert "checkpoint-500/trainer_state.json" in archive.getnames()
    assert not list(archive_dir.glob("*.part"))

    second = archive_ready_checkpoints(model_dir, archive_dir)
    assert second[0]["status"] == "already_verified"
    manifest = json.loads((archive_dir / "checkpoint_archives.json").read_text(encoding="utf-8"))
    assert [item["step"] for item in manifest["checkpoints"]] == [500]
    assert checkpoint.is_dir()


def test_mismatched_trainer_step_is_not_archived(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path, 1000)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 999, "log_history": []}), encoding="utf-8"
    )
    assert completed_checkpoint(checkpoint) is None

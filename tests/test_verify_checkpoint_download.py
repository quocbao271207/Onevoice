from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_checkpoint_download import validate_output_path, verify_checkpoint_download
from scripts.watch_checkpoints import archive_ready_checkpoints


def make_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": step,
                "max_steps": 1000,
                "best_metric": 12.5,
                "best_model_checkpoint": str(root / "checkpoint-250"),
                "log_history": [
                    {"step": step, "eval_loss": 0.75, "eval_wer": 13.0}
                ],
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(name.encode())
    return checkpoint


def archived_bundle(tmp_path: Path, step: int = 500) -> tuple[Path, Path]:
    model_dir = tmp_path / "model"
    make_checkpoint(model_dir, step)
    archive_dir = tmp_path / "archives"
    archive_ready_checkpoints(model_dir, archive_dir)
    return archive_dir / f"checkpoint-{step}.tar.gz", archive_dir / "checkpoint_archives.json"


def test_downloaded_checkpoint_verifier_binds_archive_index_and_trainer_state(
    tmp_path: Path,
):
    archive, index = archived_bundle(tmp_path)

    report = verify_checkpoint_download(archive, index)

    assert report["checkpoint_step"] == 500
    assert report["trainer_state"] == {
        "member": "checkpoint-500/trainer_state.json",
        "global_step": 500,
        "max_steps": 1000,
        "eval_loss": 0.75,
        "eval_wer": 13.0,
        "best_metric": 12.5,
        "best_model_checkpoint": str(tmp_path / "model" / "checkpoint-250"),
        "best_checkpoint_step": 250,
    }
    assert report["content_manifest"]["file_count"] == 5
    assert report["checks"]["all_member_hashes_match"] is True
    assert report["checks"]["duplicate_member_paths"] is False
    assert report["checks"]["special_members"] is False


def test_downloaded_checkpoint_verifier_rejects_tampered_index_identity(tmp_path: Path):
    archive, index = archived_bundle(tmp_path)
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["checkpoints"][0]["sha256"] = "0" * 64
    index.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="archive identity mismatch"):
        verify_checkpoint_download(archive, index)


def test_downloaded_checkpoint_verifier_rejects_missing_step_record(tmp_path: Path):
    archive, index = archived_bundle(tmp_path)
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["checkpoints"] = []
    index.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one record for step 500"):
        verify_checkpoint_download(archive, index)


def test_verification_output_cannot_overwrite_downloaded_evidence(tmp_path: Path):
    archive, index = archived_bundle(tmp_path)

    with pytest.raises(ValueError, match="cannot overwrite evidence"):
        validate_output_path(archive, (archive, index))

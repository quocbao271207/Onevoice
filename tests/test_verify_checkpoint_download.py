from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.candidate_evidence import (
    archive_evidence,
    evidence_sidecars,
    verify_evidence_archive,
)
from scripts.verify_checkpoint_download import (
    validate_output_path,
    verify_checkpoint_download,
    write_verification,
)
from scripts.watch_checkpoints import archive_ready_checkpoints
from src.utils.bounded_file import sha256_stable_regular_file


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


def archive_with_trainer_state(
    tmp_path: Path,
    trainer_state: str,
    *,
    step: int = 500,
) -> tuple[Path, Path]:
    checkpoint = tmp_path / "model" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(trainer_state, encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    archive, digest = archive_evidence(
        checkpoint,
        archive_dir / f"checkpoint-{step}.tar.gz",
    )
    _, manifest_path = evidence_sidecars(archive)
    manifest = verify_evidence_archive(archive)
    archive_digest, archive_bytes = sha256_stable_regular_file(
        archive,
        maximum_bytes=100_000_000,
        label="Test checkpoint archive",
    )
    manifest_digest, manifest_bytes = sha256_stable_regular_file(
        manifest_path,
        maximum_bytes=100_000_000,
        label="Test checkpoint manifest",
    )
    assert archive_digest == digest
    index = archive_dir / "checkpoint_archives.json"
    index.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "checkpoints": [
                    {
                        "step": step,
                        "checkpoint": str(checkpoint),
                        "archive": str(archive),
                        "bytes": archive_bytes,
                        "sha256": archive_digest,
                        "manifest": str(manifest_path),
                        "manifest_bytes": manifest_bytes,
                        "manifest_sha256": manifest_digest,
                        "file_count": manifest["file_count"],
                        "content_bytes": manifest["content_bytes"],
                        "eval_loss": 0.75,
                        "status": "archived",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return archive, index


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


def test_downloaded_checkpoint_verifier_rejects_duplicate_index_keys(tmp_path: Path):
    archive, index = archived_bundle(tmp_path)
    payload = index.read_text(encoding="utf-8")
    index.write_text(
        payload.replace('"checkpoints":', '"checkpoints": [], "checkpoints":', 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        verify_checkpoint_download(archive, index)


def test_downloaded_checkpoint_verifier_rejects_non_finite_index_number(
    tmp_path: Path,
):
    archive, index = archived_bundle(tmp_path)
    payload = index.read_text(encoding="utf-8")
    index.write_text(
        payload.replace('"eval_loss": 0.75', '"eval_loss": NaN', 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        verify_checkpoint_download(archive, index)


def test_downloaded_checkpoint_verifier_rejects_duplicate_trainer_state_keys(
    tmp_path: Path,
):
    archive, index = archive_with_trainer_state(
        tmp_path,
        '{"global_step": 500, "global_step": 500, "max_steps": 1000, '
        '"best_metric": 12.5, "best_model_checkpoint": "checkpoint-250", '
        '"log_history": [{"step": 500, "eval_loss": 0.75}]}',
    )

    with pytest.raises(
        ValueError,
        match="Trainer state is not valid strict UTF-8 JSON",
    ):
        verify_checkpoint_download(archive, index)


def test_downloaded_checkpoint_verifier_rejects_linked_index(tmp_path: Path):
    archive, index = archived_bundle(tmp_path)
    linked_index = tmp_path / "linked-index.json"
    try:
        linked_index.symlink_to(index)
    except OSError:
        pytest.skip("symlink creation unavailable")

    with pytest.raises(ValueError, match="symlink or junction"):
        verify_checkpoint_download(archive, linked_index)


def test_verification_output_cannot_overwrite_downloaded_evidence(tmp_path: Path):
    archive, index = archived_bundle(tmp_path)

    with pytest.raises(ValueError, match="cannot overwrite evidence"):
        validate_output_path(archive, (archive, index))


def test_verification_receipt_is_immutable_and_idempotent(tmp_path: Path):
    archive, index = archived_bundle(tmp_path)
    report = verify_checkpoint_download(archive, index)
    output = tmp_path / "checkpoint-500.local-verification.json"

    write_verification(output, report)
    persisted = output.read_bytes()
    repeated = dict(report)
    repeated["verified_at"] = datetime.now(timezone.utc).isoformat()
    write_verification(output, repeated)

    assert output.read_bytes() == persisted
    changed = dict(report)
    changed["archive"] = {**report["archive"], "sha256": "0" * 64}
    with pytest.raises(FileExistsError, match="different evidence"):
        write_verification(output, changed)

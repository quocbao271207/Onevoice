"""Verify a downloaded Trainer checkpoint evidence bundle without extracting it."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.candidate_evidence import (  # noqa: E402
    evidence_sidecars,
    sha256,
    verify_evidence_archive,
)


ARCHIVE_PATTERN = re.compile(r"^checkpoint-(\d+)\.tar\.gz$")


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid integer for {label}: {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {label}: {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"Invalid integer for {label}: {value!r}")
    if isinstance(value, str) and not re.fullmatch(r"\d+", value.strip()):
        raise ValueError(f"Invalid integer for {label}: {value!r}")
    if parsed < 0:
        raise ValueError(f"Invalid negative integer for {label}: {parsed}")
    return parsed


def _finite_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid number for {label}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite number for {label}: {value!r}")
    return parsed


def _checkpoint_step(archive_path: Path) -> int:
    match = ARCHIVE_PATTERN.fullmatch(archive_path.name)
    if not match:
        raise ValueError(
            "Archive name must be checkpoint-<step>.tar.gz: " f"{archive_path.name}"
        )
    return int(match.group(1))


def _checkpoint_record(index: dict[str, Any], step: int) -> dict[str, Any]:
    records = index.get("checkpoints")
    if not isinstance(records, list):
        raise ValueError("Checkpoint index must contain a checkpoints list")
    matches = [
        item
        for item in records
        if isinstance(item, dict) and _integer(item.get("step"), "index.step") == step
    ]
    if len(matches) != 1:
        raise ValueError(f"Checkpoint index must contain exactly one record for step {step}")
    return matches[0]


def _trainer_state(archive_path: Path, step: int) -> tuple[str, dict[str, Any]]:
    member_name = f"checkpoint-{step}/trainer_state.json"
    with tarfile.open(archive_path, "r:gz") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError as exc:
            raise ValueError(f"Missing trainer state member: {member_name}") from exc
        if not member.isfile():
            raise ValueError(f"Trainer state is not a regular file: {member_name}")
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"Cannot read trainer state member: {member_name}")
        try:
            payload = json.loads(handle.read())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid trainer state JSON: {member_name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Trainer state root must be an object: {member_name}")
    return member_name, payload


def _best_checkpoint_step(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"(?:^|[/\\])checkpoint-(\d+)$", value)
    return int(match.group(1)) if match else None


def _last_evaluation(trainer_state: dict[str, Any]) -> dict[str, Any]:
    history = trainer_state.get("log_history")
    if not isinstance(history, list):
        raise ValueError("trainer_state.log_history must be a list")
    for item in reversed(history):
        if isinstance(item, dict) and "eval_loss" in item:
            return item
    raise ValueError("trainer_state has no evaluation record")


def verify_checkpoint_download(
    archive_path: Path,
    checkpoint_index_path: Path,
) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    checkpoint_index_path = checkpoint_index_path.resolve()
    checksum_path, manifest_path = evidence_sidecars(archive_path)
    bundle_paths = (archive_path, checksum_path, manifest_path, checkpoint_index_path)
    for path in bundle_paths:
        if path.is_symlink():
            raise ValueError(f"Checkpoint evidence cannot be a symlink: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)

    step = _checkpoint_step(archive_path)
    manifest = verify_evidence_archive(archive_path)
    index = _json_object(checkpoint_index_path)
    record = _checkpoint_record(index, step)

    archive_digest = sha256(archive_path)
    manifest_digest = sha256(manifest_path)
    expected_status = record.get("status")
    if expected_status not in {"archived", "already_verified"}:
        raise ValueError(f"Checkpoint index record is not archived: {expected_status!r}")
    if Path(str(record.get("archive", ""))).name != archive_path.name:
        raise ValueError("Checkpoint index archive name mismatch")
    if (
        _integer(record.get("bytes"), "index.bytes") != archive_path.stat().st_size
        or str(record.get("sha256", "")).lower() != archive_digest
    ):
        raise ValueError("Checkpoint index archive identity mismatch")
    if Path(str(record.get("manifest", ""))).name != manifest_path.name:
        raise ValueError("Checkpoint index manifest name mismatch")
    if (
        _integer(record.get("manifest_bytes"), "index.manifest_bytes")
        != manifest_path.stat().st_size
        or str(record.get("manifest_sha256", "")).lower() != manifest_digest
    ):
        raise ValueError("Checkpoint index manifest identity mismatch")
    if (
        _integer(record.get("file_count"), "index.file_count")
        != int(manifest["file_count"])
        or _integer(record.get("content_bytes"), "index.content_bytes")
        != int(manifest["content_bytes"])
    ):
        raise ValueError("Checkpoint index content totals mismatch")

    member_name, trainer_state = _trainer_state(archive_path, step)
    global_step = _integer(trainer_state.get("global_step"), "trainer_state.global_step")
    if global_step != step:
        raise ValueError(
            f"Trainer global step does not match archive: {global_step}/{step}"
        )
    max_steps = _integer(trainer_state.get("max_steps"), "trainer_state.max_steps")
    if max_steps < global_step:
        raise ValueError(f"Trainer max_steps is below global_step: {max_steps}/{global_step}")
    evaluation = _last_evaluation(trainer_state)
    eval_loss = _finite_float(evaluation.get("eval_loss"), "trainer_state.eval_loss")
    eval_wer = _finite_float(evaluation.get("eval_wer"), "trainer_state.eval_wer")
    index_eval_loss = _finite_float(record.get("eval_loss"), "index.eval_loss")
    if eval_loss is None or index_eval_loss is None or not math.isclose(
        eval_loss, index_eval_loss, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Checkpoint index eval_loss does not match trainer state")

    best_model_checkpoint = trainer_state.get("best_model_checkpoint")
    return {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_step": step,
        "archive": {
            "name": archive_path.name,
            "bytes": archive_path.stat().st_size,
            "sha256": archive_digest,
        },
        "sidecar": {
            "name": checksum_path.name,
            "bytes": checksum_path.stat().st_size,
            "sha256": sha256(checksum_path),
        },
        "content_manifest": {
            "name": manifest_path.name,
            "bytes": manifest_path.stat().st_size,
            "sha256": manifest_digest,
            "file_count": int(manifest["file_count"]),
            "content_bytes": int(manifest["content_bytes"]),
        },
        "checkpoint_index": {
            "name": checkpoint_index_path.name,
            "bytes": checkpoint_index_path.stat().st_size,
            "sha256": sha256(checkpoint_index_path),
        },
        "trainer_state": {
            "member": member_name,
            "global_step": global_step,
            "max_steps": max_steps,
            "eval_loss": eval_loss,
            "eval_wer": eval_wer,
            "best_metric": _finite_float(
                trainer_state.get("best_metric"), "trainer_state.best_metric"
            ),
            "best_model_checkpoint": best_model_checkpoint,
            "best_checkpoint_step": _best_checkpoint_step(best_model_checkpoint),
        },
        "checks": {
            "archive_sha256_matches_sidecar": True,
            "archive_matches_content_manifest": True,
            "archive_matches_checkpoint_index": True,
            "manifest_matches_checkpoint_index": True,
            "all_member_hashes_match": True,
            "safe_member_paths": True,
            "duplicate_member_paths": False,
            "special_members": False,
            "file_count_matches": True,
            "content_bytes_match": True,
            "trainer_step_matches": True,
            "eval_loss_matches_checkpoint_index": True,
        },
    }


def write_verification(path: Path, payload: dict[str, Any]) -> None:
    if path.is_symlink():
        raise ValueError(f"Verification output cannot be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def validate_output_path(path: Path, reserved_paths: tuple[Path, ...]) -> Path:
    output = path.resolve()
    if output in {reserved.resolve() for reserved in reserved_paths}:
        raise ValueError(f"Verification output cannot overwrite evidence: {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint-index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    archive = args.archive.resolve()
    step = _checkpoint_step(archive)
    checkpoint_index = (
        args.checkpoint_index.resolve()
        if args.checkpoint_index is not None
        else archive.parent / "checkpoint_archives.json"
    )
    requested_output = (
        args.output.resolve()
        if args.output is not None
        else archive.parent / f"checkpoint-{step}.local-verification.json"
    )
    checksum_path, manifest_path = evidence_sidecars(archive)
    output = validate_output_path(
        requested_output,
        (archive, checksum_path, manifest_path, checkpoint_index),
    )
    payload = verify_checkpoint_download(archive, checkpoint_index)
    write_verification(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

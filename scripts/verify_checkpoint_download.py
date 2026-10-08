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
    MAX_EVIDENCE_ARCHIVE_BYTES,
    MAX_EVIDENCE_CHECKSUM_BYTES,
    MAX_EVIDENCE_MANIFEST_BYTES,
    evidence_sidecars,
    verify_evidence_archive,
)
from src.pipeline.durable_json import write_durable_json_exclusive  # noqa: E402
from src.pipeline.evidence_paths import (  # noqa: E402
    is_link_or_junction,
    resolve_regular_file_without_links,
)
from src.pipeline.stable_json import read_stable_json_mapping  # noqa: E402
from src.utils.bounded_file import sha256_stable_regular_file  # noqa: E402


ARCHIVE_PATTERN = re.compile(r"^checkpoint-(\d+)\.tar\.gz$")
MAX_CHECKPOINT_INDEX_BYTES = 100_000_000
MAX_TRAINER_STATE_BYTES = 10_000_000
MAX_VERIFICATION_REPORT_BYTES = 10_000_000
CHECKPOINT_INDEX_KEYS = {"updated_at", "checkpoints"}
CHECKPOINT_RECORD_KEYS = {
    "step",
    "checkpoint",
    "archive",
    "bytes",
    "sha256",
    "manifest",
    "manifest_bytes",
    "manifest_sha256",
    "file_count",
    "content_bytes",
    "eval_loss",
    "status",
}
VERIFICATION_REPORT_KEYS = {
    "schema_version",
    "verified_at",
    "checkpoint_step",
    "archive",
    "sidecar",
    "content_manifest",
    "checkpoint_index",
    "trainer_state",
    "checks",
}


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("JSON document contains a duplicate key")
        payload[key] = value
    return payload


def _reject_json_constant(_value: str) -> object:
    raise ValueError("JSON document contains a non-finite number")


def _strict_json_mapping(payload: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError(f"{label} is not valid strict UTF-8 JSON") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} root must be an object")
    return parsed


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Invalid integer for {label}: {value!r}")
    if value < 0:
        raise ValueError(f"Invalid negative integer for {label}: {value}")
    return value


def _finite_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid number for {label}: {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite number for {label}: {value!r}")
    return parsed


def _timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Invalid timestamp for {label}: {value!r}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"Invalid timestamp for {label}: {value!r}") from None
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp for {label} must include a timezone")
    return value


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Invalid SHA-256 for {label}: {value!r}")
    return value


def _checkpoint_step(archive_path: Path) -> int:
    match = ARCHIVE_PATTERN.fullmatch(archive_path.name)
    if not match:
        raise ValueError(
            "Archive name must be checkpoint-<step>.tar.gz: " f"{archive_path.name}"
        )
    return int(match.group(1))


def _checkpoint_record(index: dict[str, Any], step: int) -> dict[str, Any]:
    if set(index) != CHECKPOINT_INDEX_KEYS:
        raise ValueError("Checkpoint index has an invalid schema")
    _timestamp(index.get("updated_at"), "index.updated_at")
    records = index.get("checkpoints")
    if not isinstance(records, list):
        raise ValueError("Checkpoint index must contain a checkpoints list")
    matches: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    for item in records:
        if not isinstance(item, dict) or set(item) != CHECKPOINT_RECORD_KEYS:
            raise ValueError("Checkpoint index contains an invalid record schema")
        record_step = _integer(item.get("step"), "index.step")
        if record_step < 1:
            raise ValueError("Checkpoint index steps must be positive")
        for field in ("checkpoint", "archive", "manifest"):
            if not isinstance(item.get(field), str) or not item[field]:
                raise ValueError(f"Checkpoint index {field} must be a non-empty string")
        for field in ("bytes", "manifest_bytes", "file_count", "content_bytes"):
            parsed = _integer(item.get(field), f"index.{field}")
            if field != "content_bytes" and parsed < 1:
                raise ValueError(f"Checkpoint index {field} must be positive")
        _digest(item.get("sha256"), "index.sha256")
        _digest(item.get("manifest_sha256"), "index.manifest_sha256")
        _finite_float(item.get("eval_loss"), "index.eval_loss")
        if item.get("status") not in {"archived", "already_verified"}:
            raise ValueError("Checkpoint index record has an invalid status")
        if record_step in seen_steps:
            raise ValueError("Checkpoint index contains duplicate steps")
        seen_steps.add(record_step)
        if record_step == step:
            matches.append(item)
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
        if member.size < 1 or member.size > MAX_TRAINER_STATE_BYTES:
            raise ValueError(
                f"Trainer state size is outside 1..{MAX_TRAINER_STATE_BYTES} bytes: "
                f"{member_name}"
            )
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"Cannot read trainer state member: {member_name}")
        raw_payload = handle.read(MAX_TRAINER_STATE_BYTES + 1)
        if len(raw_payload) != member.size:
            raise ValueError(f"Trainer state member size mismatch: {member_name}")
        payload = _strict_json_mapping(raw_payload, "Trainer state")
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
    archive_path = resolve_regular_file_without_links(
        archive_path,
        label="Checkpoint archive",
        maximum_bytes=MAX_EVIDENCE_ARCHIVE_BYTES,
    )
    checkpoint_index_path = resolve_regular_file_without_links(
        checkpoint_index_path,
        label="Checkpoint archive index",
        maximum_bytes=MAX_CHECKPOINT_INDEX_BYTES,
    )
    checksum_path, manifest_path = evidence_sidecars(archive_path)

    step = _checkpoint_step(archive_path)
    manifest = verify_evidence_archive(archive_path)
    checksum_path = resolve_regular_file_without_links(
        checksum_path,
        label="Checkpoint checksum sidecar",
        maximum_bytes=MAX_EVIDENCE_CHECKSUM_BYTES,
    )
    manifest_path = resolve_regular_file_without_links(
        manifest_path,
        label="Checkpoint content manifest",
        maximum_bytes=MAX_EVIDENCE_MANIFEST_BYTES,
    )
    archive_digest = str(manifest["archive_sha256"])
    archive_bytes = int(manifest["archive_bytes"])
    checksum_digest, checksum_bytes = sha256_stable_regular_file(
        checksum_path,
        maximum_bytes=MAX_EVIDENCE_CHECKSUM_BYTES,
        label="Checkpoint checksum sidecar",
    )
    manifest_digest, manifest_bytes = sha256_stable_regular_file(
        manifest_path,
        maximum_bytes=MAX_EVIDENCE_MANIFEST_BYTES,
        label="Checkpoint content manifest",
    )
    index_document = read_stable_json_mapping(
        checkpoint_index_path,
        maximum_bytes=MAX_CHECKPOINT_INDEX_BYTES,
        label="Checkpoint archive index",
    )
    index = index_document.mapping
    record = _checkpoint_record(index, step)

    expected_status = record.get("status")
    if expected_status not in {"archived", "already_verified"}:
        raise ValueError(f"Checkpoint index record is not archived: {expected_status!r}")
    archive_record_path = record.get("archive")
    manifest_record_path = record.get("manifest")
    checkpoint_record_path = record.get("checkpoint")
    if (
        not isinstance(archive_record_path, str)
        or Path(archive_record_path).name != archive_path.name
    ):
        raise ValueError("Checkpoint index archive name mismatch")
    if (
        _integer(record.get("bytes"), "index.bytes") != archive_bytes
        or _digest(record.get("sha256"), "index.sha256") != archive_digest
    ):
        raise ValueError("Checkpoint index archive identity mismatch")
    if (
        not isinstance(manifest_record_path, str)
        or Path(manifest_record_path).name != manifest_path.name
    ):
        raise ValueError("Checkpoint index manifest name mismatch")
    if (
        _integer(record.get("manifest_bytes"), "index.manifest_bytes")
        != manifest_bytes
        or _digest(record.get("manifest_sha256"), "index.manifest_sha256")
        != manifest_digest
    ):
        raise ValueError("Checkpoint index manifest identity mismatch")
    if (
        not isinstance(checkpoint_record_path, str)
        or Path(checkpoint_record_path).name != f"checkpoint-{step}"
    ):
        raise ValueError("Checkpoint index checkpoint name mismatch")
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
    if global_step < 1 or max_steps < 1:
        raise ValueError("Trainer global_step and max_steps must be positive")
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
    if best_model_checkpoint is not None and not isinstance(best_model_checkpoint, str):
        raise ValueError("trainer_state.best_model_checkpoint must be a string or null")
    final_archive_digest, final_archive_bytes = sha256_stable_regular_file(
        archive_path,
        maximum_bytes=MAX_EVIDENCE_ARCHIVE_BYTES,
        label="Checkpoint archive",
    )
    final_checksum_digest, final_checksum_bytes = sha256_stable_regular_file(
        checksum_path,
        maximum_bytes=MAX_EVIDENCE_CHECKSUM_BYTES,
        label="Checkpoint checksum sidecar",
    )
    final_manifest_digest, final_manifest_bytes = sha256_stable_regular_file(
        manifest_path,
        maximum_bytes=MAX_EVIDENCE_MANIFEST_BYTES,
        label="Checkpoint content manifest",
    )
    final_index = read_stable_json_mapping(
        checkpoint_index_path,
        maximum_bytes=MAX_CHECKPOINT_INDEX_BYTES,
        label="Checkpoint archive index",
        expected_sha256=index_document.sha256,
    )
    if (
        (final_archive_digest, final_archive_bytes) != (archive_digest, archive_bytes)
        or (final_checksum_digest, final_checksum_bytes)
        != (checksum_digest, checksum_bytes)
        or (final_manifest_digest, final_manifest_bytes)
        != (manifest_digest, manifest_bytes)
        or final_index.bytes != index_document.bytes
    ):
        raise RuntimeError("Checkpoint evidence changed while verifying")
    return {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_step": step,
        "archive": {
            "name": archive_path.name,
            "bytes": archive_bytes,
            "sha256": archive_digest,
        },
        "sidecar": {
            "name": checksum_path.name,
            "bytes": checksum_bytes,
            "sha256": checksum_digest,
        },
        "content_manifest": {
            "name": manifest_path.name,
            "bytes": manifest_bytes,
            "sha256": manifest_digest,
            "file_count": int(manifest["file_count"]),
            "content_bytes": int(manifest["content_bytes"]),
        },
        "checkpoint_index": {
            "name": checkpoint_index_path.name,
            "bytes": index_document.bytes,
            "sha256": index_document.sha256,
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
    if not isinstance(payload, dict) or set(payload) != VERIFICATION_REPORT_KEYS:
        raise ValueError("Checkpoint verification payload has an invalid schema")
    if payload.get("schema_version") != 1:
        raise ValueError("Checkpoint verification schema_version must be 1")
    if _integer(payload.get("checkpoint_step"), "verification.checkpoint_step") < 1:
        raise ValueError("Checkpoint verification step must be positive")
    _timestamp(payload.get("verified_at"), "verification.verified_at")
    comparable = {key: value for key, value in payload.items() if key != "verified_at"}

    def existing_matches() -> bool:
        existing = read_stable_json_mapping(
            path,
            maximum_bytes=MAX_VERIFICATION_REPORT_BYTES,
            label="Checkpoint verification report",
        ).mapping
        if set(existing) != set(payload):
            return False
        _timestamp(existing.get("verified_at"), "verification.verified_at")
        return (
            {key: value for key, value in existing.items() if key != "verified_at"}
            == comparable
        )

    if path.exists() or is_link_or_junction(path):
        if existing_matches():
            return
        raise FileExistsError(
            f"Checkpoint verification report already exists with different evidence: {path}"
        )
    try:
        write_durable_json_exclusive(
            path,
            payload,
            maximum_bytes=MAX_VERIFICATION_REPORT_BYTES,
            label="Checkpoint verification report",
        )
    except FileExistsError:
        if not existing_matches():
            raise FileExistsError(
                "Checkpoint verification report was concurrently published with "
                f"different evidence: {path}"
            ) from None


def validate_output_path(path: Path, reserved_paths: tuple[Path, ...]) -> Path:
    output = Path(os.path.abspath(path))
    reserved = {Path(os.path.abspath(item)) for item in reserved_paths}
    resolved_reserved = {item.resolve() for item in reserved_paths}
    if output in reserved or output.resolve() in resolved_reserved:
        raise ValueError(f"Verification output cannot overwrite evidence: {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint-index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    archive = Path(os.path.abspath(args.archive))
    step = _checkpoint_step(archive)
    checkpoint_index = (
        Path(os.path.abspath(args.checkpoint_index))
        if args.checkpoint_index is not None
        else archive.parent / "checkpoint_archives.json"
    )
    requested_output = (
        Path(os.path.abspath(args.output))
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

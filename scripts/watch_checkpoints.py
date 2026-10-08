"""Archive completed Trainer checkpoints atomically while a GPU job is running."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.candidate_evidence import (  # noqa: E402
    MAX_EVIDENCE_ARCHIVE_BYTES,
    MAX_EVIDENCE_CONTENT_BYTES,
    MAX_EVIDENCE_MEMBERS,
    archive_evidence,
    evidence_sidecars,
    sha256,
    verify_evidence_archive,
)
from src.pipeline.durable_json import write_durable_json  # noqa: E402
from src.pipeline.evidence_paths import is_link_or_junction  # noqa: E402
from src.pipeline.stable_json import read_stable_json_mapping  # noqa: E402
from src.utils.bounded_file import sha256_stable_regular_file  # noqa: E402


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")
REQUIRED_TRAINER_FILES = {
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
}
MODEL_FILES = {"adapter_model.safetensors", "model.safetensors", "pytorch_model.bin"}
MAX_TRAINER_STATE_BYTES = 10_000_000
MAX_CHECKPOINT_INDEX_BYTES = 100_000_000


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def completed_checkpoint(path: Path) -> tuple[int, dict[str, Any]] | None:
    step = checkpoint_step(path)
    if step is None or is_link_or_junction(path) or not path.is_dir():
        return None
    entries = list(path.iterdir())
    if any(is_link_or_junction(item) for item in entries):
        return None
    names = {item.name for item in entries if item.is_file()}
    if not REQUIRED_TRAINER_FILES <= names or not MODEL_FILES & names:
        return None
    try:
        trainer_state = read_stable_json_mapping(
            path / "trainer_state.json",
            maximum_bytes=MAX_TRAINER_STATE_BYTES,
            label="Trainer checkpoint state",
        ).mapping
    except (FileNotFoundError, RuntimeError, ValueError):
        return None
    global_step = trainer_state.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        return None
    if global_step != step:
        return None
    return step, trainer_state


def verified_existing_archive(archive_path: Path, checksum_path: Path) -> bool:
    expected_checksum, _ = evidence_sidecars(archive_path)
    if Path(os.path.abspath(checksum_path)) != Path(os.path.abspath(expected_checksum)):
        return False
    try:
        verify_evidence_archive(archive_path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return False
    return True


def checkpoint_file_records(checkpoint: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    content_bytes = 0
    for path in sorted(checkpoint.rglob("*")):
        if is_link_or_junction(path):
            raise ValueError(f"Checkpoint cannot contain symlinks: {path}")
        if not path.is_file():
            continue
        digest, size = sha256_stable_regular_file(
            path,
            maximum_bytes=MAX_EVIDENCE_ARCHIVE_BYTES,
            label="Checkpoint file",
        )
        records.append(
            {
                "path": (Path(checkpoint.name) / path.relative_to(checkpoint)).as_posix(),
                "bytes": size,
                "sha256": digest,
                "source": path,
            }
        )
        content_bytes += size
        if len(records) > MAX_EVIDENCE_MEMBERS:
            raise ValueError("Checkpoint has too many files")
        if content_bytes > MAX_EVIDENCE_CONTENT_BYTES:
            raise ValueError("Checkpoint exceeds the content byte limit")
    if not records:
        raise ValueError(f"Checkpoint is empty: {checkpoint}")
    return records


def archive_checkpoint(checkpoint: Path, archive_dir: Path) -> dict[str, Any]:
    complete = completed_checkpoint(checkpoint)
    if complete is None:
        raise ValueError(f"Checkpoint is incomplete: {checkpoint}")
    step, trainer_state = complete
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{checkpoint.name}.tar.gz"
    preexisting_archive = archive_path.exists() or is_link_or_junction(archive_path)
    archive_path, digest = archive_evidence(checkpoint, archive_path)
    _, manifest_path = evidence_sidecars(archive_path)
    manifest = verify_evidence_archive(archive_path)
    archive_digest, archive_bytes = sha256_stable_regular_file(
        archive_path,
        maximum_bytes=MAX_EVIDENCE_ARCHIVE_BYTES,
        label="Checkpoint archive",
    )
    if archive_digest != digest:
        raise RuntimeError("Checkpoint archive changed after publication")
    manifest_digest, manifest_bytes = sha256_stable_regular_file(
        manifest_path,
        maximum_bytes=MAX_CHECKPOINT_INDEX_BYTES,
        label="Checkpoint content manifest",
    )
    eval_loss = next(
        (
            float(item["eval_loss"])
            for item in reversed(trainer_state.get("log_history", []))
            if "eval_loss" in item
        ),
        None,
    )
    return {
        "step": step,
        "checkpoint": str(checkpoint.resolve()),
        "archive": str(archive_path),
        "bytes": archive_bytes,
        "sha256": digest,
        "manifest": str(manifest_path.resolve(strict=True)),
        "manifest_bytes": manifest_bytes,
        "manifest_sha256": manifest_digest,
        "file_count": int(manifest["file_count"]),
        "content_bytes": int(manifest["content_bytes"]),
        "eval_loss": eval_loss,
        "status": "already_verified" if preexisting_archive else "archived",
    }


def write_manifest(archive_dir: Path, records: list[dict[str, Any]]) -> None:
    manifest_path = archive_dir / "checkpoint_archives.json"
    existing: dict[int, dict[str, Any]] = {}
    if manifest_path.is_file():
        payload = read_stable_json_mapping(
            manifest_path,
            maximum_bytes=MAX_CHECKPOINT_INDEX_BYTES,
            label="Checkpoint archive index",
        ).mapping
        checkpoints = payload.get("checkpoints")
        if set(payload) != {"updated_at", "checkpoints"} or not isinstance(
            checkpoints,
            list,
        ):
            raise ValueError("Checkpoint archive index has an invalid schema")
        for item in checkpoints:
            if not isinstance(item, dict):
                raise ValueError("Checkpoint archive index has an invalid record")
            step = item.get("step")
            if isinstance(step, bool) or not isinstance(step, int) or step < 1:
                raise ValueError("Checkpoint archive index has an invalid step")
            if step in existing:
                raise ValueError("Checkpoint archive index has duplicate steps")
            existing[step] = item
    for record in records:
        existing[int(record["step"])] = record
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoints": [existing[step] for step in sorted(existing)],
    }
    write_durable_json(
        manifest_path,
        payload,
        maximum_bytes=MAX_CHECKPOINT_INDEX_BYTES,
        label="Checkpoint archive index",
    )


def archive_ready_checkpoints(
    model_dir: Path, archive_dir: Path, skip_steps: set[int] | None = None
) -> list[dict[str, Any]]:
    skipped = skip_steps or set()
    checkpoints = sorted(
        (
            path
            for path in model_dir.glob("checkpoint-*")
            if checkpoint_step(path) is not None and checkpoint_step(path) not in skipped
        ),
        key=lambda path: checkpoint_step(path) or -1,
    )
    records = []
    for checkpoint in checkpoints:
        if completed_checkpoint(checkpoint) is None:
            continue
        records.append(archive_checkpoint(checkpoint, archive_dir))
    if records:
        write_manifest(archive_dir, records)
    return records


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        parser.error("--model-dir must exist")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")

    seen_archived: set[int] = set()
    while True:
        records = archive_ready_checkpoints(args.model_dir, args.archive_dir, seen_archived)
        for record in records:
            step = int(record["step"])
            if record["status"] == "archived" and step not in seen_archived:
                print(json.dumps(record, ensure_ascii=False), flush=True)
            seen_archived.add(step)
        if args.once:
            break
        if args.watch_pid is not None and not pid_is_alive(args.watch_pid):
            archive_ready_checkpoints(args.model_dir, args.archive_dir, seen_archived)
            break
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

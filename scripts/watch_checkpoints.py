"""Archive completed Trainer checkpoints atomically while a GPU job is running."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.candidate_evidence import evidence_sidecars, verify_evidence_archive  # noqa: E402


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")
REQUIRED_TRAINER_FILES = {
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
}
MODEL_FILES = {"adapter_model.safetensors", "model.safetensors", "pytorch_model.bin"}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def completed_checkpoint(path: Path) -> tuple[int, dict[str, Any]] | None:
    step = checkpoint_step(path)
    if step is None or not path.is_dir():
        return None
    names = {item.name for item in path.iterdir() if item.is_file()}
    if not REQUIRED_TRAINER_FILES <= names or not MODEL_FILES & names:
        return None
    try:
        trainer_state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if int(trainer_state.get("global_step", -1)) != step:
        return None
    return step, trainer_state


def verified_existing_archive(archive_path: Path, checksum_path: Path) -> bool:
    if not archive_path.is_file() or not checksum_path.is_file():
        return False
    fields = checksum_path.read_text(encoding="utf-8").split()
    return len(fields) >= 2 and fields[0].lower() == sha256(archive_path)


def checkpoint_file_records(checkpoint: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Checkpoint cannot contain symlinks: {path}")
        if not path.is_file():
            continue
        records.append(
            {
                "path": (Path(checkpoint.name) / path.relative_to(checkpoint)).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "source": path,
            }
        )
    if not records:
        raise ValueError(f"Checkpoint is empty: {checkpoint}")
    return records


def write_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def ensure_content_manifest(
    archive_path: Path,
    records: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    _, manifest_path = evidence_sidecars(archive_path)
    if manifest_path.is_file():
        return manifest_path, verify_evidence_archive(archive_path)
    public_records = [
        {key: record[key] for key in ("path", "bytes", "sha256")} for record in records
    ]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "file_count": len(public_records),
        "content_bytes": sum(int(record["bytes"]) for record in public_records),
        "files": public_records,
    }
    write_atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest_path, verify_evidence_archive(archive_path)


def archive_checkpoint(checkpoint: Path, archive_dir: Path) -> dict[str, Any]:
    complete = completed_checkpoint(checkpoint)
    if complete is None:
        raise ValueError(f"Checkpoint is incomplete: {checkpoint}")
    step, trainer_state = complete
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{checkpoint.name}.tar.gz"
    checksum_path = archive_dir / f"{checkpoint.name}.tar.gz.sha256"
    records = checkpoint_file_records(checkpoint)
    if verified_existing_archive(archive_path, checksum_path):
        manifest_path, manifest = ensure_content_manifest(archive_path, records)
        return {
            "step": step,
            "checkpoint": str(checkpoint.resolve()),
            "archive": str(archive_path.resolve()),
            "bytes": archive_path.stat().st_size,
            "sha256": checksum_path.read_text(encoding="utf-8").split()[0].lower(),
            "manifest": str(manifest_path.resolve()),
            "manifest_bytes": manifest_path.stat().st_size,
            "manifest_sha256": sha256(manifest_path),
            "file_count": int(manifest["file_count"]),
            "content_bytes": int(manifest["content_bytes"]),
            "eval_loss": next(
                (
                    float(item["eval_loss"])
                    for item in reversed(trainer_state.get("log_history", []))
                    if "eval_loss" in item
                ),
                None,
            ),
            "status": "already_verified",
        }

    temporary = archive_dir / f".{checkpoint.name}.tar.gz.part"
    temporary.unlink(missing_ok=True)
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for record in records:
                archive.add(record["source"], arcname=record["path"], recursive=False)
        os.replace(temporary, archive_path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256(archive_path)
    write_atomic(checksum_path, f"{digest}  {archive_path.name}\n")
    manifest_path, manifest = ensure_content_manifest(archive_path, records)
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
        "archive": str(archive_path.resolve()),
        "bytes": archive_path.stat().st_size,
        "sha256": digest,
        "manifest": str(manifest_path.resolve()),
        "manifest_bytes": manifest_path.stat().st_size,
        "manifest_sha256": sha256(manifest_path),
        "file_count": int(manifest["file_count"]),
        "content_bytes": int(manifest["content_bytes"]),
        "eval_loss": eval_loss,
        "status": "archived",
    }


def write_manifest(archive_dir: Path, records: list[dict[str, Any]]) -> None:
    manifest_path = archive_dir / "checkpoint_archives.json"
    existing: dict[int, dict[str, Any]] = {}
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing = {int(item["step"]): item for item in payload.get("checkpoints", [])}
        except (OSError, ValueError, json.JSONDecodeError):
            existing = {}
    for record in records:
        existing[int(record["step"])] = record
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoints": [existing[step] for step in sorted(existing)],
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, manifest_path)


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

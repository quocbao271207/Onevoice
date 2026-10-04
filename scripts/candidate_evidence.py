"""Shared helpers for immutable candidate-evaluation evidence bundles."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def archive_evidence(output_dir: Path) -> tuple[Path, str]:
    """Archive a completed evidence directory and write a SHA-256 sidecar."""
    archive_path = output_dir.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(output_dir.parent), recursive=False)
    digest = sha256(archive_path)
    archive_path.with_suffix(archive_path.suffix + ".sha256").write_text(
        f"{digest}  {archive_path.name}\n", encoding="utf-8"
    )
    return archive_path, digest

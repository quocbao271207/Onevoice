"""Crash-durable persistence for bounded pipeline JSON evidence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .evidence_paths import is_link_or_junction
from ..utils.bounded_file import read_stable_regular_file


def _checked_destination(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    for ancestor in reversed((absolute.parent, *absolute.parent.parents)):
        if is_link_or_junction(ancestor):
            raise ValueError(f"{label} destination cannot traverse a link: {ancestor}")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    for ancestor in reversed((absolute.parent, *absolute.parent.parents)):
        if is_link_or_junction(ancestor):
            raise ValueError(f"{label} destination cannot traverse a link: {ancestor}")
    if is_link_or_junction(absolute):
        raise ValueError(f"{label} destination cannot be a link: {absolute}")
    if absolute.exists() and not absolute.is_file():
        raise ValueError(f"{label} destination must be a regular file: {absolute}")
    return absolute


def _sync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_durable_json(
    path: Path,
    payload: dict[str, Any],
    *,
    maximum_bytes: int,
    label: str,
) -> None:
    """Durably replace one strict, bounded JSON mapping through an owned temp."""
    if (
        isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or maximum_bytes < 1
    ):
        raise ValueError("maximum_bytes must be a positive integer")
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")
    if not isinstance(payload, dict):
        raise ValueError(f"{label} payload must be a mapping")
    try:
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise ValueError(f"{label} payload is not serializable") from None
    if len(encoded) > maximum_bytes:
        raise ValueError(f"{label} payload exceeds {maximum_bytes} bytes")

    destination = _checked_destination(path, label=label)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "posix":
            os.chmod(temporary, 0o600)
        staged = read_stable_regular_file(
            temporary,
            maximum_bytes=maximum_bytes,
            label=f"Staged {label}",
        )
        if staged != encoded:
            raise RuntimeError(f"Staged {label} does not match its payload")
        _checked_destination(destination, label=label)
        os.replace(temporary, destination)
        _sync_directory(destination.parent)
        persisted = read_stable_regular_file(
            destination,
            maximum_bytes=maximum_bytes,
            label=f"Persisted {label}",
        )
        if persisted != encoded:
            raise RuntimeError(f"Persisted {label} does not match its payload")
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        temporary.unlink(missing_ok=True)

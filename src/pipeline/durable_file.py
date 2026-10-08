"""Crash-durable publication for immutable bounded files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .evidence_paths import (
    prepare_new_file_destination_without_links,
    resolve_regular_file_without_links,
)
from ..utils.bounded_file import (
    read_stable_regular_file,
    sha256_stable_regular_file,
)


def _validate_arguments(*, maximum_bytes: int, label: str) -> None:
    if (
        isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or maximum_bytes < 1
    ):
        raise ValueError("maximum_bytes must be a positive integer")
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")


def _sync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_durable_bytes_exclusive(
    path: Path,
    payload: bytes,
    *,
    maximum_bytes: int,
    label: str,
) -> None:
    """Durably create one immutable byte payload without overwriting."""
    _validate_arguments(maximum_bytes=maximum_bytes, label=label)
    if not isinstance(payload, bytes) or not payload:
        raise ValueError(f"{label} payload must be non-empty bytes")
    if len(payload) > maximum_bytes:
        raise ValueError(f"{label} payload exceeds {maximum_bytes} bytes")
    destination = prepare_new_file_destination_without_links(path, label=label)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    published = False
    try:
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "posix":
            os.chmod(temporary, 0o600)
        if read_stable_regular_file(
            temporary,
            maximum_bytes=maximum_bytes,
            label=f"Staged {label}",
        ) != payload:
            raise RuntimeError(f"Staged {label} does not match its payload")
        destination = prepare_new_file_destination_without_links(
            destination,
            label=label,
        )
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(f"{label} already exists: {destination}") from None
        published = True
        _sync_directory(destination.parent)
        if read_stable_regular_file(
            destination,
            maximum_bytes=maximum_bytes,
            label=f"Persisted {label}",
        ) != payload:
            raise RuntimeError(f"Persisted {label} does not match its payload")
    except Exception:
        if published:
            destination.unlink(missing_ok=True)
            _sync_directory(destination.parent)
        raise
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        temporary.unlink(missing_ok=True)


def publish_durable_file_exclusive(
    staged_path: Path,
    destination_path: Path,
    *,
    maximum_bytes: int,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[str, int]:
    """Publish a verified staged file through an exclusive same-volume link."""
    _validate_arguments(maximum_bytes=maximum_bytes, label=label)
    staged_path = resolve_regular_file_without_links(
        staged_path,
        label=f"Staged {label}",
        maximum_bytes=maximum_bytes,
    )
    digest, size = sha256_stable_regular_file(
        staged_path,
        maximum_bytes=maximum_bytes,
        label=f"Staged {label}",
    )
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Staged {label} checksum does not match")
    destination = prepare_new_file_destination_without_links(
        destination_path,
        label=label,
    )
    published = False
    try:
        try:
            os.link(staged_path, destination)
        except FileExistsError:
            raise FileExistsError(f"{label} already exists: {destination}") from None
        published = True
        _sync_directory(destination.parent)
        persisted_digest, persisted_size = sha256_stable_regular_file(
            destination,
            maximum_bytes=maximum_bytes,
            label=f"Persisted {label}",
        )
        if persisted_digest != digest or persisted_size != size:
            raise RuntimeError(f"Persisted {label} does not match staged file")
        return digest, size
    except Exception:
        if published:
            destination.unlink(missing_ok=True)
            _sync_directory(destination.parent)
        raise

"""Read small runtime configuration files through one fail-closed boundary."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def _is_link_or_reparse(file_stat: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(file_stat, "st_file_attributes", 0)
    return stat.S_ISLNK(file_stat.st_mode) or bool(
        reparse_flag and attributes & reparse_flag
    )


def _file_identity(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        getattr(file_stat, "st_file_attributes", 0),
    )


def read_stable_regular_file(
    path: str | Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes:
    """Read one unchanged regular file without following a final symlink."""
    if (
        isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or maximum_bytes < 1
    ):
        raise ValueError("maximum_bytes must be a positive integer")
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")
    candidate = Path(path)
    try:
        before = candidate.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} does not exist: {candidate}")
    except OSError:
        raise ValueError(f"Failed to inspect {label.lower()}") from None
    if _is_link_or_reparse(before):
        raise ValueError(f"{label} must not be a symlink or junction")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if before.st_size < 1 or before.st_size > maximum_bytes:
        raise ValueError(f"{label} size is outside 1..{maximum_bytes} bytes")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor_number: int | None = None
    try:
        descriptor_number = os.open(candidate, flags)
        handle = os.fdopen(descriptor_number, "rb")
        descriptor_number = None
        with handle:
            descriptor_before = os.fstat(handle.fileno())
            if not stat.S_ISREG(descriptor_before.st_mode):
                raise ValueError(f"{label} must be a regular file")
            payload = handle.read(maximum_bytes + 1)
            descriptor_after_first_read = os.fstat(handle.fileno())
            handle.seek(0)
            verification_payload = handle.read(maximum_bytes + 1)
            descriptor_after_second_read = os.fstat(handle.fileno())
        after = candidate.lstat()
    except ValueError:
        raise
    except OSError:
        raise ValueError(f"Failed to read {label.lower()}") from None
    finally:
        if descriptor_number is not None:
            try:
                os.close(descriptor_number)
            except OSError:
                pass

    if (
        len(payload) != before.st_size
        or len(payload) > maximum_bytes
        or payload != verification_payload
        or _is_link_or_reparse(after)
        or _file_identity(before) != _file_identity(descriptor_before)
        or _file_identity(before) != _file_identity(descriptor_after_first_read)
        or _file_identity(before) != _file_identity(descriptor_after_second_read)
        or _file_identity(before) != _file_identity(after)
    ):
        raise RuntimeError(f"{label} changed while reading")
    return payload

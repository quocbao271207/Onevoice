"""Strict filesystem boundaries for immutable pipeline evidence."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any


def is_link_or_junction(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(
        reparse_flag
        and getattr(metadata, "st_file_attributes", 0) & reparse_flag
    )


def resolve_regular_file(
    value: Any,
    *,
    label: str,
    maximum_bytes: int | None = None,
) -> Path:
    """Resolve one regular file without following a final link or junction."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{label} path is missing")
    path = Path(os.path.abspath(Path(raw)))
    if is_link_or_junction(path):
        raise ValueError(f"{label} cannot be a symlink or junction: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing or not a regular file: {path}")
    if maximum_bytes is not None and path.stat().st_size > maximum_bytes:
        raise ValueError(f"{label} exceeds {maximum_bytes} bytes")
    return path.resolve(strict=True)


def resolve_regular_file_without_links(
    value: Any,
    *,
    label: str,
    maximum_bytes: int | None = None,
) -> Path:
    """Resolve an arbitrary regular file without link traversal in any component."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{label} path is missing")
    path = Path(os.path.abspath(Path(raw)))
    current = Path(path.parts[0])
    for part in path.parts[1:]:
        current /= part
        if is_link_or_junction(current):
            raise ValueError(
                f"{label} cannot traverse a symlink or junction: {current}"
            )
    return resolve_regular_file(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
    )


def resolve_regular_directory_without_links(
    value: Any,
    *,
    label: str,
) -> Path:
    """Resolve an arbitrary directory without link traversal in any component."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{label} path is missing")
    path = Path(os.path.abspath(Path(raw)))
    current = Path(path.parts[0])
    for part in path.parts[1:]:
        current /= part
        if is_link_or_junction(current):
            raise ValueError(
                f"{label} cannot traverse a symlink or junction: {current}"
            )
    if not path.is_dir():
        raise FileNotFoundError(f"{label} is missing or not a directory: {path}")
    return path.resolve(strict=True)


def prepare_new_file_destination_without_links(
    value: Any,
    *,
    label: str,
) -> Path:
    """Create destination parents and reject existing/link-traversing targets."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{label} path is missing")
    path = Path(os.path.abspath(Path(raw)))
    path.parent.mkdir(parents=True, exist_ok=True)
    current = Path(path.parts[0])
    for part in path.parts[1:]:
        current /= part
        if is_link_or_junction(current):
            raise ValueError(
                f"{label} cannot traverse a symlink or junction: {current}"
            )
    if path.exists():
        raise FileExistsError(f"{label} already exists: {path}")
    return path


def resolve_regular_file_under(
    value: Any,
    *,
    project_root: Path,
    allowed_root: Path,
    label: str,
    maximum_bytes: int | None = None,
) -> Path:
    """Resolve one regular file without permitting link traversal or path escape."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{label} path is missing")
    project_boundary = Path(os.path.abspath(project_root))
    root = Path(os.path.abspath(allowed_root))
    path = Path(raw)
    if not path.is_absolute():
        path = project_boundary / path
    path = Path(os.path.abspath(path))
    try:
        root_relative = root.relative_to(project_boundary)
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain under {root}") from exc
    if is_link_or_junction(project_boundary):
        raise ValueError(
            f"{label} cannot traverse a symlink or junction: {project_boundary}"
        )
    current = project_boundary
    for part in (*root_relative.parts, *relative.parts):
        current /= part
        if is_link_or_junction(current):
            raise ValueError(f"{label} cannot traverse a symlink or junction: {current}")
    resolved = resolve_regular_file(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
    )
    resolved_root = root.resolve(strict=True)
    if resolved_root not in resolved.parents:
        raise ValueError(f"{label} must remain under {resolved_root}")
    return resolved


def resolve_regular_directory_under(
    value: Any,
    *,
    project_root: Path,
    allowed_root: Path,
    label: str,
) -> Path:
    """Resolve one directory without permitting link traversal or path escape."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{label} path is missing")
    project_boundary = Path(os.path.abspath(project_root))
    root = Path(os.path.abspath(allowed_root))
    path = Path(raw)
    if not path.is_absolute():
        path = project_boundary / path
    path = Path(os.path.abspath(path))
    try:
        root_relative = root.relative_to(project_boundary)
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain under {root}") from exc
    if is_link_or_junction(project_boundary):
        raise ValueError(
            f"{label} cannot traverse a symlink or junction: {project_boundary}"
        )
    current = project_boundary
    for part in (*root_relative.parts, *relative.parts):
        current /= part
        if is_link_or_junction(current):
            raise ValueError(f"{label} cannot traverse a symlink or junction: {current}")
    if not path.is_dir():
        raise FileNotFoundError(f"{label} is missing or not a directory: {path}")
    resolved = path.resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if resolved_root not in resolved.parents:
        raise ValueError(f"{label} must remain under {resolved_root}")
    return resolved

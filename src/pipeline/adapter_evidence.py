"""Stable, bounded evidence identities for PEFT adapter directory trees."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable

from .evidence_paths import (
    is_link_or_junction,
    resolve_regular_directory_without_links,
)
from ..utils.bounded_file import sha256_stable_regular_file


MAX_ADAPTER_FILES = 10_000
MAX_ADAPTER_DIRECTORIES = 10_000
MAX_ADAPTER_ENTRIES = 20_000
MAX_ADAPTER_FILE_BYTES = 8_000_000_000
MAX_ADAPTER_TREE_BYTES = 16_000_000_000

StableHash = Callable[..., tuple[str, int]]


def _filesystem_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        getattr(metadata, "st_file_attributes", 0),
    )


def _positive_limit(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_adapter_root(adapter: Path, *, label: str) -> tuple[Path, Path]:
    lexical = Path(os.path.abspath(adapter))
    try:
        before = lexical.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} is missing: {lexical}") from None
    except OSError:
        raise ValueError(f"Failed to inspect {label.lower()}") from None
    if is_link_or_junction(lexical):
        raise ValueError(f"{label} cannot be a symlink or junction: {lexical}")
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"{label} must be a directory: {lexical}")
    resolved = resolve_regular_directory_without_links(lexical, label=label)
    try:
        lexical_after = lexical.lstat()
        resolved_metadata = resolved.lstat()
    except OSError:
        raise RuntimeError(f"{label} root changed while resolving") from None
    if (
        is_link_or_junction(lexical)
        or _filesystem_identity(before) != _filesystem_identity(lexical_after)
        or _filesystem_identity(before) != _filesystem_identity(resolved_metadata)
    ):
        raise RuntimeError(f"{label} root changed while resolving")
    return lexical, resolved


def _enumerate_adapter_tree(
    root: Path,
    *,
    label: str,
    maximum_files: int,
    maximum_directories: int,
    maximum_entries: int,
) -> tuple[list[Path], list[str]]:
    pending = [root]
    files: list[Path] = []
    directories: list[str] = []
    directory_count = 0
    entry_count = 0
    while pending:
        directory = pending.pop()
        if is_link_or_junction(directory):
            raise ValueError(
                f"{label} cannot contain symlinks or junctions: {directory}"
            )
        directory_count += 1
        if directory_count > maximum_directories:
            raise ValueError(f"{label} exceeds {maximum_directories} directories")
        directories.append(
            "." if directory == root else directory.relative_to(root).as_posix()
        )
        try:
            before = directory.lstat()
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError(f"{label} entry must be a directory: {directory}")
            children = []
            for child in directory.iterdir():
                entry_count += 1
                if entry_count > maximum_entries:
                    raise ValueError(f"{label} exceeds {maximum_entries} entries")
                children.append(child)
            children.sort(key=lambda child: child.name)
            after = directory.lstat()
        except ValueError:
            raise
        except OSError:
            raise ValueError(f"Failed to enumerate {label.lower()}: {directory}") from None
        if _filesystem_identity(before) != _filesystem_identity(after):
            raise RuntimeError(f"{label} directory changed while reading: {directory}")
        for child in children:
            if is_link_or_junction(child):
                raise ValueError(
                    f"{label} cannot contain symlinks or junctions: {child}"
                )
            try:
                metadata = child.lstat()
            except OSError:
                raise RuntimeError(
                    f"{label} entry changed while reading: {child}"
                ) from None
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(child)
            elif stat.S_ISREG(metadata.st_mode):
                files.append(child)
                if len(files) > maximum_files:
                    raise ValueError(f"{label} exceeds {maximum_files} files")
            else:
                raise ValueError(f"{label} contains a special file: {child}")
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    directories.sort()
    return files, directories


def stable_adapter_tree_manifest(
    adapter: Path,
    *,
    label: str = "Adapter",
    required_relative_paths: tuple[str, ...] = ("adapter_config.json",),
    maximum_files: int = MAX_ADAPTER_FILES,
    maximum_directories: int = MAX_ADAPTER_DIRECTORIES,
    maximum_entries: int = MAX_ADAPTER_ENTRIES,
    maximum_file_bytes: int = MAX_ADAPTER_FILE_BYTES,
    maximum_tree_bytes: int = MAX_ADAPTER_TREE_BYTES,
    hash_file: StableHash = sha256_stable_regular_file,
) -> dict[str, Any]:
    """Return one identity after repeated tree scans and two content passes."""
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")
    for value, name in (
        (maximum_files, "maximum_files"),
        (maximum_directories, "maximum_directories"),
        (maximum_entries, "maximum_entries"),
        (maximum_file_bytes, "maximum_file_bytes"),
        (maximum_tree_bytes, "maximum_tree_bytes"),
    ):
        _positive_limit(value, name)
    if (
        not isinstance(required_relative_paths, tuple)
        or any(not isinstance(path, str) or not path for path in required_relative_paths)
        or len(required_relative_paths) != len(set(required_relative_paths))
    ):
        raise ValueError("required_relative_paths must contain unique non-empty strings")

    lexical, root = _resolve_adapter_root(adapter, label=label)
    paths, directories = _enumerate_adapter_tree(
        root,
        label=label,
        maximum_files=maximum_files,
        maximum_directories=maximum_directories,
        maximum_entries=maximum_entries,
    )
    relative_paths = [path.relative_to(root).as_posix() for path in paths]
    missing = sorted(set(required_relative_paths) - set(relative_paths))
    if missing:
        raise FileNotFoundError(f"{label} is incomplete; missing: {missing}")
    if not paths:
        raise ValueError(f"{label} is empty: {root}")

    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path, relative in zip(paths, relative_paths, strict=True):
        digest, size = hash_file(
            path,
            maximum_bytes=maximum_file_bytes,
            label=f"{label} file {relative}",
        )
        total_bytes += size
        if total_bytes > maximum_tree_bytes:
            raise ValueError(f"{label} exceeds {maximum_tree_bytes} total bytes")
        files.append({"path": relative, "bytes": size, "sha256": digest})

    verified_lexical, verified_root = _resolve_adapter_root(lexical, label=label)
    if verified_lexical != lexical or verified_root != root:
        raise RuntimeError(f"{label} root changed while hashing")
    verified_paths, verified_directories = _enumerate_adapter_tree(
        verified_root,
        label=label,
        maximum_files=maximum_files,
        maximum_directories=maximum_directories,
        maximum_entries=maximum_entries,
    )
    verified_relative_paths = [
        path.relative_to(verified_root).as_posix() for path in verified_paths
    ]
    if (
        verified_relative_paths != relative_paths
        or verified_directories != directories
    ):
        raise RuntimeError(f"{label} tree changed while hashing")
    for path, record in zip(verified_paths, files, strict=True):
        digest, size = hash_file(
            path,
            maximum_bytes=maximum_file_bytes,
            label=f"{label} file {record['path']}",
        )
        if digest != record["sha256"] or size != record["bytes"]:
            raise RuntimeError(f"{label} file changed while hashing: {record['path']}")

    final_lexical, final_root = _resolve_adapter_root(lexical, label=label)
    if final_lexical != lexical or final_root != root:
        raise RuntimeError(f"{label} root changed while hashing")
    final_paths, final_directories = _enumerate_adapter_tree(
        final_root,
        label=label,
        maximum_files=maximum_files,
        maximum_directories=maximum_directories,
        maximum_entries=maximum_entries,
    )
    final_relative_paths = [
        path.relative_to(final_root).as_posix() for path in final_paths
    ]
    if final_relative_paths != relative_paths or final_directories != directories:
        raise RuntimeError(f"{label} tree changed while hashing")

    manifest_digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "root": str(root),
        "file_count": len(files),
        "bytes": total_bytes,
        "manifest_sha256": manifest_digest,
        "files": files,
    }

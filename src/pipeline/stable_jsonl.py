"""Stable, strict and bounded JSONL evidence ingestion."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evidence_paths import resolve_regular_file_without_links
from ..utils.bounded_file import sha256_stable_regular_file


@dataclass(frozen=True)
class StableJsonlDocument:
    path: Path
    rows: list[dict[str, Any]]
    sha256: str
    bytes: int


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON document contains a duplicate key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> object:
    raise ValueError("JSON document contains a non-finite number")


def _reject_decoded_nonfinite_numbers(value: Any) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("JSON document contains a non-finite number")
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def _decode_row(
    raw_line: bytes,
    *,
    label: str,
    line_number: int,
) -> dict[str, Any]:
    try:
        row = json.loads(
            raw_line.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
        _reject_decoded_nonfinite_numbers(row)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        raise ValueError(
            f"{label} line {line_number} is not valid strict UTF-8 JSONL"
        ) from None
    if not isinstance(row, dict):
        raise ValueError(f"{label} line {line_number} is not an object")
    return row


def read_stable_jsonl_mappings(
    path: Path,
    *,
    maximum_bytes: int,
    maximum_line_bytes: int,
    maximum_rows: int,
    label: str,
    expected_sha256: str | None = None,
) -> StableJsonlDocument:
    """Read mappings from one unchanged JSONL file and return its identity."""
    for value, name in (
        (maximum_bytes, "maximum_bytes"),
        (maximum_line_bytes, "maximum_line_bytes"),
        (maximum_rows, "maximum_rows"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")

    resolved = resolve_regular_file_without_links(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
    )
    expected_digest, expected_size = sha256_stable_regular_file(
        resolved,
        maximum_bytes=maximum_bytes,
        label=label,
    )
    if expected_sha256 is not None and expected_digest != expected_sha256:
        raise ValueError(f"{label} checksum does not match")
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    observed_size = 0
    try:
        with resolved.open("rb") as handle:
            line_number = 0
            while raw_line := handle.readline(maximum_line_bytes + 1):
                line_number += 1
                if len(raw_line) > maximum_line_bytes:
                    raise ValueError(
                        f"{label} line {line_number} exceeds "
                        f"{maximum_line_bytes} bytes"
                    )
                observed_size += len(raw_line)
                if observed_size > maximum_bytes:
                    raise ValueError(f"{label} exceeds {maximum_bytes} bytes")
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                rows.append(
                    _decode_row(
                        raw_line,
                        label=label,
                        line_number=line_number,
                    )
                )
                if len(rows) > maximum_rows:
                    raise ValueError(f"{label} exceeds {maximum_rows} rows")
    except OSError:
        raise ValueError(f"Unable to read {label}") from None
    if observed_size != expected_size or digest.hexdigest() != expected_digest:
        raise RuntimeError(f"{label} changed while parsing")
    if not rows:
        raise ValueError(f"{label} is empty")
    return StableJsonlDocument(
        path=resolved,
        rows=rows,
        sha256=expected_digest,
        bytes=expected_size,
    )

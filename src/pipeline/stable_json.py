"""Stable, strict and bounded JSON mapping ingestion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evidence_paths import resolve_regular_file_without_links
from ..utils.bounded_file import read_stable_regular_file


@dataclass(frozen=True)
class StableJsonDocument:
    path: Path
    mapping: dict[str, Any]
    sha256: str
    bytes: int


class StableJsonDigestMismatch(ValueError):
    """The stable JSON payload did not match an external digest binding."""


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


def read_stable_json_mapping(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
    expected_sha256: str | None = None,
) -> StableJsonDocument:
    """Read one unchanged strict-JSON mapping and return its exact identity."""
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
    payload = read_stable_regular_file(
        resolved,
        maximum_bytes=maximum_bytes,
        label=label,
    )
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise StableJsonDigestMismatch(f"{label} checksum does not match")
    try:
        mapping = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError(f"{label} is not valid strict UTF-8 JSON") from None
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} root must be a mapping")
    return StableJsonDocument(
        path=resolved,
        mapping=mapping,
        sha256=digest,
        bytes=len(payload),
    )

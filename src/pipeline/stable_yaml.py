"""Stable, strict and bounded YAML mapping ingestion."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .evidence_paths import resolve_regular_file_without_links
from ..utils.bounded_file import read_stable_regular_file


@dataclass(frozen=True)
class StableYamlDocument:
    path: Path
    mapping: dict[str, Any]
    sha256: str
    bytes: int


class StableYamlDigestMismatch(ValueError):
    """The stable YAML payload did not match an external digest binding."""


class StableYamlEncodingError(ValueError):
    """The stable YAML payload was not strict UTF-8."""


class StableYamlSyntaxError(ValueError):
    """The stable YAML payload violated the strict YAML syntax policy."""


class _StrictYamlLoader(yaml.SafeLoader):
    def __init__(
        self,
        stream: str,
        *,
        label: str,
        maximum_depth: int,
        maximum_nodes: int,
    ) -> None:
        super().__init__(stream)
        self._document_label = label
        self._maximum_depth = maximum_depth
        self._maximum_nodes = maximum_nodes
        self._composition_depth = 0
        self._composed_nodes = 0

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"{self._document_label} aliases are not allowed",
                self.peek_event().start_mark,
            )
        if self._composition_depth >= self._maximum_depth:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"{self._document_label} nesting is too deep",
                self.peek_event().start_mark,
            )
        if self._composed_nodes >= self._maximum_nodes:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"{self._document_label} contains too many nodes",
                self.peek_event().start_mark,
            )
        self._composition_depth += 1
        self._composed_nodes += 1
        try:
            return super().compose_node(parent, index)
        finally:
            self._composition_depth -= 1


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found a duplicate key",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictYamlLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _validate_tree(value: Any, *, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{label} mapping keys must be strings")
        for item in value.values():
            _validate_tree(item, label=label)
        return
    if isinstance(value, list):
        for item in value:
            _validate_tree(item, label=label)
        return
    raise ValueError(f"{label} contains an unsupported YAML type")


def read_stable_yaml_mapping(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
    maximum_depth: int = 32,
    maximum_nodes: int = 10_000,
    expected_sha256: str | None = None,
) -> StableYamlDocument:
    """Read one unchanged strict-YAML mapping and return its exact identity."""
    for value, name in (
        (maximum_bytes, "maximum_bytes"),
        (maximum_depth, "maximum_depth"),
        (maximum_nodes, "maximum_nodes"),
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

    payload = read_stable_regular_file(
        path,
        maximum_bytes=maximum_bytes,
        label=label,
    )
    resolved = resolve_regular_file_without_links(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
    )
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise StableYamlDigestMismatch(f"{label} checksum does not match")
    try:
        content = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise StableYamlEncodingError(f"{label} must be valid UTF-8") from None

    loader = _StrictYamlLoader(
        content,
        label=label,
        maximum_depth=maximum_depth,
        maximum_nodes=maximum_nodes,
    )
    try:
        mapping = loader.get_single_data()
    except (yaml.YAMLError, RecursionError):
        raise StableYamlSyntaxError(f"{label} is not valid strict YAML") from None
    finally:
        loader.dispose()
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} root must be a mapping")
    _validate_tree(mapping, label=label)
    return StableYamlDocument(
        path=resolved,
        mapping=mapping,
        sha256=digest,
        bytes=len(payload),
    )

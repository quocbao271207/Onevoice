import hashlib
from pathlib import Path

import pytest

from src.pipeline.stable_yaml import (
    StableYamlDigestMismatch,
    StableYamlEncodingError,
    StableYamlSyntaxError,
    read_stable_yaml_mapping,
)


def read_yaml(path: Path, **overrides):
    arguments = {
        "maximum_bytes": 10_000,
        "label": "Test config",
    }
    arguments.update(overrides)
    return read_stable_yaml_mapping(path, **arguments)


def test_stable_yaml_returns_mapping_and_exact_identity(tmp_path: Path):
    path = tmp_path / "config.yaml"
    payload = b"version: 1\nitems:\n  - enabled: true\n"
    path.write_bytes(payload)

    document = read_yaml(
        path,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert document.path == path.resolve()
    assert document.mapping == {"version": 1, "items": [{"enabled": True}]}
    assert document.sha256 == hashlib.sha256(payload).hexdigest()
    assert document.bytes == len(payload)


@pytest.mark.parametrize(
    "payload",
    [
        "version: 1\nversion: 2\n",
        "version: &version 1\ncopy: *version\n",
    ],
)
def test_stable_yaml_rejects_duplicate_keys_and_aliases(
    tmp_path: Path,
    payload: str,
):
    path = tmp_path / "config.yaml"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(StableYamlSyntaxError, match="strict YAML"):
        read_yaml(path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("limit: .nan\n", "non-finite"),
        ("date: 2026-10-09\n", "unsupported YAML type"),
        ("1: value\n", "mapping keys must be strings"),
    ],
)
def test_stable_yaml_rejects_non_json_value_types(
    tmp_path: Path,
    payload: str,
    message: str,
):
    path = tmp_path / "config.yaml"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_yaml(path)


def test_stable_yaml_rejects_invalid_encoding_and_digest(tmp_path: Path):
    invalid = tmp_path / "invalid.yaml"
    invalid.write_bytes(b"value: \xff")
    with pytest.raises(StableYamlEncodingError, match="valid UTF-8"):
        read_yaml(invalid)

    valid = tmp_path / "valid.yaml"
    valid.write_text("value: 1\n", encoding="utf-8")
    with pytest.raises(StableYamlDigestMismatch, match="checksum"):
        read_yaml(valid, expected_sha256="0" * 64)


def test_stable_yaml_enforces_depth_node_and_regular_file_boundaries(tmp_path: Path):
    nested = tmp_path / "nested.yaml"
    nested.write_text("value: [[[1]]]\n", encoding="utf-8")
    with pytest.raises(StableYamlSyntaxError, match="strict YAML"):
        read_yaml(nested, maximum_depth=3)

    nodes = tmp_path / "nodes.yaml"
    nodes.write_text("value: [1, 2]\n", encoding="utf-8")
    with pytest.raises(StableYamlSyntaxError, match="strict YAML"):
        read_yaml(nodes, maximum_nodes=2)

    target = tmp_path / "target.yaml"
    target.write_text("value: 1\n", encoding="utf-8")
    linked = tmp_path / "linked.yaml"
    try:
        linked.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="symlink or junction"):
        read_yaml(linked)

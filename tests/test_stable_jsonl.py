from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.pipeline.stable_jsonl import read_stable_jsonl_mappings


def test_stable_jsonl_can_bind_an_external_digest(tmp_path: Path):
    path = tmp_path / "records.jsonl"
    payload = b'{"id":"one"}\n'
    path.write_bytes(payload)

    document = read_stable_jsonl_mappings(
        path,
        maximum_bytes=100,
        maximum_line_bytes=100,
        maximum_rows=10,
        label="Records",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert document.rows == [{"id": "one"}]
    with pytest.raises(ValueError, match="checksum does not match"):
        read_stable_jsonl_mappings(
            path,
            maximum_bytes=100,
            maximum_line_bytes=100,
            maximum_rows=10,
            label="Records",
            expected_sha256="0" * 64,
        )

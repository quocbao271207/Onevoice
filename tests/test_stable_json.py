from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.pipeline.stable_json import (
    StableJsonDigestMismatch,
    read_stable_json_mapping,
)


def test_stable_json_returns_exact_payload_identity(tmp_path: Path):
    path = tmp_path / "evidence.json"
    payload = b'{"status":"pass","count":1}'
    path.write_bytes(payload)

    document = read_stable_json_mapping(
        path,
        maximum_bytes=1_000,
        label="Test evidence",
    )

    assert document.path == path.resolve()
    assert document.mapping == {"status": "pass", "count": 1}
    assert document.sha256 == hashlib.sha256(payload).hexdigest()
    assert document.bytes == len(payload)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"status":"pass","status":"fail"}',
        b'{"score":NaN}',
        b'[{"status":"pass"}]',
    ],
)
def test_stable_json_rejects_non_strict_or_non_mapping_payload(
    tmp_path: Path,
    payload: bytes,
):
    path = tmp_path / "evidence.json"
    path.write_bytes(payload)

    with pytest.raises(ValueError):
        read_stable_json_mapping(
            path,
            maximum_bytes=1_000,
            label="Test evidence",
        )


def test_stable_json_rejects_digest_mismatch(tmp_path: Path):
    path = tmp_path / "evidence.json"
    path.write_text('{"status":"pass"}', encoding="utf-8")

    with pytest.raises(StableJsonDigestMismatch):
        read_stable_json_mapping(
            path,
            maximum_bytes=1_000,
            label="Test evidence",
            expected_sha256="0" * 64,
        )


def test_stable_json_rejects_linked_path_component(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "evidence.json").write_text('{"status":"pass"}', encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="symlink or junction"):
        read_stable_json_mapping(
            link / "evidence.json",
            maximum_bytes=1_000,
            label="Test evidence",
        )

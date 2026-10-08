from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.durable_file import (
    publish_durable_file_exclusive,
    write_durable_bytes_exclusive,
)


def test_write_durable_bytes_exclusive_refuses_overwrite(tmp_path: Path):
    destination = tmp_path / "evidence.sha256"
    write_durable_bytes_exclusive(
        destination,
        b"trusted\n",
        maximum_bytes=100,
        label="Evidence sidecar",
    )

    with pytest.raises(FileExistsError):
        write_durable_bytes_exclusive(
            destination,
            b"replacement\n",
            maximum_bytes=100,
            label="Evidence sidecar",
        )

    assert destination.read_bytes() == b"trusted\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_publish_durable_file_exclusive_binds_staged_identity(tmp_path: Path):
    staged = tmp_path / "staged.part"
    staged.write_bytes(b"archive-payload")
    destination = tmp_path / "archive.tar.gz"

    digest, size = publish_durable_file_exclusive(
        staged,
        destination,
        maximum_bytes=100,
        label="Evidence archive",
    )

    assert len(digest) == 64
    assert size == len(b"archive-payload")
    assert destination.read_bytes() == b"archive-payload"
    with pytest.raises(FileExistsError):
        publish_durable_file_exclusive(
            staged,
            destination,
            maximum_bytes=100,
            label="Evidence archive",
        )
    assert destination.read_bytes() == b"archive-payload"


def test_durable_file_rejects_linked_destination_parent(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    staged = target / "staged.part"
    staged.write_bytes(b"archive-payload")
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="symlink or junction"):
        write_durable_bytes_exclusive(
            link / "evidence.sha256",
            b"trusted\n",
            maximum_bytes=100,
            label="Evidence sidecar",
        )
    with pytest.raises(ValueError, match="symlink or junction"):
        publish_durable_file_exclusive(
            link / staged.name,
            tmp_path / "archive.tar.gz",
            maximum_bytes=100,
            label="Evidence archive",
        )

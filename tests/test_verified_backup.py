from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

import scripts.create_verified_backup as backup


def write_bundle(root: Path, output: Path) -> list[dict[str, object]]:
    reports = root / "data" / "reports"
    reports.mkdir(parents=True)
    (root / "models").mkdir()
    (root / "data" / "manifest.jsonl").write_text('{"id":"sample"}\n', encoding="utf-8")
    (root / "models" / "adapter.bin").write_bytes(b"adapter")
    (reports / "metrics.json").write_text('{"wer":0.1}\n', encoding="utf-8")
    output.mkdir()
    archives = [
        backup.archive(root / "data", output / "data.tar", excluded=[reports]),
        backup.archive(root / "models", output / "models.tar"),
        backup.archive(reports, output / "reports.tar"),
    ]
    (output / "backup_manifest.json").write_text(
        json.dumps({"archives": archives}), encoding="utf-8"
    )
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{item['sha256']}  {item['archive']}\n" for item in archives),
        encoding="utf-8",
    )
    return archives


def test_backup_verifies_checksum_counts_bytes_and_structure(tmp_path: Path, monkeypatch):
    root = tmp_path / "project"
    output = tmp_path / "backup"
    monkeypatch.setattr(backup, "ROOT", root)
    archives = write_bundle(root, output)
    archives[2]["source"] = "data\\reports"
    (output / "backup_manifest.json").write_text(
        json.dumps({"archives": archives}), encoding="utf-8"
    )

    verified = backup.verify(output)
    assert verified["verification"] == "pass"
    assert [item["files"] for item in archives] == [1, 1, 1]

    archives[0]["source_bytes"] = int(archives[0]["source_bytes"]) + 1
    (output / "backup_manifest.json").write_text(
        json.dumps({"archives": archives}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="Source byte-count mismatch"):
        backup.verify(output)


def test_backup_rejects_empty_source(tmp_path: Path, monkeypatch):
    root = tmp_path / "project"
    source = root / "models"
    source.mkdir(parents=True)
    monkeypatch.setattr(backup, "ROOT", root)
    with pytest.raises(ValueError, match="empty backup archive"):
        backup.archive(source, tmp_path / "models.tar")


def test_backup_rejects_unsafe_tar_member(tmp_path: Path, monkeypatch):
    root = tmp_path / "project"
    output = tmp_path / "backup"
    monkeypatch.setattr(backup, "ROOT", root)
    archives = write_bundle(root, output)

    payload = b"escape"
    with tarfile.open(output / "data.tar", mode="w") as tar:
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    archives[0].update(
        {
            "files": 1,
            "source_bytes": len(payload),
            "archive_bytes": (output / "data.tar").stat().st_size,
            "sha256": backup.sha256(output / "data.tar"),
        }
    )
    (output / "backup_manifest.json").write_text(
        json.dumps({"archives": archives}), encoding="utf-8"
    )
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{item['sha256']}  {item['archive']}\n" for item in archives),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Unsafe or misplaced member"):
        backup.verify(output)

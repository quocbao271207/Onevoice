from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

import scripts.create_verified_backup as backup


def write_metadata(output: Path, archives: list[dict[str, object]]) -> None:
    backup.write_metadata(
        output,
        {
            "schema_version": backup.SCHEMA_VERSION,
            "format": "test",
            "archives": archives,
        },
    )


def write_sources(root: Path) -> Path:
    reports = root / "data" / "reports"
    reports.mkdir(parents=True)
    (root / "models").mkdir()
    (root / "data" / "manifest.jsonl").write_text('{"id":"sample"}\n', encoding="utf-8")
    (root / "models" / "adapter.bin").write_bytes(b"adapter")
    (reports / "metrics.json").write_text('{"wer":0.1}\n', encoding="utf-8")
    return reports


def write_bundle(root: Path, output: Path) -> list[dict[str, object]]:
    reports = write_sources(root)
    output.mkdir()
    archives = [
        backup.archive(root / "data", output / "data.tar", excluded=[reports]),
        backup.archive(root / "models", output / "models.tar"),
        backup.archive(reports, output / "reports.tar"),
    ]
    write_metadata(output, archives)
    return archives


def rewrite_tar(path: Path, name: str, payload: bytes) -> None:
    with tarfile.open(path, mode="w", format=tarfile.PAX_FORMAT) as tar:
        member = tarfile.TarInfo(name)
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))


def refresh_archive_envelope(item: dict[str, object], path: Path) -> None:
    item["archive_bytes"] = path.stat().st_size
    item["sha256"] = backup.sha256(path)


def test_backup_verifies_member_checksums_counts_bytes_and_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "project"
    output = tmp_path / "backup"
    monkeypatch.setattr(backup, "ROOT", root)
    archives = write_bundle(root, output)
    archives[2]["source"] = "data\\reports"
    write_metadata(output, archives)

    verified = backup.verify(output)
    assert verified["verification"] == "pass"
    assert verified["schema_version"] == backup.SCHEMA_VERSION
    assert [item["files"] for item in archives] == [1, 1, 1]
    assert all(len(item["members"]) == 1 for item in archives)

    archives[0]["source_bytes"] = int(archives[0]["source_bytes"]) + 1
    write_metadata(output, archives)
    with pytest.raises(RuntimeError, match="Source byte-count mismatch in manifest"):
        backup.verify(output)


def test_backup_rejects_empty_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "project"
    source = root / "models"
    source.mkdir(parents=True)
    monkeypatch.setattr(backup, "ROOT", root)
    with pytest.raises(ValueError, match="empty backup archive"):
        backup.archive(source, tmp_path / "models.tar")


def test_backup_rejects_unsafe_tar_member(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "project"
    output = tmp_path / "backup"
    monkeypatch.setattr(backup, "ROOT", root)
    archives = write_bundle(root, output)

    payload = b"escape"
    rewrite_tar(output / "data.tar", "../escape.txt", payload)
    archives[0].update(
        {
            "files": 1,
            "source_bytes": len(payload),
            "members": [
                {
                    "path": "../escape.txt",
                    "bytes": len(payload),
                    "sha256": backup.sha256(root / "models" / "adapter.bin"),
                }
            ],
        }
    )
    refresh_archive_envelope(archives[0], output / "data.tar")
    write_metadata(output, archives)
    with pytest.raises(RuntimeError, match="Unsafe or misplaced member"):
        backup.verify(output)


def test_backup_detects_same_size_member_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "project"
    output = tmp_path / "backup"
    monkeypatch.setattr(backup, "ROOT", root)
    archives = write_bundle(root, output)
    member = archives[0]["members"][0]
    payload = b"x" * int(member["bytes"])
    rewrite_tar(output / "data.tar", str(member["path"]), payload)
    refresh_archive_envelope(archives[0], output / "data.tar")
    write_metadata(output, archives)

    with pytest.raises(RuntimeError, match="Member checksum mismatch"):
        backup.verify(output)


def test_backup_rejects_non_regular_tar_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "project"
    output = tmp_path / "backup"
    monkeypatch.setattr(backup, "ROOT", root)
    archives = write_bundle(root, output)
    member_name = str(archives[0]["members"][0]["path"])
    with tarfile.open(output / "data.tar", mode="w") as tar:
        member = tarfile.TarInfo(member_name)
        member.type = tarfile.SYMTYPE
        member.linkname = "../../escape"
        tar.addfile(member)
    refresh_archive_envelope(archives[0], output / "data.tar")
    write_metadata(output, archives)

    with pytest.raises(RuntimeError, match="Non-regular tar member"):
        backup.verify(output)


def test_backup_rejects_manifest_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "project"
    output = tmp_path / "backup"
    monkeypatch.setattr(backup, "ROOT", root)
    write_bundle(root, output)
    manifest = output / backup.MANIFEST_NAME
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(RuntimeError, match=backup.MANIFEST_CHECKSUM_NAME):
        backup.verify(output)


def test_backup_rejects_extra_output_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "project"
    output = tmp_path / "backup"
    monkeypatch.setattr(backup, "ROOT", root)
    write_bundle(root, output)
    (output / "unexpected.txt").write_text("not sealed", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Backup file set mismatch"):
        backup.verify(output)


def test_backup_rejects_source_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "project"
    source = root / "models"
    source.mkdir(parents=True)
    target = root / "target.bin"
    target.write_bytes(b"model")
    link = source / "adapter.bin"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"Symlink creation is unavailable: {error}")
    monkeypatch.setattr(backup, "ROOT", root)

    with pytest.raises(ValueError, match="source entry symlink or junction"):
        backup.archive(source, tmp_path / "models.tar")


def test_backup_rejects_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "project"
    source = root / "models"
    source.mkdir(parents=True)
    (source / "adapter.bin").write_bytes(b"model")
    destination = tmp_path / "models.tar"
    destination.write_bytes(b"existing")
    monkeypatch.setattr(backup, "ROOT", root)

    with pytest.raises(FileExistsError):
        backup.archive(source, destination)
    assert destination.read_bytes() == b"existing"


def test_create_backup_publishes_only_after_full_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "project"
    output = tmp_path / "published-backup"
    write_sources(root)
    monkeypatch.setattr(backup, "ROOT", root)

    verified = backup.create_backup(output)

    assert verified["verification"] == "pass"
    assert {path.name for path in output.iterdir()} == backup.EXPECTED_OUTPUT_NAMES
    assert backup.verify(output)["verification"] == "pass"
    assert not list(tmp_path.glob(f".{output.name}.part-*"))


def test_create_backup_removes_staging_directory_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "project"
    output = tmp_path / "failed-backup"
    write_sources(root)
    monkeypatch.setattr(backup, "ROOT", root)
    original_archive = backup.archive
    calls = 0

    def fail_on_second_archive(
        source: Path, destination: Path, excluded: tuple[Path, ...] | list[Path] = ()
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated archive failure")
        return original_archive(source, destination, excluded)

    monkeypatch.setattr(backup, "archive", fail_on_second_archive)

    with pytest.raises(RuntimeError, match="simulated archive failure"):
        backup.create_backup(output)
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.part-*"))

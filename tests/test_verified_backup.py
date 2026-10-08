from __future__ import annotations

import io
import json
import os
import stat
import tarfile
from pathlib import Path

import pytest

import scripts.create_verified_backup as backup


def write_metadata(output: Path, archives: list[dict[str, object]]) -> None:
    backup.write_metadata(
        output,
        {
            "schema_version": backup.SCHEMA_VERSION,
            "created_at": "2026-10-09T00:00:00+00:00",
            "project_root": str(backup.ROOT),
            "git_head": None,
            "format": backup.BACKUP_FORMAT,
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
        member.mode = backup.CANONICAL_TAR_MODE
        member.mtime = backup.CANONICAL_TAR_MTIME
        member.uid = backup.CANONICAL_TAR_OWNER_ID
        member.gid = backup.CANONICAL_TAR_OWNER_ID
        member.uname = ""
        member.gname = ""
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


def test_backup_tar_is_deterministic_and_has_canonical_private_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "project"
    source = root / "models"
    source.mkdir(parents=True)
    model = source / "adapter.bin"
    model.write_bytes(b"model")
    monkeypatch.setattr(backup, "ROOT", root)

    first = tmp_path / "first.tar"
    second = tmp_path / "second.tar"
    backup.archive(source, first)
    os.utime(model, (1_700_000_000, 1_700_000_000))
    backup.archive(source, second)

    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, mode="r") as tar:
        member = tar.getmembers()[0]
    assert member.mode == backup.CANONICAL_TAR_MODE
    assert member.mtime == backup.CANONICAL_TAR_MTIME
    assert member.uid == member.gid == backup.CANONICAL_TAR_OWNER_ID
    assert member.uname == member.gname == ""
    if os.name == "posix":
        assert stat.S_IMODE(first.stat().st_mode) == backup.CANONICAL_TAR_MODE


def test_backup_rejects_noncanonical_tar_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "project"
    output = tmp_path / "backup"
    monkeypatch.setattr(backup, "ROOT", root)
    archives = write_bundle(root, output)
    member_name = str(archives[0]["members"][0]["path"])
    payload = (root / member_name).read_bytes()
    with tarfile.open(output / "data.tar", mode="w", format=tarfile.PAX_FORMAT) as tar:
        member = tarfile.TarInfo(member_name)
        member.size = len(payload)
        member.mode = 0o644
        member.mtime = backup.CANONICAL_TAR_MTIME
        member.uname = "local-user"
        tar.addfile(member, io.BytesIO(payload))
    refresh_archive_envelope(archives[0], output / "data.tar")
    write_metadata(output, archives)

    with pytest.raises(RuntimeError, match="Non-canonical tar metadata"):
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


def test_backup_rejects_noncanonical_manifest_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "project"
    output = tmp_path / "backup"
    monkeypatch.setattr(backup, "ROOT", root)
    write_bundle(root, output)
    manifest_path = output / backup.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["untrusted_note"] = "looks verified"
    backup.write_metadata(output, manifest)
    with pytest.raises(ValueError, match="manifest schema is invalid"):
        backup.verify(output)

    manifest.pop("untrusted_note")
    manifest["archives"][0]["untrusted_note"] = "looks verified"
    backup.write_metadata(output, manifest)
    with pytest.raises(ValueError, match="archive record schema is invalid"):
        backup.verify(output)

    manifest["archives"][0].pop("untrusted_note")
    manifest["archives"][0], manifest["archives"][1] = (
        manifest["archives"][1],
        manifest["archives"][0],
    )
    backup.write_metadata(output, manifest)
    with pytest.raises(ValueError, match="canonical order"):
        backup.verify(output)


def test_backup_rejects_invalid_manifest_types_and_unbounded_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "project"
    output = tmp_path / "backup"
    monkeypatch.setattr(backup, "ROOT", root)
    write_bundle(root, output)
    manifest_path = output / backup.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["created_at"] = "2026-10-09T00:00:00"
    backup.write_metadata(output, manifest)
    with pytest.raises(ValueError, match="must include a timezone"):
        backup.verify(output)

    manifest["created_at"] = "2026-10-09T00:00:00+00:00"
    manifest["git_head"] = "not-a-commit"
    backup.write_metadata(output, manifest)
    with pytest.raises(ValueError, match="git_head is invalid"):
        backup.verify(output)

    manifest["git_head"] = None
    manifest["archives"][0]["files"] = "1"
    backup.write_metadata(output, manifest)
    with pytest.raises(ValueError, match="Invalid integer"):
        backup.verify(output)

    manifest["archives"][0]["files"] = 1
    backup.write_metadata(output, manifest)
    manifest_path.write_bytes(b"x" * (backup.MAX_BACKUP_MANIFEST_BYTES + 1))
    with pytest.raises(ValueError, match="manifest size is outside"):
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


def test_create_backup_rejects_linked_output_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "project"
    write_sources(root)
    real_parent = tmp_path / "real-backups"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-backups"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Directory symlink creation is unavailable: {error}")
    monkeypatch.setattr(backup, "ROOT", root)

    with pytest.raises(ValueError, match="backup output ancestor symlink or junction"):
        backup.create_backup(linked_parent / "release")
    assert not (real_parent / "release").exists()

    backup.create_backup(real_parent / "release")
    with pytest.raises(ValueError, match="backup directory ancestor symlink or junction"):
        backup.verify(linked_parent / "release")


def test_release_backup_binds_terminal_comparison_and_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "project"
    output = tmp_path / "release-backup"
    reports = write_sources(root)
    comparison = reports / "model_bakeoff" / "comparison.json"
    comparison.parent.mkdir()
    comparison.write_text(
        json.dumps(
            {
                "status": "complete",
                "scope": "production",
                "promotion_allowed": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(backup, "ROOT", root)
    release = backup.build_release_evidence(
        comparison,
        decision="promote",
        scope="production",
    )

    verified = backup.create_backup(output, release_evidence=release)

    assert verified["release_evidence"] == release
    assert backup.verify(output)["release_evidence"] == release

    manifest_path = output / backup.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["release_evidence"]["decision"] = "reject"
    backup.write_metadata(output, manifest)
    with pytest.raises(ValueError, match="decision disagrees"):
        backup.verify(output)


def test_release_evidence_rejects_nonterminal_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "project"
    reports = write_sources(root)
    comparison = reports / "model_bakeoff" / "comparison.json"
    comparison.parent.mkdir()
    comparison.write_text(
        json.dumps(
            {
                "status": "blind_complete",
                "scope": "research",
                "promotion_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(backup, "ROOT", root)

    with pytest.raises(ValueError, match="not terminal"):
        backup.build_release_evidence(
            comparison,
            decision="reject",
            scope="research",
        )


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

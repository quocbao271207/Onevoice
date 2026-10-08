"""Shared helpers for immutable candidate-evaluation evidence bundles."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from src.pipeline.durable_file import (  # noqa: E402
    publish_durable_file_exclusive,
    write_durable_bytes_exclusive,
)
from src.pipeline.durable_json import write_durable_json_exclusive  # noqa: E402
from src.pipeline.evidence_paths import (  # noqa: E402
    is_link_or_junction,
    prepare_new_file_destination_without_links,
    resolve_regular_directory_without_links,
    resolve_regular_file_without_links,
)
from src.pipeline.stable_json import read_stable_json_mapping
from src.utils.bounded_file import (
    read_stable_regular_file,
    sha256_stable_regular_file,
)


MAX_EVIDENCE_ARCHIVE_BYTES = 16_000_000_000
MAX_EVIDENCE_MANIFEST_BYTES = 100_000_000
MAX_EVIDENCE_CHECKSUM_BYTES = 4_096
MAX_EVIDENCE_REPORT_BYTES = 100_000_000
MAX_EVIDENCE_MEMBERS = 100_000
MAX_EVIDENCE_CONTENT_BYTES = 64_000_000_000


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    """Hash JSON data using the canonical encoding used by provenance records."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def adapter_identity(adapter: Path | None) -> dict[str, Any] | None:
    """Return the exact path/size/content identity embedded in prediction provenance."""
    if adapter is None:
        return None
    root = adapter.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing adapter directory: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Adapter checkpoint cannot contain symlinks: {path}")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    if not files:
        raise ValueError(f"Adapter checkpoint is empty: {root}")
    return {
        "path": str(root),
        "file_count": len(files),
        "bytes": sum(int(item["bytes"]) for item in files),
        "manifest_sha256": canonical_sha256(files),
    }


def read_json(path: Path) -> dict[str, Any]:
    return read_stable_json_mapping(
        path,
        maximum_bytes=MAX_EVIDENCE_REPORT_BYTES,
        label="Candidate evidence JSON",
    ).mapping


def evidence_sidecars(archive_path: Path) -> tuple[Path, Path]:
    """Return the checksum and content-manifest paths for an evidence archive."""
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    manifest_path = archive_path.with_suffix(archive_path.suffix + ".manifest.json")
    return checksum_path, manifest_path


def _safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def _archive_files(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    content_bytes = 0
    for path in sorted(output_dir.rglob("*")):
        if is_link_or_junction(path):
            raise ValueError(f"Evidence bundle cannot contain symlinks: {path}")
        if not path.is_file():
            continue
        archive_name = (Path(output_dir.name) / path.relative_to(output_dir)).as_posix()
        if not _safe_member_name(archive_name):
            raise ValueError(f"Unsafe evidence archive member: {archive_name!r}")
        digest, size = sha256_stable_regular_file(
            path,
            maximum_bytes=MAX_EVIDENCE_ARCHIVE_BYTES,
            label="Evidence source file",
        )
        records.append(
            {
                "path": archive_name,
                "bytes": size,
                "sha256": digest,
                "source": path,
            }
        )
        content_bytes += size
        if len(records) > MAX_EVIDENCE_MEMBERS:
            raise ValueError("Evidence directory has too many files")
        if content_bytes > MAX_EVIDENCE_CONTENT_BYTES:
            raise ValueError("Evidence directory exceeds the content byte limit")
    if not records:
        raise ValueError(f"Evidence directory is empty: {output_dir}")
    return records


def _public_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: record[key] for key in ("path", "bytes", "sha256")}
        for record in records
    ]


def _assert_live_tree_unchanged(
    output_dir: Path,
    expected: list[dict[str, Any]],
) -> None:
    if _public_records(_archive_files(output_dir)) != expected:
        raise RuntimeError("Evidence directory changed while archiving")


def _verify_payload(
    archive_path: Path,
    expected_files: list[dict[str, Any]],
) -> int:
    if len(expected_files) > MAX_EVIDENCE_MEMBERS:
        raise ValueError("Evidence content manifest has too many files")
    paths: list[str] = []
    content_bytes = 0
    for item in expected_files:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("Invalid evidence content record")
        path = item.get("path")
        size = item.get("bytes")
        digest = item.get("sha256")
        if (
            not isinstance(path, str)
            or not _safe_member_name(path)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Invalid evidence content record: {path!r}")
        paths.append(path)
        content_bytes += size
    if len(paths) != len(set(paths)):
        raise ValueError("Evidence content manifest contains duplicate paths")
    if content_bytes > MAX_EVIDENCE_CONTENT_BYTES:
        raise ValueError("Evidence content manifest exceeds the byte limit")
    expected = {str(item["path"]): item for item in expected_files}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > MAX_EVIDENCE_MEMBERS:
            raise ValueError("Evidence archive has too many members")
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Evidence archive contains duplicate member paths")
        if any(not member.isfile() or not _safe_member_name(member.name) for member in members):
            raise ValueError("Evidence archive contains an unsafe or non-file member")
        if set(names) != set(expected):
            raise ValueError("Evidence archive members do not match the content manifest")
        if sum(member.size for member in members) != content_bytes:
            raise ValueError("Evidence archive content byte count does not match")
        for member in members:
            if member.size != int(expected[member.name]["bytes"]):
                raise ValueError(
                    f"Evidence archive member size does not match: {member.name}"
                )
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Cannot read evidence archive member: {member.name}")
            member_digest = hashlib.sha256()
            size = 0
            while chunk := handle.read(8 * 1024 * 1024):
                size += len(chunk)
                member_digest.update(chunk)
            record = expected[member.name]
            if size != int(record["bytes"]) or member_digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Evidence archive member verification failed: {member.name}")
    return content_bytes


def verify_evidence_archive(archive_path: Path) -> dict[str, Any]:
    """Verify archive bytes, checksum sidecar, and every manifest member."""
    checksum_path, manifest_path = evidence_sidecars(archive_path)
    try:
        archive_path = resolve_regular_file_without_links(
            archive_path,
            label="Evidence archive",
            maximum_bytes=MAX_EVIDENCE_ARCHIVE_BYTES,
        )
        checksum_path = resolve_regular_file_without_links(
            checksum_path,
            label="Evidence checksum sidecar",
            maximum_bytes=MAX_EVIDENCE_CHECKSUM_BYTES,
        )
        manifest_path = resolve_regular_file_without_links(
            manifest_path,
            label="Evidence content manifest",
            maximum_bytes=MAX_EVIDENCE_MANIFEST_BYTES,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Incomplete evidence bundle for {archive_path}"
        ) from None
    checksum_payload = read_stable_regular_file(
        checksum_path,
        maximum_bytes=MAX_EVIDENCE_CHECKSUM_BYTES,
        label="Evidence checksum sidecar",
    )
    try:
        fields = checksum_payload.decode("utf-8", errors="strict").split()
    except UnicodeDecodeError:
        raise ValueError(f"Invalid evidence checksum sidecar: {checksum_path}") from None
    if len(fields) != 2 or fields[1] != archive_path.name:
        raise ValueError(f"Invalid evidence checksum sidecar: {checksum_path}")
    digest, archive_bytes = sha256_stable_regular_file(
        archive_path,
        maximum_bytes=MAX_EVIDENCE_ARCHIVE_BYTES,
        label="Evidence archive",
    )
    if fields[0].lower() != digest:
        raise ValueError(f"Evidence archive checksum mismatch: {archive_path}")
    manifest_document = read_stable_json_mapping(
        manifest_path,
        maximum_bytes=MAX_EVIDENCE_MANIFEST_BYTES,
        label="Evidence content manifest",
    )
    manifest = manifest_document.mapping
    if set(manifest) != {
        "schema_version",
        "created_at",
        "archive",
        "archive_bytes",
        "archive_sha256",
        "file_count",
        "content_bytes",
        "files",
    }:
        raise ValueError(f"Invalid evidence content manifest: {manifest_path}")
    created_at = manifest.get("created_at")
    try:
        created_time = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        if not isinstance(created_at, str) or created_time.tzinfo is None:
            raise ValueError("timezone required")
    except ValueError:
        raise ValueError(f"Invalid evidence content manifest: {manifest_path}") from None
    if (
        manifest.get("schema_version") != 1
        or manifest.get("archive") != archive_path.name
        or isinstance(manifest.get("archive_bytes"), bool)
        or not isinstance(manifest.get("archive_bytes"), int)
        or manifest.get("archive_bytes") != archive_bytes
        or manifest.get("archive_sha256") != digest
    ):
        raise ValueError(f"Evidence archive metadata mismatch: {manifest_path}")
    files = manifest.get("files")
    file_count = manifest.get("file_count")
    content_bytes = manifest.get("content_bytes")
    if (
        not isinstance(files, list)
        or isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count != len(files)
    ):
        raise ValueError(f"Invalid evidence content manifest: {manifest_path}")
    if isinstance(content_bytes, bool) or not isinstance(content_bytes, int):
        raise ValueError(f"Invalid evidence content manifest: {manifest_path}")
    verified_content_bytes = _verify_payload(archive_path, files)
    if content_bytes != verified_content_bytes:
        raise ValueError(f"Evidence content byte count mismatch: {manifest_path}")
    final_digest, final_bytes = sha256_stable_regular_file(
        archive_path,
        maximum_bytes=MAX_EVIDENCE_ARCHIVE_BYTES,
        label="Evidence archive",
    )
    if final_digest != digest or final_bytes != archive_bytes:
        raise RuntimeError("Evidence archive changed while verifying")
    final_checksum_payload = read_stable_regular_file(
        checksum_path,
        maximum_bytes=MAX_EVIDENCE_CHECKSUM_BYTES,
        label="Evidence checksum sidecar",
    )
    if final_checksum_payload != checksum_payload:
        raise RuntimeError("Evidence checksum sidecar changed while verifying")
    final_manifest = read_stable_json_mapping(
        manifest_path,
        maximum_bytes=MAX_EVIDENCE_MANIFEST_BYTES,
        label="Evidence content manifest",
        expected_sha256=manifest_document.sha256,
    )
    if final_manifest.bytes != manifest_document.bytes:
        raise RuntimeError("Evidence content manifest changed while verifying")
    return manifest


def derive_legacy_evidence_manifest(
    archive_path: Path,
    output_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Inventory a checksum-verified legacy archive without making it resume-eligible.

    Older runners emitted an archive and SHA-256 sidecar but no content manifest.
    This forensic record preserves what was actually downloaded while remaining
    deliberately distinct from the canonical ``.manifest.json`` contract.
    """
    if archive_path.is_symlink():
        raise ValueError(f"Legacy evidence archive must not be a symlink: {archive_path}")
    archive_path = archive_path.resolve()
    checksum_path, canonical_manifest_path = evidence_sidecars(archive_path)
    if canonical_manifest_path.exists():
        raise ValueError(
            f"Canonical evidence manifest already exists; verify it instead: "
            f"{canonical_manifest_path}"
        )
    if not archive_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(f"Incomplete legacy evidence bundle for {archive_path}")
    if checksum_path.is_symlink():
        raise ValueError(f"Evidence checksum sidecar must not be a symlink: {checksum_path}")
    fields = checksum_path.read_text(encoding="utf-8").split()
    if len(fields) != 2 or fields[1] != archive_path.name:
        raise ValueError(f"Invalid evidence checksum sidecar: {checksum_path}")
    digest = sha256(archive_path)
    if fields[0].lower() != digest:
        raise ValueError(f"Evidence archive checksum mismatch: {archive_path}")

    records: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Evidence archive contains duplicate member paths")
        if any(not member.isfile() or not _safe_member_name(member.name) for member in members):
            raise ValueError("Evidence archive contains an unsafe or non-file member")
        for member in members:
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Cannot read evidence archive member: {member.name}")
            member_digest = hashlib.sha256()
            size = 0
            while chunk := handle.read(8 * 1024 * 1024):
                size += len(chunk)
                member_digest.update(chunk)
            records.append(
                {
                    "path": member.name,
                    "bytes": size,
                    "sha256": member_digest.hexdigest(),
                }
            )
    records.sort(key=lambda record: str(record["path"]))
    _verify_payload(archive_path, records)

    requested_output = (
        output_path
        if output_path is not None
        else archive_path.with_name(f"{archive_path.name}.derived-manifest.json")
    )
    if requested_output.is_symlink():
        raise ValueError(f"Derived manifest output must not be a symlink: {requested_output}")
    derived_path = requested_output.resolve()
    reserved_paths = {
        archive_path,
        checksum_path.resolve(),
        canonical_manifest_path.resolve(),
    }
    if derived_path in reserved_paths or not derived_path.name.endswith(
        ".derived-manifest.json"
    ):
        raise ValueError(
            "Derived manifest output must use a distinct *.derived-manifest.json path"
        )
    payload = {
        "schema_version": 1,
        "evidence_status": "forensic_derived_from_legacy_archive",
        "resume_eligible": False,
        "original_manifest_present": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": digest,
        "checksum_sidecar": {
            "path": checksum_path.name,
            "bytes": checksum_path.stat().st_size,
            "sha256": sha256(checksum_path),
        },
        "file_count": len(records),
        "content_bytes": sum(int(record["bytes"]) for record in records),
        "files": records,
    }
    if derived_path.exists():
        existing = read_stable_json_mapping(
            derived_path,
            maximum_bytes=MAX_EVIDENCE_MANIFEST_BYTES,
            label="Derived evidence manifest",
        ).mapping
        comparable_keys = {
            "schema_version",
            "evidence_status",
            "resume_eligible",
            "original_manifest_present",
            "archive",
            "archive_bytes",
            "archive_sha256",
            "checksum_sidecar",
            "file_count",
            "content_bytes",
            "files",
        }
        if any(existing.get(key) != payload.get(key) for key in comparable_keys):
            raise ValueError(f"Existing derived manifest does not match archive: {derived_path}")
        return derived_path, existing
    write_durable_json_exclusive(
        derived_path,
        payload,
        maximum_bytes=MAX_EVIDENCE_MANIFEST_BYTES,
        label="Derived evidence manifest",
    )
    return derived_path, payload


def _manifest_payload(
    archive_path: Path,
    *,
    archive_bytes: int,
    archive_sha256: str,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": archive_path.name,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "file_count": len(files),
        "content_bytes": sum(int(record["bytes"]) for record in files),
        "files": files,
    }


def _existing_manifest_matches(
    manifest_path: Path,
    expected: dict[str, Any],
) -> None:
    existing = read_stable_json_mapping(
        manifest_path,
        maximum_bytes=MAX_EVIDENCE_MANIFEST_BYTES,
        label="Evidence content manifest",
    ).mapping
    comparable = set(expected) - {"created_at"}
    if set(existing) != set(expected) or any(
        existing.get(key) != expected.get(key) for key in comparable
    ):
        raise ValueError("Existing evidence manifest does not match archive contents")


def _existing_checksum_matches(
    checksum_path: Path,
    *,
    archive_path: Path,
    digest: str,
) -> None:
    resolved = resolve_regular_file_without_links(
        checksum_path,
        label="Evidence checksum sidecar",
        maximum_bytes=MAX_EVIDENCE_CHECKSUM_BYTES,
    )
    payload = read_stable_regular_file(
        resolved,
        maximum_bytes=MAX_EVIDENCE_CHECKSUM_BYTES,
        label="Evidence checksum sidecar",
    )
    try:
        fields = payload.decode("utf-8", errors="strict").split()
    except UnicodeDecodeError:
        raise ValueError("Existing evidence checksum sidecar is invalid") from None
    if len(fields) != 2 or fields != [digest, archive_path.name]:
        raise ValueError("Existing evidence checksum sidecar does not match archive")


def _finish_existing_bundle(
    archive_path: Path,
    checksum_path: Path,
    manifest_path: Path,
    files: list[dict[str, Any]],
) -> tuple[Path, str]:
    archive_path = resolve_regular_file_without_links(
        archive_path,
        label="Evidence archive",
        maximum_bytes=MAX_EVIDENCE_ARCHIVE_BYTES,
    )
    digest, archive_bytes = sha256_stable_regular_file(
        archive_path,
        maximum_bytes=MAX_EVIDENCE_ARCHIVE_BYTES,
        label="Evidence archive",
    )
    _verify_payload(archive_path, files)
    manifest = _manifest_payload(
        archive_path,
        archive_bytes=archive_bytes,
        archive_sha256=digest,
        files=files,
    )
    if checksum_path.exists() or is_link_or_junction(checksum_path):
        _existing_checksum_matches(
            checksum_path,
            archive_path=archive_path,
            digest=digest,
        )
    else:
        write_durable_bytes_exclusive(
            checksum_path,
            f"{digest}  {archive_path.name}\n".encode("utf-8"),
            maximum_bytes=MAX_EVIDENCE_CHECKSUM_BYTES,
            label="Evidence checksum sidecar",
        )
    if manifest_path.exists() or is_link_or_junction(manifest_path):
        _existing_manifest_matches(manifest_path, manifest)
    else:
        write_durable_json_exclusive(
            manifest_path,
            manifest,
            maximum_bytes=MAX_EVIDENCE_MANIFEST_BYTES,
            label="Evidence content manifest",
        )
    verified = verify_evidence_archive(archive_path)
    if verified["files"] != files:
        raise ValueError("Existing evidence bundle does not match live contents")
    return archive_path, digest


def archive_evidence(
    output_dir: Path,
    archive_path: Path | None = None,
) -> tuple[Path, str]:
    """Create and fully verify an immutable archive/checksum/manifest bundle."""
    output_dir = resolve_regular_directory_without_links(
        output_dir,
        label="Evidence directory",
    )
    archive_path = (
        archive_path if archive_path is not None else output_dir.with_suffix(".tar.gz")
    )
    archive_path = Path(os.path.abspath(archive_path))
    checksum_path, manifest_path = evidence_sidecars(archive_path)
    records = _archive_files(output_dir)
    public_records = _public_records(records)
    existing = [
        path.exists() or is_link_or_junction(path)
        for path in (archive_path, checksum_path, manifest_path)
    ]
    if all(existing):
        verified = verify_evidence_archive(archive_path)
        if verified["files"] != public_records:
            raise ValueError("Existing evidence bundle does not match live contents")
        _assert_live_tree_unchanged(output_dir, public_records)
        return archive_path.resolve(strict=True), str(verified["archive_sha256"])
    if any(existing):
        if not existing[0]:
            raise ValueError("Evidence sidecar exists without its archive")
        result = _finish_existing_bundle(
            archive_path,
            checksum_path,
            manifest_path,
            public_records,
        )
        _assert_live_tree_unchanged(output_dir, public_records)
        return result

    archive_path = prepare_new_file_destination_without_links(
        archive_path,
        label="Evidence archive",
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive_path.name}.",
        suffix=".part",
        dir=archive_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for record in records:
                archive.add(
                    record["source"],
                    arcname=record["path"],
                    recursive=False,
                )
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        _verify_payload(temporary, public_records)
        staged_digest, _ = sha256_stable_regular_file(
            temporary,
            maximum_bytes=MAX_EVIDENCE_ARCHIVE_BYTES,
            label="Staged evidence archive",
        )
        digest, archive_bytes = publish_durable_file_exclusive(
            temporary,
            archive_path,
            maximum_bytes=MAX_EVIDENCE_ARCHIVE_BYTES,
            label="Evidence archive",
            expected_sha256=staged_digest,
        )
    finally:
        temporary.unlink(missing_ok=True)

    manifest = _manifest_payload(
        archive_path,
        archive_bytes=archive_bytes,
        archive_sha256=digest,
        files=public_records,
    )
    write_durable_bytes_exclusive(
        checksum_path,
        f"{digest}  {archive_path.name}\n".encode("utf-8"),
        maximum_bytes=MAX_EVIDENCE_CHECKSUM_BYTES,
        label="Evidence checksum sidecar",
    )
    write_durable_json_exclusive(
        manifest_path,
        manifest,
        maximum_bytes=MAX_EVIDENCE_MANIFEST_BYTES,
        label="Evidence content manifest",
    )
    verify_evidence_archive(archive_path)
    _assert_live_tree_unchanged(output_dir, public_records)
    return archive_path, digest

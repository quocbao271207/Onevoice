"""Shared helpers for immutable candidate-evaluation evidence bundles."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


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
    return json.loads(path.read_text(encoding="utf-8"))


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
    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Evidence bundle cannot contain symlinks: {path}")
        if not path.is_file():
            continue
        archive_name = (Path(output_dir.name) / path.relative_to(output_dir)).as_posix()
        if not _safe_member_name(archive_name):
            raise ValueError(f"Unsafe evidence archive member: {archive_name!r}")
        records.append(
            {
                "path": archive_name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "source": path,
            }
        )
    if not records:
        raise ValueError(f"Evidence directory is empty: {output_dir}")
    return records


def _verify_payload(archive_path: Path, expected_files: list[dict[str, Any]]) -> None:
    paths = [str(item.get("path", "")) for item in expected_files]
    if len(paths) != len(set(paths)):
        raise ValueError("Evidence content manifest contains duplicate paths")
    for item, path in zip(expected_files, paths):
        digest = item.get("sha256")
        try:
            size = int(item.get("bytes", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid evidence content size: {path}") from exc
        if (
            not _safe_member_name(path)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise ValueError(f"Invalid evidence content record: {path!r}")
    expected = {str(item["path"]): item for item in expected_files}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Evidence archive contains duplicate member paths")
        if any(not member.isfile() or not _safe_member_name(member.name) for member in members):
            raise ValueError("Evidence archive contains an unsafe or non-file member")
        if set(names) != set(expected):
            raise ValueError("Evidence archive members do not match the content manifest")
        for member in members:
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


def verify_evidence_archive(archive_path: Path) -> dict[str, Any]:
    """Verify archive bytes, checksum sidecar, and every manifest member."""
    checksum_path, manifest_path = evidence_sidecars(archive_path)
    if not archive_path.is_file() or not checksum_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete evidence bundle for {archive_path}")
    fields = checksum_path.read_text(encoding="utf-8").split()
    if len(fields) != 2 or fields[1] != archive_path.name:
        raise ValueError(f"Invalid evidence checksum sidecar: {checksum_path}")
    digest = sha256(archive_path)
    if fields[0].lower() != digest:
        raise ValueError(f"Evidence archive checksum mismatch: {archive_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("archive") != archive_path.name
        or int(manifest.get("archive_bytes", -1)) != archive_path.stat().st_size
        or manifest.get("archive_sha256") != digest
    ):
        raise ValueError(f"Evidence archive metadata mismatch: {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, list) or int(manifest.get("file_count", -1)) != len(files):
        raise ValueError(f"Invalid evidence content manifest: {manifest_path}")
    if int(manifest.get("content_bytes", -1)) != sum(int(item["bytes"]) for item in files):
        raise ValueError(f"Evidence content byte count mismatch: {manifest_path}")
    _verify_payload(archive_path, files)
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
        if not derived_path.is_file():
            raise ValueError(f"Derived manifest output is not a regular file: {derived_path}")
        existing = json.loads(derived_path.read_text(encoding="utf-8"))
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
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(derived_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return derived_path, payload


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def archive_evidence(output_dir: Path) -> tuple[Path, str]:
    """Create and fully verify an immutable archive/checksum/manifest bundle."""
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    archive_path = output_dir.with_suffix(".tar.gz")
    checksum_path, manifest_path = evidence_sidecars(archive_path)
    records = _archive_files(output_dir)
    temporary = archive_path.with_name(f".{archive_path.name}.part")
    temporary.unlink(missing_ok=True)
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for record in records:
                archive.add(
                    record["source"],
                    arcname=record["path"],
                    recursive=False,
                )
        public_records = [
            {key: record[key] for key in ("path", "bytes", "sha256")} for record in records
        ]
        _verify_payload(temporary, public_records)
        os.replace(temporary, archive_path)
    finally:
        temporary.unlink(missing_ok=True)

    digest = sha256(archive_path)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archive": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": digest,
        "file_count": len(public_records),
        "content_bytes": sum(int(record["bytes"]) for record in public_records),
        "files": public_records,
    }
    _atomic_write(checksum_path, f"{digest}  {archive_path.name}\n")
    _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    verify_evidence_archive(archive_path)
    return archive_path, digest

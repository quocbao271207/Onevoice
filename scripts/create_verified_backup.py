"""Create sealed local backups for data, models, and reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
MANIFEST_NAME = "backup_manifest.json"
MANIFEST_CHECKSUM_NAME = f"{MANIFEST_NAME}.sha256"
CHECKSUMS_NAME = "SHA256SUMS.txt"
EXPECTED_ARCHIVES = {
    "data.tar": "data",
    "models.tar": "models",
    "reports.tar": "data/reports",
}
EXPECTED_OUTPUT_NAMES = {
    *EXPECTED_ARCHIVES,
    MANIFEST_NAME,
    MANIFEST_CHECKSUM_NAME,
    CHECKSUMS_NAME,
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RELEASE_COMPARISON_MEMBER = "data/reports/model_bakeoff/comparison.json"
RELEASE_EVIDENCE_FIELDS = {
    "comparison_path",
    "comparison_bytes",
    "comparison_sha256",
    "decision",
    "promotion_allowed",
    "scope",
}
MAX_RELEASE_COMPARISON_BYTES = 50_000_000


def _sha256_stream(handle: BinaryIO, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    if path.is_symlink():
        raise ValueError(f"Refusing to hash a symlink: {path}")
    with path.open("rb") as handle:
        return _sha256_stream(handle, chunk_size)


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return bool(checker and checker())


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _reject_link(path: Path, label: str) -> None:
    if path.is_symlink() or _is_junction(path):
        raise ValueError(f"Refusing {label} symlink or junction: {path}")


def _reject_link_ancestors(path: Path, label: str) -> None:
    """Reject an existing symlink or junction anywhere in an output path."""
    absolute = _absolute(path)
    for candidate in (absolute, *absolute.parents):
        _reject_link(candidate, label)


def iter_files(source: Path, excluded: Iterable[Path]) -> list[Path]:
    source = _absolute(source)
    _reject_link(source, "source")
    if not source.is_dir():
        raise FileNotFoundError(source)
    excluded_absolute = [_absolute(path) for path in excluded]
    files: list[Path] = []
    for path in source.rglob("*"):
        path = _absolute(path)
        if any(_is_within(path, item) for item in excluded_absolute):
            continue
        _reject_link(path, "source entry")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Unsupported source entry type: {path}")
        files.append(path)
    return sorted(files)


def _source_records(files: Iterable[Path]) -> list[dict[str, object]]:
    root = _absolute(ROOT)
    records = []
    for path in files:
        absolute = _absolute(path)
        try:
            member_path = absolute.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"Backup source is outside project root: {path}") from error
        size = absolute.stat().st_size
        digest = sha256(absolute)
        if absolute.stat().st_size != size:
            raise RuntimeError(f"Source changed while hashing: {absolute}")
        records.append({"path": member_path, "bytes": size, "sha256": digest})
    return records


def _validated_sha256(value: object, label: str) -> str:
    text = str(value)
    if not SHA256_PATTERN.fullmatch(text):
        raise ValueError(f"Invalid SHA-256 for {label}: {value!r}")
    return text


def _non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid integer for {label}: {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid integer for {label}: {value!r}") from error
    if parsed < 0 or str(parsed) != str(value):
        raise ValueError(f"Invalid integer for {label}: {value!r}")
    return parsed


def _validate_member_path(value: object, expected_source: str) -> str:
    text = str(value)
    member_path = PurePosixPath(text)
    source_prefix = PurePosixPath(expected_source)
    if (
        not text
        or "\\" in text
        or member_path.is_absolute()
        or ".." in member_path.parts
        or member_path.as_posix() != text
        or source_prefix not in member_path.parents
    ):
        raise RuntimeError(f"Unsafe or misplaced member: {text}")
    return text


def _validated_member_records(
    value: object, expected_source: str, archive_name: str
) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Member manifest for {archive_name} must be a non-empty list")
    records: list[dict[str, object]] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise ValueError(f"Invalid member record {index} for {archive_name}")
        name = _validate_member_path(item.get("path"), expected_source)
        if name in names:
            raise RuntimeError(f"Duplicate member path in manifest for {archive_name}: {name}")
        names.add(name)
        records.append(
            {
                "path": name,
                "bytes": _non_negative_int(item.get("bytes"), f"{archive_name}:{name}:bytes"),
                "sha256": _validated_sha256(
                    item.get("sha256"), f"{archive_name}:{name}"
                ),
            }
        )
    if [str(item["path"]) for item in records] != sorted(names):
        raise ValueError(f"Member manifest for {archive_name} must be sorted by path")
    return records


def _release_comparison_payload(payload: bytes) -> dict[str, object]:
    if len(payload) < 2 or len(payload) > MAX_RELEASE_COMPARISON_BYTES:
        raise ValueError("Release comparison size is outside the valid range")
    try:
        comparison = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Release comparison is not valid UTF-8 JSON") from error
    if not isinstance(comparison, dict):
        raise ValueError("Release comparison root must be an object")
    return comparison


def _validate_release_decision(
    comparison: dict[str, object],
    *,
    decision: object,
    promotion_allowed: object,
    scope: object,
) -> None:
    if comparison.get("status") != "complete":
        raise ValueError("Release comparison is not terminal")
    if not isinstance(promotion_allowed, bool):
        raise ValueError("Release promotion flag must be boolean")
    expected_decision = "promote" if promotion_allowed else "reject"
    if decision != expected_decision:
        raise ValueError("Release decision disagrees with promotion flag")
    if comparison.get("promotion_allowed") is not promotion_allowed:
        raise ValueError("Release evidence promotion flag mismatch")
    if scope not in {"research", "production"} or comparison.get("scope") != scope:
        raise ValueError("Release evidence scope mismatch")


def build_release_evidence(
    comparison_path: Path,
    *,
    decision: str,
    scope: str,
) -> dict[str, object]:
    """Bind a terminal bake-off decision to the comparison archived in reports.tar."""
    comparison_path = _absolute(comparison_path)
    _reject_link(comparison_path, "release comparison")
    if not comparison_path.is_file():
        raise FileNotFoundError(comparison_path)
    try:
        member = comparison_path.relative_to(_absolute(ROOT)).as_posix()
    except ValueError as error:
        raise ValueError("Release comparison is outside project root") from error
    if member != RELEASE_COMPARISON_MEMBER:
        raise ValueError(
            f"Release comparison must be {RELEASE_COMPARISON_MEMBER}: {member}"
        )
    size = comparison_path.stat().st_size
    if size < 2 or size > MAX_RELEASE_COMPARISON_BYTES:
        raise ValueError("Release comparison size is outside the valid range")
    with comparison_path.open("rb") as handle:
        payload = handle.read(MAX_RELEASE_COMPARISON_BYTES + 1)
    if len(payload) != size:
        raise RuntimeError("Release comparison changed while reading")
    comparison = _release_comparison_payload(payload)
    promotion_allowed = comparison.get("promotion_allowed")
    _validate_release_decision(
        comparison,
        decision=decision,
        promotion_allowed=promotion_allowed,
        scope=scope,
    )
    return {
        "comparison_path": member,
        "comparison_bytes": len(payload),
        "comparison_sha256": hashlib.sha256(payload).hexdigest(),
        "decision": decision,
        "promotion_allowed": promotion_allowed,
        "scope": scope,
    }


def _verify_release_evidence(
    value: object,
    report_records: list[dict[str, object]],
    report_archive: Path,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != RELEASE_EVIDENCE_FIELDS:
        raise ValueError("Backup release_evidence schema is invalid")
    member = str(value.get("comparison_path") or "")
    if member != RELEASE_COMPARISON_MEMBER:
        raise ValueError("Backup release comparison path is invalid")
    expected = next(
        (record for record in report_records if record["path"] == member),
        None,
    )
    if expected is None:
        raise ValueError("Backup reports archive lacks the release comparison")
    declared_bytes = _non_negative_int(
        value.get("comparison_bytes"),
        "release_evidence:comparison_bytes",
    )
    declared_sha256 = _validated_sha256(
        value.get("comparison_sha256"),
        "release_evidence:comparison_sha256",
    )
    if declared_bytes != expected["bytes"] or declared_sha256 != expected["sha256"]:
        raise RuntimeError("Backup release comparison binding mismatch")
    with tarfile.open(report_archive, mode="r") as tar:
        try:
            archived = tar.getmember(member)
        except KeyError as error:
            raise RuntimeError("Backup release comparison member is missing") from error
        extracted = tar.extractfile(archived)
        if extracted is None:
            raise RuntimeError("Backup release comparison member is unreadable")
        with extracted:
            payload = extracted.read(MAX_RELEASE_COMPARISON_BYTES + 1)
    comparison = _release_comparison_payload(payload)
    _validate_release_decision(
        comparison,
        decision=value.get("decision"),
        promotion_allowed=value.get("promotion_allowed"),
        scope=value.get("scope"),
    )
    return dict(value)


def _verify_tar(
    path: Path, expected_source: str, records: list[dict[str, object]]
) -> None:
    _reject_link(path, "archive")
    expected_names = [str(item["path"]) for item in records]
    with tarfile.open(path, mode="r") as tar:
        members = tar.getmembers()
        for member in members:
            if not member.isfile():
                raise RuntimeError(f"Non-regular tar member in {path}: {member.name}")
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError(f"Duplicate member path: {path}")
        for name in names:
            _validate_member_path(name, expected_source)
        if names != expected_names:
            missing = sorted(set(expected_names) - set(names))
            extra = sorted(set(names) - set(expected_names))
            raise RuntimeError(
                f"Tar member set/order mismatch for {path}; missing={missing[:5]}, extra={extra[:5]}"
            )
        for member, expected in zip(members, records, strict=True):
            expected_size = int(expected["bytes"])
            if member.size != expected_size:
                raise RuntimeError(
                    f"Member byte-size mismatch in {path}: {member.name} "
                    f"({member.size} != {expected_size})"
                )
            extracted = tar.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Cannot read tar member in {path}: {member.name}")
            with extracted:
                actual_digest = _sha256_stream(extracted)
            if actual_digest != expected["sha256"]:
                raise RuntimeError(f"Member checksum mismatch in {path}: {member.name}")


def _prepare_destination(destination: Path) -> Path:
    _reject_link(destination, "destination")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    _reject_link(temporary, "temporary destination")
    if temporary.exists():
        if not temporary.is_file():
            raise ValueError(f"Temporary destination is not a regular file: {temporary}")
        temporary.unlink()
    return temporary


def archive(source: Path, destination: Path, excluded: Iterable[Path] = ()) -> dict[str, object]:
    files = iter_files(source, excluded)
    if not files:
        raise ValueError(f"Refusing to create an empty backup archive for {source}")
    records = _source_records(files)
    source_bytes = sum(int(item["bytes"]) for item in records)
    source_name = _absolute(source).relative_to(_absolute(ROOT)).as_posix()
    temporary = _prepare_destination(destination)
    try:
        with tarfile.open(temporary, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for path, record in zip(files, records, strict=True):
                tar.add(path, arcname=str(record["path"]), recursive=False)
        _verify_tar(temporary, source_name, records)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists() and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        raise
    return {
        "archive": destination.name,
        "source": source_name,
        "files": len(records),
        "source_bytes": source_bytes,
        "archive_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "members": records,
        "verified": True,
    }


def _atomic_write(path: Path, content: str) -> None:
    _reject_link(path, "metadata destination")
    temporary = path.with_name(f".{path.name}.part")
    _reject_link(temporary, "temporary metadata destination")
    if temporary.exists():
        if not temporary.is_file():
            raise ValueError(f"Temporary metadata destination is not a regular file: {temporary}")
        temporary.unlink()
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists() and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        raise


def write_metadata(output_dir: Path, manifest: dict[str, object]) -> None:
    manifest_path = output_dir / MANIFEST_NAME
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    _atomic_write(manifest_path, manifest_text)
    _atomic_write(
        output_dir / MANIFEST_CHECKSUM_NAME,
        f"{sha256(manifest_path)}  {MANIFEST_NAME}\n",
    )
    archives = manifest.get("archives")
    if not isinstance(archives, list):
        raise ValueError("Backup manifest archives must be a list")
    by_name = {
        str(item.get("archive")): item for item in archives if isinstance(item, dict)
    }
    _atomic_write(
        output_dir / CHECKSUMS_NAME,
        "".join(f"{by_name[name]['sha256']}  {name}\n" for name in EXPECTED_ARCHIVES),
    )


def _load_manifest(output_dir: Path) -> dict[str, object]:
    manifest_path = output_dir / MANIFEST_NAME
    sidecar_path = output_dir / MANIFEST_CHECKSUM_NAME
    checksum_path = output_dir / CHECKSUMS_NAME
    for path in (manifest_path, sidecar_path, checksum_path):
        _reject_link(path, "backup metadata")
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_sidecar = f"{sha256(manifest_path)}  {MANIFEST_NAME}\n"
    if sidecar_path.read_text(encoding="utf-8") != expected_sidecar:
        raise RuntimeError(f"{MANIFEST_CHECKSUM_NAME} does not match {MANIFEST_NAME}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Backup manifest schema_version must be {SCHEMA_VERSION}")
    return manifest


def verify(output_dir: Path) -> dict[str, object]:
    output_dir = _absolute(output_dir)
    _reject_link_ancestors(output_dir.parent, "backup directory ancestor")
    _reject_link(output_dir, "backup directory")
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    entries = list(output_dir.iterdir())
    for entry in entries:
        _reject_link(entry, "backup entry")
    actual_names = {entry.name for entry in entries}
    if actual_names != EXPECTED_OUTPUT_NAMES:
        missing = sorted(EXPECTED_OUTPUT_NAMES - actual_names)
        extra = sorted(actual_names - EXPECTED_OUTPUT_NAMES)
        raise RuntimeError(f"Backup file set mismatch; missing={missing}, extra={extra}")

    manifest = _load_manifest(output_dir)
    archives = manifest.get("archives")
    if not isinstance(archives, list):
        raise ValueError("Backup manifest archives must be a list")
    if any(not isinstance(item, dict) for item in archives):
        raise ValueError("Every backup archive record must be an object")
    by_name = {str(item.get("archive")): item for item in archives}
    if len(by_name) != len(archives) or set(by_name) != set(EXPECTED_ARCHIVES):
        raise ValueError("Backup manifest must contain data.tar, models.tar, and reports.tar exactly once")

    expected_checksum_file = "".join(
        f"{_validated_sha256(by_name[name].get('sha256'), name)}  {name}\n"
        for name in EXPECTED_ARCHIVES
    )
    checksum_path = output_dir / CHECKSUMS_NAME
    if checksum_path.read_text(encoding="utf-8") != expected_checksum_file:
        raise RuntimeError(f"{CHECKSUMS_NAME} does not match {MANIFEST_NAME}")

    validated_records: dict[str, list[dict[str, object]]] = {}
    for name, expected_source in EXPECTED_ARCHIVES.items():
        item = by_name[name]
        declared_source = str(item.get("source") or "").replace("\\", "/")
        if declared_source != expected_source:
            raise ValueError(f"Unexpected source for {name}: {item.get('source')!r}")
        records = _validated_member_records(item.get("members"), expected_source, name)
        validated_records[name] = records
        declared_files = _non_negative_int(item.get("files"), f"{name}:files")
        declared_source_bytes = _non_negative_int(
            item.get("source_bytes"), f"{name}:source_bytes"
        )
        if declared_files != len(records):
            raise RuntimeError(f"File-count mismatch in manifest: {name}")
        if declared_source_bytes != sum(int(record["bytes"]) for record in records):
            raise RuntimeError(f"Source byte-count mismatch in manifest: {name}")
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        archive_bytes = _non_negative_int(item.get("archive_bytes"), f"{name}:archive_bytes")
        if path.stat().st_size != archive_bytes:
            raise RuntimeError(f"Archive byte-size mismatch: {path}")
        actual = sha256(path)
        expected_digest = _validated_sha256(item.get("sha256"), name)
        if actual != expected_digest:
            raise RuntimeError(f"Checksum mismatch: {path}")
        _verify_tar(path, expected_source, records)
    release_evidence = manifest.get("release_evidence")
    if release_evidence is not None:
        _verify_release_evidence(
            release_evidence,
            validated_records["reports.tar"],
            output_dir / "reports.tar",
        )
    verified = dict(manifest)
    verified["verification"] = "pass"
    return verified


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _summary(output_dir: Path, manifest: dict[str, object]) -> dict[str, object]:
    archives = manifest["archives"]
    assert isinstance(archives, list)
    summary: dict[str, object] = {
        "output_dir": str(output_dir),
        "schema_version": manifest["schema_version"],
        "verification": manifest.get("verification"),
        "archives": [
            {
                key: item[key]
                for key in (
                    "archive",
                    "source",
                    "files",
                    "source_bytes",
                    "archive_bytes",
                    "sha256",
                )
            }
            for item in archives
            if isinstance(item, dict)
        ],
    }
    if "release_evidence" in manifest:
        summary["release_evidence"] = manifest["release_evidence"]
    return summary


def create_backup(
    output_dir: Path,
    *,
    release_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    output_dir = _absolute(output_dir)
    _reject_link_ancestors(output_dir.parent, "backup output ancestor")
    _reject_link(output_dir, "backup output")
    source_roots = tuple(
        path.resolve()
        for path in (ROOT / "data", ROOT / "models", ROOT / "data" / "reports")
    )
    if any(_is_within(output_dir, source) for source in source_roots):
        raise ValueError("Backup output must be outside data, models, and reports")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _reject_link_ancestors(output_dir.parent, "backup output ancestor")
    staging = output_dir.with_name(f".{output_dir.name}.part-{uuid.uuid4().hex}")
    staging.mkdir(exist_ok=False)
    try:
        reports = ROOT / "data" / "reports"
        archives = [
            archive(ROOT / "data", staging / "data.tar", excluded=[reports]),
            archive(ROOT / "models", staging / "models.tar"),
            archive(reports, staging / "reports.tar"),
        ]
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "project_root": str(ROOT),
            "git_head": _git_head(),
            "format": "uncompressed POSIX tar with per-member SHA-256",
            "archives": archives,
        }
        if release_evidence is not None:
            manifest["release_evidence"] = dict(release_evidence)
        write_metadata(staging, manifest)
        verified = verify(staging)
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise
    return verified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-only", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        verified = verify(args.verify_only)
        print(json.dumps(_summary(_absolute(args.verify_only), verified), indent=2))
        return 0

    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    requested_output = args.output_dir or ROOT / ".backups" / f"onevoice-{stamp}"
    output_dir = requested_output.resolve(strict=False)
    verified = create_backup(requested_output)
    print(json.dumps(_summary(output_dir, verified), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

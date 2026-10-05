"""Create local tar backups for data, models, and reports with SHA-256 verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ARCHIVES = {
    "data.tar": "data",
    "models.tar": "models",
    "reports.tar": "data/reports",
}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(source: Path, excluded: Iterable[Path]) -> list[Path]:
    excluded_resolved = [path.resolve() for path in excluded]
    files = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if any(resolved == item or item in resolved.parents for item in excluded_resolved):
            continue
        files.append(path)
    return sorted(files)


def archive(source: Path, destination: Path, excluded: Iterable[Path] = ()) -> dict[str, object]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = iter_files(source, excluded)
    if not files:
        raise ValueError(f"Refusing to create an empty backup archive for {source}")
    source_bytes = sum(path.stat().st_size for path in files)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, mode="w") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(ROOT), recursive=False)
    with tarfile.open(destination, mode="r") as tar:
        file_members = [member for member in tar.getmembers() if member.isfile()]
        archived_files = len(file_members)
        archived_bytes = sum(member.size for member in file_members)
    if archived_files != len(files):
        raise RuntimeError(f"Archive file-count mismatch for {destination}: {archived_files} != {len(files)}")
    if archived_bytes != source_bytes:
        raise RuntimeError(
            f"Archive byte-count mismatch for {destination}: {archived_bytes} != {source_bytes}"
        )
    return {
        "archive": destination.name,
        "source": source.relative_to(ROOT).as_posix(),
        "files": len(files),
        "source_bytes": source_bytes,
        "archive_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "verified": True,
    }


def verify(output_dir: Path) -> dict[str, object]:
    manifest_path = output_dir / "backup_manifest.json"
    checksum_path = output_dir / "SHA256SUMS.txt"
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError("Backup manifest and SHA256SUMS.txt are both required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archives = manifest.get("archives")
    if not isinstance(archives, list):
        raise ValueError("Backup manifest archives must be a list")
    by_name = {str(item.get("archive")): item for item in archives if isinstance(item, dict)}
    if len(by_name) != len(archives) or set(by_name) != set(EXPECTED_ARCHIVES):
        raise ValueError("Backup manifest must contain data.tar, models.tar, and reports.tar exactly once")
    expected_checksum_file = "".join(
        f"{by_name[name]['sha256']}  {name}\n" for name in EXPECTED_ARCHIVES
    )
    if checksum_path.read_text(encoding="utf-8") != expected_checksum_file:
        raise RuntimeError("SHA256SUMS.txt does not match backup_manifest.json")

    for name, expected_source in EXPECTED_ARCHIVES.items():
        item = by_name[name]
        declared_source = str(item.get("source") or "").replace("\\", "/")
        if declared_source != expected_source:
            raise ValueError(f"Unexpected source for {name}: {item.get('source')!r}")
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(item["archive_bytes"]):
            raise RuntimeError(f"Archive byte-size mismatch: {path}")
        actual = sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {path}")
        with tarfile.open(path, mode="r") as tar:
            members = tar.getmembers()
            file_members = [member for member in members if member.isfile()]
            if len(file_members) != int(item["files"]):
                raise RuntimeError(f"File-count mismatch: {path}")
            if sum(member.size for member in file_members) != int(item["source_bytes"]):
                raise RuntimeError(f"Source byte-count mismatch: {path}")
            names = [member.name for member in file_members]
            if len(names) != len(set(names)):
                raise RuntimeError(f"Duplicate member path: {path}")
            source_prefix = PurePosixPath(expected_source)
            for member_name in names:
                member_path = PurePosixPath(member_name)
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or source_prefix not in member_path.parents
                ):
                    raise RuntimeError(f"Unsafe or misplaced member in {path}: {member_name}")
    manifest["verification"] = "pass"
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-only", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify(args.verify_only.resolve()), indent=2))
        return 0

    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output_dir = (args.output_dir or ROOT / ".backups" / f"onevoice-{stamp}").resolve()
    sources = (ROOT / "data", ROOT / "models", ROOT / "data" / "reports")
    if any(output_dir == source.resolve() or source.resolve() in output_dir.parents for source in sources):
        raise ValueError("Backup output must be outside data, models, and reports")
    output_dir.mkdir(parents=True, exist_ok=False)
    reports = ROOT / "data" / "reports"
    archives = [
        archive(ROOT / "data", output_dir / "data.tar", excluded=[reports]),
        archive(ROOT / "models", output_dir / "models.tar"),
        archive(reports, output_dir / "reports.tar"),
    ]
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "git_head": git_head,
        "format": "uncompressed POSIX tar",
        "archives": archives,
    }
    (output_dir / "backup_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(
            f"{next(item for item in archives if item['archive'] == name)['sha256']}  {name}\n"
            for name in EXPECTED_ARCHIVES
        ),
        encoding="utf-8",
    )
    verified = verify(output_dir)
    print(json.dumps({"output_dir": str(output_dir), **verified}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

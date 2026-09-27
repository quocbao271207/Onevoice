"""Create local tar backups for data, models, and reports with SHA-256 verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


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
    files = iter_files(source, excluded)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, mode="w") as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(ROOT), recursive=False)
    with tarfile.open(destination, mode="r") as tar:
        archived_files = sum(member.isfile() for member in tar.getmembers())
    if archived_files != len(files):
        raise RuntimeError(f"Archive file-count mismatch for {destination}: {archived_files} != {len(files)}")
    return {
        "archive": destination.name,
        "source": str(source.relative_to(ROOT)),
        "files": len(files),
        "source_bytes": sum(path.stat().st_size for path in files),
        "archive_bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "verified": True,
    }


def verify(output_dir: Path) -> dict[str, object]:
    manifest_path = output_dir / "backup_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["archives"]:
        path = output_dir / item["archive"]
        actual = sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {path}")
        with tarfile.open(path, mode="r") as tar:
            if sum(member.isfile() for member in tar.getmembers()) != item["files"]:
                raise RuntimeError(f"File-count mismatch: {path}")
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
        "".join(f"{item['sha256']}  {item['archive']}\n" for item in archives),
        encoding="utf-8",
    )
    verified = verify(output_dir)
    print(json.dumps({"output_dir": str(output_dir), **verified}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

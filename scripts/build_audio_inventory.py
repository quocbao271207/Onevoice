"""Build a deterministic SHA-256 inventory for every GPU-bound ASR FLAC."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "processed" / "manifests"
DEFAULT_OUTPUT = ROOT / "data" / "reports" / "audio_inventory.jsonl"
LOCAL_MANIFESTS = (
    MANIFEST_DIR / "asr--train-local.jsonl",
    MANIFEST_DIR / "asr--validation-local.jsonl",
    MANIFEST_DIR / "asr--test-local.jsonl",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_entries() -> list[dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for manifest in LOCAL_MANIFESTS:
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                relative = Path(row["audio_path"])
                if relative.is_absolute():
                    raise ValueError(f"{manifest}:{line_number}: absolute audio path")
                key = relative.as_posix()
                if key in entries:
                    raise ValueError(f"Duplicate audio path across manifests: {key}")
                path = ROOT / relative
                if not path.is_file():
                    raise FileNotFoundError(path)
                entries[key] = {
                    "audio_path": key,
                    "id": row["id"],
                    "source": row["source"],
                    "role": row["role"],
                    "bytes": path.stat().st_size,
                    "_path": path,
                }
    return [entries[key] for key in sorted(entries)]


def build_inventory(output: Path, workers: int) -> dict[str, Any]:
    entries = load_entries()

    def hash_entry(entry: dict[str, Any]) -> dict[str, Any]:
        result = {key: value for key, value in entry.items() if key != "_path"}
        result["sha256"] = sha256(entry["_path"])
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashed = list(pool.map(hash_entry, entries))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in hashed:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    hash_counts = Counter(row["sha256"] for row in hashed)
    duplicate_hash_groups = sum(count > 1 for count in hash_counts.values())
    if duplicate_hash_groups:
        raise ValueError(
            f"Exact-audio dedup gate failed: {duplicate_hash_groups} duplicate SHA-256 groups remain"
        )
    summary = {
        "status": "pass",
        "files": len(hashed),
        "unique_audio_hashes": len(hash_counts),
        "duplicate_audio_hash_groups": 0,
        "bytes": sum(row["bytes"] for row in hashed),
        "inventory_sha256": sha256(output),
        "role_counts": dict(sorted(Counter(row["role"] for row in hashed).items())),
        "source_counts": dict(sorted(Counter(row["source"] for row in hashed).items())),
    }
    summary_path = output.with_name(f"{output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    print(json.dumps(build_inventory(args.output, args.workers), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

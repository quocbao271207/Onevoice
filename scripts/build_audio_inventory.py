"""Build a deterministic SHA-256 inventory for every GPU-bound ASR FLAC."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.durable_file import write_durable_bytes  # noqa: E402
from src.pipeline.durable_json import write_durable_json  # noqa: E402
from src.pipeline.evidence_paths import resolve_regular_file_under  # noqa: E402
from src.pipeline.stable_jsonl import read_stable_jsonl_mappings  # noqa: E402
from src.utils.bounded_file import sha256_stable_regular_file  # noqa: E402


MANIFEST_DIR = ROOT / "data" / "processed" / "manifests"
DEFAULT_OUTPUT = ROOT / "data" / "reports" / "audio_inventory.jsonl"
MAX_MANIFEST_BYTES = 250_000_000
MAX_MANIFEST_LINE_BYTES = 2_000_000
MAX_MANIFEST_ROWS = 100_000
MAX_AUDIO_BYTES = 128 * 1024 * 1024
MAX_INVENTORY_BYTES = 250_000_000
MAX_INVENTORY_SUMMARY_BYTES = 1_000_000
MAX_WORKERS = 32
LOCAL_MANIFESTS = (
    MANIFEST_DIR / "asr--train-local.jsonl",
    MANIFEST_DIR / "asr--validation-local.jsonl",
    MANIFEST_DIR / "asr--test-local.jsonl",
)
MANIFEST_ROLES = {
    "asr--train-local.jsonl": "train",
    "asr--validation-local.jsonl": "validation",
    "asr--test-local.jsonl": "test",
}


def load_entries() -> list[dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    audio_root = ROOT / "data" / "processed" / "audio_16k"
    for manifest in LOCAL_MANIFESTS:
        expected_role = MANIFEST_ROLES.get(manifest.name)
        if expected_role is None:
            raise ValueError(f"Unknown local manifest role: {manifest.name}")
        document = read_stable_jsonl_mappings(
            manifest,
            maximum_bytes=MAX_MANIFEST_BYTES,
            maximum_line_bytes=MAX_MANIFEST_LINE_BYTES,
            maximum_rows=MAX_MANIFEST_ROWS,
            label=f"Local ASR {expected_role} manifest",
        )
        for line_number, row in enumerate(document.rows, 1):
            record_id = row.get("id")
            source = row.get("source")
            role = row.get("role")
            raw_path = row.get("audio_path")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{manifest}:{line_number}: invalid record ID")
            if record_id in seen_ids:
                raise ValueError(f"Duplicate record ID across manifests: {record_id}")
            seen_ids.add(record_id)
            if not isinstance(source, str) or not source:
                raise ValueError(f"{manifest}:{line_number}: invalid source")
            if role != expected_role:
                raise ValueError(
                    f"{manifest}:{line_number}: role {role!r} does not match {expected_role!r}"
                )
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"{manifest}:{line_number}: invalid audio path")
            relative = Path(raw_path)
            if relative.is_absolute():
                raise ValueError(f"{manifest}:{line_number}: absolute audio path")
            key = relative.as_posix()
            if key in entries:
                raise ValueError(f"Duplicate audio path across manifests: {key}")
            path = resolve_regular_file_under(
                relative,
                project_root=ROOT,
                allowed_root=audio_root,
                label="Inventory source audio",
                maximum_bytes=MAX_AUDIO_BYTES,
            )
            canonical_key = path.relative_to(ROOT).as_posix()
            if key != canonical_key:
                raise ValueError(
                    f"{manifest}:{line_number}: audio path is not canonical"
                )
            entries[key] = {
                "audio_path": key,
                "id": record_id,
                "source": source,
                "role": role,
                "_path": path,
            }
            if len(entries) > MAX_MANIFEST_ROWS:
                raise ValueError(
                    f"Audio inventory exceeds {MAX_MANIFEST_ROWS} unique files"
                )
    return [entries[key] for key in sorted(entries)]


def build_inventory(output: Path, workers: int) -> dict[str, Any]:
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be an integer within 1..{MAX_WORKERS}")
    entries = load_entries()

    def hash_entry(entry: dict[str, Any]) -> dict[str, Any]:
        result = {key: value for key, value in entry.items() if key != "_path"}
        digest, size = sha256_stable_regular_file(
            entry["_path"],
            maximum_bytes=MAX_AUDIO_BYTES,
            label="Inventory source audio",
        )
        result["bytes"] = size
        result["sha256"] = digest
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashed = list(pool.map(hash_entry, entries))

    hash_counts = Counter(row["sha256"] for row in hashed)
    duplicate_hash_groups = sum(count > 1 for count in hash_counts.values())
    if duplicate_hash_groups:
        raise ValueError(
            f"Exact-audio dedup gate failed: {duplicate_hash_groups} duplicate SHA-256 groups remain"
        )
    try:
        inventory_payload = "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
            for row in hashed
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise ValueError("Audio inventory is not strict JSONL") from None
    write_durable_bytes(
        output,
        inventory_payload,
        maximum_bytes=MAX_INVENTORY_BYTES,
        label="Audio inventory",
    )
    inventory_digest = hashlib.sha256(inventory_payload).hexdigest()
    persisted = read_stable_jsonl_mappings(
        output,
        maximum_bytes=MAX_INVENTORY_BYTES,
        maximum_line_bytes=MAX_MANIFEST_LINE_BYTES,
        maximum_rows=MAX_MANIFEST_ROWS,
        label="Persisted audio inventory",
        expected_sha256=inventory_digest,
    )
    if persisted.rows != hashed:
        raise RuntimeError("Persisted audio inventory does not match its source rows")
    summary = {
        "status": "pass",
        "files": len(hashed),
        "unique_audio_hashes": len(hash_counts),
        "duplicate_audio_hash_groups": 0,
        "bytes": sum(row["bytes"] for row in hashed),
        "inventory_sha256": persisted.sha256,
        "role_counts": dict(sorted(Counter(row["role"] for row in hashed).items())),
        "source_counts": dict(sorted(Counter(row["source"] for row in hashed).items())),
    }
    summary_path = output.with_name(f"{output.stem}_summary.json")
    write_durable_json(
        summary_path,
        summary,
        maximum_bytes=MAX_INVENTORY_SUMMARY_BYTES,
        label="Audio inventory summary",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be within 1..{MAX_WORKERS}")
    print(json.dumps(build_inventory(args.output, args.workers), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Download audited ASR audio with resume support and emit local manifests."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import yaml

from audit_datasets import api_json, normalize_asr_row, split_sizes


ROOT = Path(__file__).resolve().parents[1]


def destination_path(record: dict[str, Any], audio_root: Path) -> Path:
    return audio_root / record["source"] / record["source_split"] / f"{record['id']}.wav"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def download_one(record: dict[str, Any], audio_root: Path, retries: int = 4) -> dict[str, Any]:
    destination = destination_path(record, audio_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 44:
        result = dict(record)
        result["audio_path"] = str(destination.resolve())
        return result
    url = record.get("audio_url")
    if not url:
        raise ValueError(f"{record['id']}: no audio_url")
    temp = destination.with_suffix(".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "onevoice-audio-materializer/1.0"})
            with urlopen(request, timeout=120) as response, temp.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if temp.stat().st_size <= 44:
                raise IOError("downloaded audio is empty")
            os.replace(temp, destination)
            result = dict(record)
            result["audio_path"] = str(destination.resolve())
            return result
        except Exception as exc:
            last_error = exc
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"{record['id']}: download failed: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data" / "processed" / "manifests")
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "data" / "processed" / "audio")
    parser.add_argument("--roles", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Per role; 0 means all.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "datasets.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    desired: dict[tuple[str, str], dict[str, tuple[str, dict[str, Any]]]] = {}
    role_totals: Counter[str] = Counter()
    for role in args.roles:
        records = read_jsonl(args.manifest_dir / f"asr--{role}.jsonl")
        if args.limit:
            records = records[: args.limit]
        role_totals[role] = len(records)
        for record in records:
            desired.setdefault((record["source"], record["source_split"]), {})[record["id"]] = (role, record)
    print(f"[DOWNLOAD] desired by role: {dict(role_totals)}", flush=True)

    completed: dict[str, list[dict[str, Any]]] = {role: [] for role in args.roles}
    # Resume without refreshing signed URLs for assets already present. This is
    # important after a manifest-only policy change or an interrupted long run.
    resumed = 0
    for key, wanted in list(desired.items()):
        for record_id, (role, record) in list(wanted.items()):
            destination = destination_path(record, args.audio_dir)
            if destination.is_file() and destination.stat().st_size > 44:
                item = dict(record)
                item["audio_path"] = str(destination.resolve())
                completed[role].append(item)
                del wanted[record_id]
                resumed += 1
        if not wanted:
            del desired[key]
    print(f"[RESUME] local assets reused={resumed}; source/splits still requiring API={len(desired)}", flush=True)
    errors: list[str] = []
    for (source, split), wanted in sorted(desired.items()):
        spec = config["datasets"][source]
        repo_id = spec["repo_id"]
        dataset_config = spec.get("config", "default")
        sizes = [row for row in split_sizes(repo_id) if row.get("config") == dataset_config and row.get("split") == split]
        if not sizes:
            errors.append(f"{source}/{split}: split size unavailable")
            continue
        total_rows = int(sizes[0]["num_rows"])
        found: set[str] = set()
        print(f"[SOURCE] {source}/{split}: need={len(wanted)} scan={total_rows}", flush=True)
        for offset in range(0, total_rows, 100):
            payload = api_json(
                "rows",
                {"dataset": repo_id, "config": dataset_config, "split": split, "offset": offset, "length": 100},
            )
            page_records = []
            for item in payload.get("rows", []):
                fresh = normalize_asr_row(source, spec, split, item.get("row", {}))
                if fresh["id"] not in wanted:
                    continue
                role, original = wanted[fresh["id"]]
                record = dict(original)
                record["audio_url"] = fresh["audio_url"]
                page_records.append((role, record))
                found.add(record["id"])
            if page_records:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = {
                        pool.submit(download_one, record, args.audio_dir): (role, record)
                        for role, record in page_records
                    }
                    for future in as_completed(futures):
                        role, record = futures[future]
                        try:
                            completed[role].append(future.result())
                        except Exception as exc:
                            errors.append(str(exc))
            if (offset // 100 + 1) % 25 == 0 or offset + 100 >= total_rows:
                print(f"  scanned={min(offset + 100, total_rows)}/{total_rows}; found={len(found)}/{len(wanted)}", flush=True)
        missing = set(wanted) - found
        if missing:
            errors.append(f"{source}/{split}: {len(missing)} manifest IDs not found in refreshed rows")

    for role in args.roles:
        completed[role].sort(key=lambda row: row["id"])
        output_path = args.manifest_dir / f"asr--{role}-local.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for record in completed[role]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[DONE] {output_path}: {len(completed[role])}/{role_totals[role]}", flush=True)
    if errors:
        error_path = args.manifest_dir / "asr--download-errors.json"
        error_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"Materialization failed with {len(errors)} errors; see {error_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

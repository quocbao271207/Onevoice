"""Download audited ASR audio with resume support and emit local manifests."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import yaml

try:
    from scripts.audit_datasets import (
        api_json,
        durable_asr_record,
        normalize_asr_row,
        split_sizes,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from audit_datasets import api_json, durable_asr_record, normalize_asr_row, split_sizes


ROOT = Path(__file__).resolve().parents[1]
MAX_AUDIO_URL_CHARS = 16 * 1024
MAX_AUDIO_DOWNLOAD_BYTES = 128 * 1024 * 1024
MAX_DATASET_ROWS = 10_000_000
DATASET_AUDIO_URL_HOST = "datasets-server.huggingface.co"
PATH_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return bool(checker and checker())


def _reject_link(path: Path) -> None:
    if path.is_symlink() or _is_junction(path):
        raise ValueError(f"Refusing linked audio path: {path}")


def _safe_path_component(value: Any, label: str) -> str:
    component = str(value or "")
    if not PATH_COMPONENT_RE.fullmatch(component) or component in {".", ".."}:
        raise ValueError(f"Invalid audio {label} path component")
    return component


def validated_audio_url(value: Any, *, require_dataset_server: bool = False) -> str:
    """Accept only bounded public HTTPS URLs without embedded credentials."""
    if not isinstance(value, str) or not value or len(value) > MAX_AUDIO_URL_CHARS:
        raise ValueError("Audio download URL is missing or exceeds the size limit")
    parsed = urlsplit(value)
    hostname = parsed.hostname
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("Audio download URL has an invalid port") from error
    if (
        parsed.scheme.lower() != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.fragment
    ):
        raise ValueError("Audio download URL must be public HTTPS without credentials")
    normalized_host = hostname.rstrip(".").lower()
    if require_dataset_server and normalized_host != DATASET_AUDIO_URL_HOST:
        raise ValueError("Audio download URL must come from the configured dataset server")
    if normalized_host == "localhost" or normalized_host.endswith((".localhost", ".local")):
        raise ValueError("Audio download URL must not target a local host")
    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError("Audio download URL must not target a non-public address")
    return value


def destination_path(record: dict[str, Any], audio_root: Path) -> Path:
    source = _safe_path_component(record.get("source"), "source")
    split = _safe_path_component(record.get("source_split"), "split")
    record_id = _safe_path_component(record.get("id"), "record id")
    return Path(os.path.abspath(audio_root)) / source / split / f"{record_id}.wav"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def download_one(record: dict[str, Any], audio_root: Path, retries: int = 4) -> dict[str, Any]:
    if isinstance(retries, bool) or not isinstance(retries, int) or not 1 <= retries <= 10:
        raise ValueError("Audio download retries must be an integer from 1 to 10")
    destination = destination_path(record, audio_root)
    root = Path(os.path.abspath(audio_root))
    _reject_link(root)
    current = root
    for component in destination.parent.relative_to(root).parts:
        current /= component
        _reject_link(current)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_link(destination.parent)
    _reject_link(destination)
    if destination.is_file() and destination.stat().st_size > 44:
        result = durable_asr_record(record)
        result["audio_path"] = str(destination.resolve())
        return result
    url = validated_audio_url(record.get("audio_url"), require_dataset_server=True)
    temp = destination.with_suffix(".part")
    _reject_link(temp)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "onevoice-audio-materializer/1.0"})
            with urlopen(request, timeout=120) as response, temp.open("wb") as handle:
                validated_audio_url(response.geturl())
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        declared_bytes = int(content_length)
                    except (TypeError, ValueError) as error:
                        raise IOError("audio response has an invalid content length") from error
                    if not 0 <= declared_bytes <= MAX_AUDIO_DOWNLOAD_BYTES:
                        raise IOError("audio response exceeds the download size limit")
                downloaded_bytes = 0
                while chunk := response.read(1024 * 1024):
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > MAX_AUDIO_DOWNLOAD_BYTES:
                        raise IOError("audio response exceeds the download size limit")
                    handle.write(chunk)
            if temp.stat().st_size <= 44:
                raise IOError("downloaded audio is empty")
            os.replace(temp, destination)
            result = durable_asr_record(record)
            result["audio_path"] = str(destination.resolve())
            return result
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 10))
    temp.unlink(missing_ok=True)
    error_type = type(last_error).__name__ if last_error is not None else "unknown"
    raise RuntimeError(
        f"{record['id']}: download failed after {retries} attempts ({error_type})"
    ) from None


def refresh_audio_urls(
    records: list[dict[str, Any]],
    dataset_specs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve fresh signed URLs in memory while preserving input order."""
    order: list[tuple[str, str, str]] = []
    pending: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        source = _safe_path_component(record.get("source"), "source")
        split = _safe_path_component(record.get("source_split"), "split")
        record_id = _safe_path_component(record.get("id"), "record id")
        key = (source, split, record_id)
        wanted = pending.setdefault((source, split), {})
        if record_id in wanted:
            raise ValueError(f"Duplicate ASR record id in {source}/{split}")
        wanted[record_id] = durable_asr_record(record)
        order.append(key)

    resolved: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (source, split), wanted in sorted(pending.items()):
        spec = dataset_specs.get(source)
        if not isinstance(spec, dict):
            raise ValueError(f"Unknown ASR dataset source: {source}")
        repo_id = str(spec.get("repo_id") or "")
        if not repo_id:
            raise ValueError(f"ASR dataset source has no repository id: {source}")
        dataset_config = str(spec.get("config", "default"))
        sizes = [
            row
            for row in split_sizes(repo_id)
            if row.get("config") == dataset_config and row.get("split") == split
        ]
        if len(sizes) != 1:
            raise RuntimeError(f"{source}/{split}: split size unavailable or ambiguous")
        total_rows_value = sizes[0].get("num_rows")
        if (
            isinstance(total_rows_value, bool)
            or not isinstance(total_rows_value, int)
            or total_rows_value < 0
            or total_rows_value > MAX_DATASET_ROWS
        ):
            raise RuntimeError(f"{source}/{split}: invalid split size")
        total_rows = total_rows_value
        for offset in range(0, total_rows, 100):
            payload = api_json(
                "rows",
                {
                    "dataset": repo_id,
                    "config": dataset_config,
                    "split": split,
                    "offset": offset,
                    "length": min(100, total_rows - offset),
                },
            )
            if not isinstance(payload, dict):
                raise RuntimeError(f"{source}/{split}: malformed dataset row response")
            rows = payload.get("rows")
            if not isinstance(rows, list) or len(rows) > 100:
                raise RuntimeError(f"{source}/{split}: malformed dataset row response")
            for item in rows:
                if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
                    raise RuntimeError(f"{source}/{split}: malformed dataset row response")
                fresh = normalize_asr_row(source, spec, split, item.get("row", {}))
                fresh_id = str(fresh.get("id") or "")
                if fresh_id not in wanted:
                    continue
                try:
                    url = validated_audio_url(
                        fresh.get("audio_url"),
                        require_dataset_server=True,
                    )
                except ValueError:
                    continue
                record = wanted.pop(fresh_id)
                record["audio_url"] = url
                resolved[(source, split, fresh_id)] = record
            if not wanted:
                break
        if wanted:
            raise RuntimeError(
                f"{source}/{split}: {len(wanted)} record IDs lack a fresh audio URL"
            )
    return [resolved[key] for key in order]


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
                completed[role].append(download_one(record, args.audio_dir))
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

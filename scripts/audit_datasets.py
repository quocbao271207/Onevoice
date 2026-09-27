"""Audit Hugging Face dataset metadata before any merge or training.

ASR audits use the datasets-server row API, so audio is not downloaded merely
to inspect transcripts, durations, speakers, accents, and split leakage.
Text datasets are small enough to load normally when a full audit is requested.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.quality import (  # noqa: E402
    RunningProfile,
    audio_url,
    cross_split_leakage,
    fingerprint_text,
    normalize_text,
    stable_record_id,
    vietnamese_mark_ratio,
    write_json,
)


API = "https://datasets-server.huggingface.co"


def api_json(endpoint: str, params: dict[str, Any], retries: int = 8) -> dict[str, Any]:
    url = f"{API}/{endpoint}?{urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "onevoice-dataset-audit/1.0"})
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            time.sleep(0.35)  # stay below anonymous datasets-server burst limits
            return payload
        except HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                retry_after = int(exc.headers.get("Retry-After", "20"))
                time.sleep(min(max(retry_after, 10), 60))
            else:
                time.sleep(min(2**attempt, 12))
        except Exception as exc:  # network errors are retried with bounded backoff
            last_error = exc
            time.sleep(min(2**attempt, 12))
    raise RuntimeError(f"Dataset API failed after {retries} attempts: {url}: {last_error}")


def split_sizes(repo_id: str) -> list[dict[str, Any]]:
    payload = api_json("size", {"dataset": repo_id})
    return list(payload.get("size", {}).get("splits", []))


def iter_rows(repo_id: str, config: str, split: str, limit: int | None) -> Iterator[dict[str, Any]]:
    offset = 0
    page_size = 100
    while limit is None or offset < limit:
        length = page_size if limit is None else min(page_size, limit - offset)
        payload = api_json(
            "rows",
            {"dataset": repo_id, "config": config, "split": split, "offset": offset, "length": length},
        )
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            yield item.get("row", {})
        offset += len(rows)
        if len(rows) < length:
            break


def role_for_split(spec: dict[str, Any], split: str) -> str:
    if split in spec.get("train_splits", []):
        return "train"
    if split in spec.get("validation_splits", []):
        return "validation"
    if split in spec.get("test_splits", []):
        return "test"
    if split in spec.get("calibration_splits", []):
        return "calibration"
    return "excluded"


def duplicate_rate_violations(profiles: dict[str, RunningProfile], threshold: float) -> dict[str, float]:
    """Return splits whose repeated transcript rate exceeds the declared source policy."""
    violations = {}
    for split, profile in profiles.items():
        rate = float(profile.to_dict()["exact_duplicate_rate"])
        if rate > threshold:
            violations[split] = rate
    return violations


def normalize_asr_row(name: str, spec: dict[str, Any], split: str, row: dict[str, Any]) -> dict[str, Any]:
    text = normalize_text(row.get(spec["text_column"]))
    native_id = row.get(spec.get("id_column", "")) or row.get("utterance_id") or row.get("segment_id")
    metadata = {key: row.get(key) for key in spec.get("metadata_columns", [])}
    return {
        "id": stable_record_id(name, split, native_id, text),
        "source": name,
        "repo_id": spec["repo_id"],
        "source_split": split,
        "role": role_for_split(spec, split),
        "task": "asr",
        "language": spec["language"],
        "domain": spec["domain"],
        "text": text,
        "text_fingerprint": fingerprint_text(text),
        "duration_s": row.get(spec.get("duration_column", "")),
        "audio_url": audio_url(row.get(spec.get("audio_column", "audio"))),
        "speaker": row.get(spec.get("speaker_column", "")),
        "group": row.get(spec.get("group_column", "")),
        "accent": row.get(spec.get("accent_column", "")),
        "metadata": metadata,
    }


def select_listening_records(records: list[dict[str, Any]], limit: int = 36) -> list[dict[str, Any]]:
    """Deterministic risk + coverage sample for human transcript verification."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(record: dict[str, Any], reason: str) -> None:
        if record["id"] in seen or len(selected) >= limit:
            return
        item = dict(record)
        item["review_reason"] = reason
        selected.append(item)
        seen.add(record["id"])

    def duration(record: dict[str, Any]) -> float:
        try:
            return float(record.get("duration_s") or 0)
        except (TypeError, ValueError):
            return 0.0

    # Risk quota: do not let a large cluster of sub-second outliers consume the
    # entire listening pack. Keep at most one such example per source.
    flagged = sorted(records, key=lambda row: (-len(row.get("quality_flags", [])), row["id"]))
    flagged_added = 0
    short_added = 0
    for record in flagged:
        if record.get("quality_flags"):
            is_short = duration(record) < 2.0
            if is_short and short_added:
                continue
            add(record, "quality_flag:" + ",".join(record["quality_flags"]))
            flagged_added += 1
            short_added += int(is_short)
            if flagged_added >= 2:
                break

    # Prefer clearly audible-length rows for split/accent coverage.
    coverage_order = sorted(
        records,
        key=lambda row: (not (2.0 <= duration(row) <= 20.0), row["id"]),
    )
    for key in ("source_split", "accent"):
        values: set[str] = set()
        for record in coverage_order:
            value = normalize_text(record.get(key))
            if value and value not in values:
                add(record, f"coverage:{key}={value}")
                values.add(value)
            if len(values) >= 8:
                break

    by_duration = sorted((row for row in records if duration(row) > 0), key=duration)
    if by_duration and not short_added:
        add(by_duration[0], "duration_short_tail")
    if by_duration:
        add(by_duration[-1], "duration_long_tail")

    # Fill with ordinary 2–20 second examples first; only then use any row.
    for record in coverage_order:
        if 2.0 <= duration(record) <= 20.0:
            add(record, "deterministic_normal_duration")
    for record in coverage_order:
        add(record, "deterministic_random_coverage")
    return selected


def audit_asr(name: str, spec: dict[str, Any], output: Path, max_rows: int | None) -> dict[str, Any]:
    repo_id = spec["repo_id"]
    config = spec.get("config", "default")
    sizes = [row for row in split_sizes(repo_id) if row.get("config") == config]
    available = {row["split"]: int(row.get("num_rows", 0)) for row in sizes}
    requested = list(
        dict.fromkeys(
            spec.get("train_splits", [])
            + spec.get("validation_splits", [])
            + spec.get("test_splits", [])
            + spec.get("calibration_splits", [])
        )
    )
    records: list[dict[str, Any]] = []
    profiles: dict[str, RunningProfile] = defaultdict(RunningProfile)

    record_dir = output / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    for split in requested:
        if split not in available:
            raise RuntimeError(f"{name}: configured split '{split}' not found; available={sorted(available)}")
        split_limit = min(max_rows, available[split]) if max_rows else None
        manifest_path = record_dir / f"{name}--{split}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in iter_rows(repo_id, config, split, split_limit):
                record = normalize_asr_row(name, spec, split, row)
                flags = profiles[split].add(
                    record["text"],
                    record["duration_s"],
                    {
                        "accent": record.get("accent"),
                        "speaker": record.get("speaker"),
                        "domain": record.get("domain"),
                        "role": row.get("role"),
                        "gender": row.get("gender"),
                        "topic": row.get("topic"),
                        "recording_condition": row.get("rec_condition") or row.get("recording_context"),
                    },
                )
                if record["language"].startswith("vi") and vietnamese_mark_ratio(record["text"]) == 0:
                    flags.append("no_vietnamese_marks")
                record["quality_flags"] = sorted(set(flags))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)

    leakage = cross_split_leakage(records)
    summary = {
        "dataset": name,
        "repo_id": repo_id,
        "status": "pass",
        "full_audit": max_rows is None,
        "available_rows": available,
        "audited_rows": {split: profile.rows for split, profile in profiles.items()},
        "profiles": {split: profile.to_dict() for split, profile in profiles.items()},
        "leakage": leakage,
        "license": spec.get("license", "unknown"),
        "license_note": spec.get("license_note"),
        "max_exact_duplicate_rate": float(spec.get("max_exact_duplicate_rate", 0.08)),
    }
    if spec.get("license") == "unknown" or not max_rows is None:
        summary["status"] = "review_required"
    scored_roles = {"train", "validation", "test"}
    critical_overlap = False
    for dimension in ("group", "speaker"):
        for pair, count in leakage[dimension]["role_pair_counts"].items():
            if count and set(pair.split("<->")) <= scored_roles:
                critical_overlap = True
    if critical_overlap:
        summary["status"] = "review_required"
    duplicate_violations = duplicate_rate_violations(profiles, summary["max_exact_duplicate_rate"])
    summary["duplicate_rate_violations"] = duplicate_violations
    if duplicate_violations:
        summary["status"] = "review_required"

    review_candidates = select_listening_records(records)
    write_json(output / f"{name}.json", summary)
    review_path = output / "listening" / f"{name}.jsonl"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", encoding="utf-8") as handle:
        for record in review_candidates:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summary


def pair_medev_rows(rows: list[str], split: str) -> tuple[list[dict[str, Any]], list[str]]:
    """MedEV stores all English rows first, followed by aligned Vietnamese rows."""
    errors: list[str] = []
    if len(rows) % 2:
        errors.append(f"odd_row_count:{len(rows)}")
    pairs: list[dict[str, Any]] = []
    half = len(rows) // 2
    for offset in range(half):
        source = normalize_text(rows[offset])
        target = normalize_text(rows[half + offset])
        pair_fp = fingerprint_text(source + "\x1f" + target)
        pairs.append(
            {
                "id": stable_record_id("medev", split, offset, pair_fp),
                "source": "medev",
                "source_split": split,
                "role": "validation" if split == "validation" else split,
                "task": "mt",
                "source_language": "en",
                "target_language": "vi",
                "source_text": source,
                "target_text": target,
                "pair_fingerprint": pair_fp,
            }
        )
    return pairs, errors


def audit_medev(name: str, spec: dict[str, Any], output: Path, max_rows: int | None) -> dict[str, Any]:
    from datasets import load_dataset

    record_dir = output / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    requested = list(dict.fromkeys(spec["train_splits"] + spec["validation_splits"] + spec["test_splits"]))
    profiles: dict[str, RunningProfile] = {}
    errors: list[str] = []
    pair_fingerprints: dict[str, set[str]] = {}
    row_counts: dict[str, int] = {}

    for split in requested:
        dataset = load_dataset(spec["repo_id"], spec.get("config"), split=split)
        if max_rows:
            dataset = dataset.select(range(min(max_rows, len(dataset))))
        raw_text = [normalize_text(value) for value in dataset[spec["text_column"]]]
        pairs, pair_errors = pair_medev_rows(raw_text, split)
        errors.extend(f"{split}:{error}" for error in pair_errors)
        row_counts[split] = len(raw_text)
        profile = RunningProfile()
        pair_fingerprints[split] = set()
        path = record_dir / f"{name}--{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for pair in pairs:
                profile.add(pair["source_text"] + " ||| " + pair["target_text"])
                pair_fingerprints[split].add(pair["pair_fingerprint"])
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        profiles[split] = profile

    leakage: dict[str, int] = {}
    for i, left in enumerate(requested):
        for right in requested[i + 1 :]:
            leakage[f"{left}<->{right}"] = len(pair_fingerprints[left] & pair_fingerprints[right])

    status = "pass"
    if errors or any(leakage.values()) or spec.get("license") == "unknown" or max_rows is not None:
        status = "review_required"
    summary = {
        "dataset": name,
        "repo_id": spec["repo_id"],
        "status": status,
        "full_audit": max_rows is None,
        "storage_layout": spec["storage_layout"],
        "raw_rows": row_counts,
        "paired_rows": {split: profile.rows for split, profile in profiles.items()},
        "profiles": {split: profile.to_dict() for split, profile in profiles.items()},
        "cross_split_pair_leakage": leakage,
        "structural_errors": errors,
        "license": spec.get("license", "unknown"),
        "license_note": spec.get("license_note"),
    }
    write_json(output / f"{name}.json", summary)
    return summary


def markdown_report(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# OneVoice Dataset EDA Report",
        "",
        "> Generated before merge. A sampled audit cannot unlock a production merge.",
        "",
        "| Dataset | Status | Full | Audited | License |",
        "|---|---:|---:|---:|---|",
    ]
    for item in summaries:
        counts = item.get("audited_rows") or item.get("paired_rows") or {}
        lines.append(
            f"| `{item['dataset']}` | {item['status']} | {item['full_audit']} | "
            f"{sum(counts.values()):,} | {item.get('license', 'unknown')} |"
        )
    lines.extend(["", "## Required human decisions", ""])
    for item in summaries:
        if item["status"] not in {"pass", "reviewed"}:
            lines.append(f"- `{item['dataset']}`: inspect `{item['dataset']}.json` and record a decision before merge.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "reports" / "eda")
    parser.add_argument("--dataset", action="append", help="Audit only this configured key (repeatable).")
    parser.add_argument("--max-rows", type=int, default=0, help="Per split; 0 performs a full audit.")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    selected = set(args.dataset or [])
    summaries: list[dict[str, Any]] = []
    for name, spec in config["datasets"].items():
        if selected and name not in selected:
            continue
        if not spec.get("enabled"):
            continue
        print(f"[AUDIT] {name} ({spec['repo_id']})", flush=True)
        spec = {**config.get("quality_gate", {}), **spec}
        max_rows = args.max_rows or None
        if spec["task"] == "asr":
            summary = audit_asr(name, spec, args.output_dir, max_rows)
        elif name == "medev" and spec.get("storage_layout") == "parallel_halves_en_then_vi":
            summary = audit_medev(name, spec, args.output_dir, max_rows)
        else:
            print(f"[SKIP] No normalizer yet for task={spec['task']}: {name}", flush=True)
            continue
        summaries.append(summary)
        print(f"[DONE] {name}: {summary['status']}", flush=True)

    # Preserve summaries from earlier per-dataset runs so a long audit can be
    # resumed without re-downloading every source merely to rebuild the index.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined: dict[str, dict[str, Any]] = {}
    for path in args.output_dir.glob("*.json"):
        if path.name == "audit_index.json":
            continue
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(item, dict) and item.get("dataset"):
            combined[item["dataset"]] = item
    for item in summaries:
        combined[item["dataset"]] = item
    all_summaries = [combined[key] for key in sorted(combined)]
    (args.output_dir / "REPORT.md").write_text(markdown_report(all_summaries), encoding="utf-8")
    write_json(args.output_dir / "audit_index.json", {"datasets": all_summaries})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

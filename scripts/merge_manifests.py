"""Build audited manifests with source-specific, leakage-safe split policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import yaml


ROOT = Path(__file__).resolve().parents[1]
ROLES = ("test", "validation", "train")
FATAL_FLAGS = {
    "empty_text",
    "invalid_duration",
    "too_short",
    "too_long",
    "high_text_audio_ratio",
    "length_ratio_outlier",
    "very_long_pair",
    "source_language_suspect",
    "target_language_suspect",
    "untranslated_identity",
}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def audit_allows_merge(audit: dict[str, Any], allow_reviewed: bool) -> bool:
    if not audit.get("full_audit"):
        return False
    if audit.get("status") == "pass":
        return True
    return allow_reviewed and audit.get("status") == "reviewed"


def hash_fraction(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}\x1f{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def repartition_role(spec: dict[str, Any], group: str) -> str:
    ratios = spec["repartition_ratios"]
    train_edge = float(ratios["train"])
    validation_edge = train_edge + float(ratios["validation"])
    value = hash_fraction(int(spec.get("repartition_seed", 0)), group)
    if value < train_edge:
        return "train"
    if value < validation_edge:
        return "validation"
    return "test"


def source_records(
    name: str,
    spec: dict[str, Any],
    records_dir: Path,
    policy_mode: str,
) -> dict[str, list[dict[str, Any]]]:
    """Load one source and assign target roles without mixing recording groups."""
    by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    use_repartition = policy_mode == "configured" and spec.get("split_policy") == "group_hash_repartition"
    if not use_repartition:
        split_keys = {"test": "test_splits", "validation": "validation_splits", "train": "train_splits"}
        for role in ROLES:
            for split in spec.get(split_keys[role], []):
                for record in read_jsonl(records_dir / f"{name}--{split}.jsonl"):
                    item = dict(record)
                    item["role"] = role
                    item["merge_policy"] = "official_split"
                    by_role[role].append(item)
        max_per_group = int(spec.get("max_train_per_group") or 0)
        if max_per_group and by_role["train"]:
            seed = int(spec.get("sampling_seed", 0))
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in by_role["train"]:
                group = str(item.get("group") or f"record:{item['id']}").strip().casefold()
                grouped[group].append(item)
            sampled = []
            for group_rows in grouped.values():
                sampled.extend(
                    sorted(group_rows, key=lambda row: hash_fraction(seed, str(row["id"])))[:max_per_group]
                )
            for item in sampled:
                item["merge_policy"] = "official_split_group_capped"
            by_role["train"] = sampled
        return by_role

    pooled_splits = list(spec.get("repartition_splits", []))
    locked_splits = set(spec.get("locked_test_splits", []))
    all_splits = list(dict.fromkeys(pooled_splits + list(locked_splits)))
    records = []
    for split in all_splits:
        records.extend(read_jsonl(records_dir / f"{name}--{split}.jsonl"))

    locked_groups = {
        str(record.get("group") or "").strip().casefold()
        for record in records
        if record.get("source_split") in locked_splits and record.get("group")
    }
    for record in records:
        item = dict(record)
        group = str(item.get("group") or f"record:{item['id']}").strip().casefold()
        if item.get("source_split") in locked_splits or group in locked_groups:
            role = "test"
            policy = "locked_test_group"
        else:
            role = repartition_role(spec, group)
            policy = "group_hash_repartition"
        item["role"] = role
        item["merge_policy"] = policy
        by_role[role].append(item)
    return by_role


def merge_task(
    task: str,
    sources: list[tuple[str, dict[str, Any]]],
    records_dir: Path,
    output_dir: Path,
    excluded_ids: set[str] | None = None,
    policy_mode: str = "configured",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_ids = excluded_ids or set()
    prepared = {
        name: source_records(name, spec, records_dir, policy_mode)
        for name, spec in sources
    }
    kept = Counter()
    dropped = Counter()
    observed = Counter()
    kept_by_source_role: dict[str, Counter[str]] = defaultdict(Counter)
    hours_by_source_role: dict[str, Counter[str]] = defaultdict(Counter)
    dropped_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    fingerprints_by_role: dict[str, set[str]] = {role: set() for role in ROLES}
    speakers_by_role: dict[str, set[str]] = {role: set() for role in ROLES}
    groups_by_role: dict[str, set[str]] = {role: set() for role in ROLES}
    seen_record_ids: set[str] = set()
    handles = {role: (output_dir / f"{task}--{role}.jsonl").open("w", encoding="utf-8") for role in ROLES}

    def drop(source: str, reason: str) -> None:
        dropped[reason] += 1
        dropped_by_source[source][reason] += 1

    try:
        for role in ROLES:
            higher_roles = ["test"] if role == "validation" else (["test", "validation"] if role == "train" else [])
            for name, spec in sorted(sources, key=lambda item: -int(item[1].get("priority", 0))):
                for record in prepared[name][role]:
                    record_id = str(record.get("id") or "")
                    if record_id and record_id in excluded_ids:
                        drop(name, "reviewed_exclusion")
                        continue
                    if record_id and record_id in seen_record_ids:
                        drop(name, "duplicate_record_id")
                        continue
                    flags = set(record.get("quality_flags") or [])
                    fatal_flags = flags & FATAL_FLAGS
                    if fatal_flags:
                        drop(name, "fatal_quality_flags")
                        continue

                    fingerprint = str(record.get("text_fingerprint") or record.get("pair_fingerprint") or "")
                    speaker = str(record.get("speaker") or "").strip().casefold()
                    group = str(record.get("group") or "").strip().casefold()
                    if task == "mt":
                        if fingerprint in fingerprints_by_role[role]:
                            drop(name, "exact_pair_duplicate")
                            continue
                        if any(fingerprint in fingerprints_by_role[higher] for higher in higher_roles):
                            drop(name, "pair_overlap_with_higher_role")
                            continue
                    elif fingerprint and any(fingerprint in fingerprints_by_role[higher] for higher in higher_roles):
                        # Same transcript with distinct audio is valid ASR data, but is
                        # counted so prompt-overlap metrics remain transparent.
                        observed["asr_transcript_overlap_with_higher_role"] += 1

                    if speaker and any(speaker in speakers_by_role[higher] for higher in higher_roles):
                        drop(name, "speaker_overlap_with_higher_role")
                        continue
                    if group and any(group in groups_by_role[higher] for higher in higher_roles):
                        drop(name, "group_overlap_with_higher_role")
                        continue

                    if record_id:
                        seen_record_ids.add(record_id)
                    if fingerprint:
                        fingerprints_by_role[role].add(fingerprint)
                    if speaker:
                        speakers_by_role[role].add(speaker)
                    if group:
                        groups_by_role[role].add(group)
                    handles[role].write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept[role] += 1
                    kept_by_source_role[name][role] += 1
                    try:
                        hours_by_source_role[name][role] += float(record.get("duration_s") or 0) / 3600
                    except (TypeError, ValueError):
                        pass
    finally:
        for handle in handles.values():
            handle.close()

    return {
        "task": task,
        "policy_mode": policy_mode,
        "kept": dict(kept),
        "kept_by_source_role": {name: dict(counts) for name, counts in sorted(kept_by_source_role.items())},
        "hours_by_source_role": {
            name: {role: round(hours, 4) for role, hours in counts.items()}
            for name, counts in sorted(hours_by_source_role.items())
        },
        "dropped": dict(dropped),
        "dropped_by_source": {name: dict(counts) for name, counts in sorted(dropped_by_source.items())},
        "observed_not_dropped": dict(observed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "data" / "reports" / "eda")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "manifests")
    parser.add_argument("--policy-mode", choices=["configured", "strict"], default="configured")
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=ROOT / "data" / "review" / "listening_pack" / "signal_exclusions.jsonl",
    )
    parser.add_argument(
        "--exclusions-config",
        type=Path,
        default=ROOT / "configs" / "data_exclusions.yaml",
    )
    parser.add_argument("--allow-reviewed", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    excluded_ids: set[str] = set()
    if args.exclusions.is_file():
        excluded_ids.update(item["id"] for item in read_jsonl(args.exclusions) if item.get("id"))
    if args.exclusions_config.is_file():
        exclusion_config = yaml.safe_load(args.exclusions_config.read_text(encoding="utf-8")) or {}
        excluded_ids.update(item["id"] for item in exclusion_config.get("exclusions", []) if item.get("id"))

    by_task: dict[str, list[tuple[str, dict[str, Any]]]] = {"asr": [], "mt": []}
    failures: list[str] = []
    for name, spec in config["datasets"].items():
        if not spec.get("enabled") or spec["task"] not in by_task:
            continue
        audit_path = args.audit_dir / f"{name}.json"
        if not audit_path.exists():
            failures.append(f"{name}: missing audit {audit_path}")
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit_allows_merge(audit, args.allow_reviewed):
            failures.append(f"{name}: audit status={audit.get('status')} full={audit.get('full_audit')}")
            continue
        by_task[spec["task"]].append((name, spec))
    if failures:
        raise SystemExit("Merge blocked by EDA gate:\n- " + "\n- ".join(failures))

    summaries = [
        merge_task(task, sources, args.audit_dir / "records", args.output_dir, excluded_ids, args.policy_mode)
        for task, sources in by_task.items()
        if sources
    ]
    (args.output_dir / "merge_summary.json").write_text(
        json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

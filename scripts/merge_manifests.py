"""Build audited manifests with source-specific, leakage-safe split policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from collections import Counter, defaultdict
from numbers import Real
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.durable_file import write_durable_bytes  # noqa: E402
from src.pipeline.durable_json import write_durable_json  # noqa: E402
from src.pipeline.stable_json import (  # noqa: E402
    StableJsonDocument,
    read_stable_json_mapping,
)
from src.pipeline.stable_jsonl import (  # noqa: E402
    StableJsonlDocument,
    read_stable_jsonl_mappings,
)
from src.pipeline.stable_yaml import (  # noqa: E402
    StableYamlDocument,
    read_stable_yaml_mapping,
)
from scripts.validate_manifests import validate_rows  # noqa: E402
from src.utils.bounded_file import read_stable_regular_file  # noqa: E402


ROLES = ("test", "validation", "train")
MAX_PROJECT_CONFIG_BYTES = 1_000_000
MAX_AUDIT_REPORT_BYTES = 50 * 1024 * 1024
MAX_RECORDS_FILE_BYTES = 512 * 1024 * 1024
MAX_RECORD_LINE_BYTES = 2 * 1024 * 1024
MAX_RECORD_ROWS = 500_000
MAX_EXCLUSIONS_BYTES = 10 * 1024 * 1024
MAX_EXCLUSION_ROWS = 100_000
MAX_OUTPUT_MANIFEST_BYTES = 512 * 1024 * 1024
MAX_MERGE_SUMMARY_BYTES = 16 * 1024 * 1024
SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
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


def read_jsonl(
    path: Path,
    *,
    label: str,
    maximum_bytes: int = MAX_RECORDS_FILE_BYTES,
    maximum_rows: int = MAX_RECORD_ROWS,
) -> StableJsonlDocument:
    return read_stable_jsonl_mappings(
        path,
        maximum_bytes=maximum_bytes,
        maximum_line_bytes=MAX_RECORD_LINE_BYTES,
        maximum_rows=maximum_rows,
        label=label,
    )


def _safe_segment(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not SAFE_SEGMENT.fullmatch(value)
        or value in {".", ".."}
    ):
        raise ValueError(f"{label} must be a safe filename segment")
    return value


def _evidence(
    document: StableJsonDocument | StableJsonlDocument | StableYamlDocument,
) -> dict[str, Any]:
    rows = len(document.rows) if isinstance(document, StableJsonlDocument) else None
    result: dict[str, Any] = {
        "path": str(document.path),
        "bytes": document.bytes,
        "sha256": document.sha256,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _serialize_manifest(
    task: str,
    role: str,
    rows: list[dict[str, Any]],
) -> bytes:
    validation_errors = validate_rows(task, role, rows)
    if validation_errors:
        raise ValueError(
            f"{task} {role} manifest failed schema validation: "
            + "; ".join(validation_errors[:10])
        )
    if not rows:
        raise ValueError(f"{task} {role} manifest is empty")
    if len(rows) > MAX_RECORD_ROWS:
        raise ValueError(f"{task} {role} manifest exceeds {MAX_RECORD_ROWS} rows")

    payload = bytearray()
    for index, row in enumerate(rows, start=1):
        try:
            encoded = (
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError):
            raise ValueError(
                f"{task} {role} manifest row {index} is not strict JSON"
            ) from None
        if len(encoded) > MAX_RECORD_LINE_BYTES:
            raise ValueError(
                f"{task} {role} manifest row {index} exceeds "
                f"{MAX_RECORD_LINE_BYTES} bytes"
            )
        payload.extend(encoded)
        if len(payload) > MAX_OUTPUT_MANIFEST_BYTES:
            raise ValueError(
                f"{task} {role} manifest exceeds {MAX_OUTPUT_MANIFEST_BYTES} bytes"
            )
    return bytes(payload)


def _normalized_quality_flags(record: dict[str, Any], *, label: str) -> set[str]:
    raw_flags = record.get("quality_flags", [])
    if not isinstance(raw_flags, list) or any(
        not isinstance(flag, str) or not flag or len(flag) > 256
        for flag in raw_flags
    ):
        raise ValueError(f"{label} quality_flags must be bounded strings")
    return set(raw_flags)


def _excluded_ids(rows: Any, *, label: str) -> set[str]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    result: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {index} must be a mapping")
        value = row.get("id")
        if not isinstance(value, str) or not value.strip() or len(value) > 1_024:
            raise ValueError(f"{label} row {index} has an invalid id")
        result.add(value.strip())
    return result


def audit_allows_merge(audit: dict[str, Any], allow_reviewed: bool) -> bool:
    if audit.get("full_audit") is not True:
        return False
    if audit.get("status") == "pass":
        return True
    return allow_reviewed and audit.get("status") == "reviewed"


def hash_fraction(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}\x1f{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def repartition_role(spec: dict[str, Any], group: str) -> str:
    ratios = spec["repartition_ratios"]
    if not isinstance(ratios, dict) or set(ratios) != set(ROLES):
        raise ValueError("repartition_ratios must contain train/validation/test")
    normalized: dict[str, float] = {}
    for role in ROLES:
        ratio = ratios[role]
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, Real)
            or not math.isfinite(float(ratio))
            or not 0 <= float(ratio) <= 1
        ):
            raise ValueError(f"repartition_ratios.{role} must be finite in [0, 1]")
        normalized[role] = float(ratio)
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("repartition_ratios must sum to 1")
    seed = spec.get("repartition_seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("repartition_seed must be an integer")
    train_edge = normalized["train"]
    validation_edge = train_edge + normalized["validation"]
    value = hash_fraction(seed, group)
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
    input_evidence: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load one source and assign target roles without mixing recording groups."""
    name = _safe_segment(name, label="Dataset name")
    if not isinstance(spec, dict):
        raise ValueError(f"Dataset {name} specification must be a mapping")
    if policy_mode not in {"configured", "strict"}:
        raise ValueError("policy_mode must be configured or strict")
    by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
    use_repartition = policy_mode == "configured" and spec.get("split_policy") == "group_hash_repartition"
    if not use_repartition:
        split_keys = {"test": "test_splits", "validation": "validation_splits", "train": "train_splits"}
        for role in ROLES:
            splits = spec.get(split_keys[role], [])
            if not isinstance(splits, list):
                raise ValueError(f"Dataset {name} {split_keys[role]} must be a list")
            for raw_split in splits:
                split = _safe_segment(raw_split, label=f"Dataset {name} split")
                document = read_jsonl(
                    records_dir / f"{name}--{split}.jsonl",
                    label=f"Dataset {name} split {split}",
                )
                if input_evidence is not None:
                    input_evidence[f"{name}--{split}"] = _evidence(document)
                for record in document.rows:
                    item = dict(record)
                    item.pop("audio_url", None)
                    item["role"] = role
                    item["merge_policy"] = "official_split"
                    by_role[role].append(item)
        max_per_group = spec.get("max_train_per_group", 0)
        if (
            isinstance(max_per_group, bool)
            or not isinstance(max_per_group, int)
            or not 0 <= max_per_group <= MAX_RECORD_ROWS
        ):
            raise ValueError(f"Dataset {name} max_train_per_group is invalid")
        if max_per_group and by_role["train"]:
            seed = spec.get("sampling_seed", 0)
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(f"Dataset {name} sampling_seed must be an integer")
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

    pooled_splits_value = spec.get("repartition_splits", [])
    locked_splits_value = spec.get("locked_test_splits", [])
    if not isinstance(pooled_splits_value, list) or not isinstance(
        locked_splits_value, list
    ):
        raise ValueError(f"Dataset {name} repartition split lists are invalid")
    pooled_splits = [
        _safe_segment(split, label=f"Dataset {name} repartition split")
        for split in pooled_splits_value
    ]
    locked_splits = {
        _safe_segment(split, label=f"Dataset {name} locked split")
        for split in locked_splits_value
    }
    all_splits = list(dict.fromkeys(pooled_splits + list(locked_splits)))
    records = []
    for split in all_splits:
        document = read_jsonl(
            records_dir / f"{name}--{split}.jsonl",
            label=f"Dataset {name} split {split}",
        )
        if input_evidence is not None:
            input_evidence[f"{name}--{split}"] = _evidence(document)
        records.extend(document.rows)

    locked_groups = {
        str(record.get("group") or "").strip().casefold()
        for record in records
        if record.get("source_split") in locked_splits and record.get("group")
    }
    for record in records:
        item = dict(record)
        item.pop("audio_url", None)
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
    if task not in {"asr", "mt"}:
        raise ValueError("task must be asr or mt")
    if not sources:
        raise ValueError(f"{task} requires at least one source")
    excluded_ids = excluded_ids or set()
    if any(not isinstance(item, str) or not item or len(item) > 1_024 for item in excluded_ids):
        raise ValueError("excluded_ids must contain bounded non-empty strings")
    input_evidence: dict[str, dict[str, Any]] = {}
    prepared = {
        name: source_records(
            name,
            spec,
            records_dir,
            policy_mode,
            input_evidence,
        )
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
    output_rows: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}

    def drop(source: str, reason: str) -> None:
        dropped[reason] += 1
        dropped_by_source[source][reason] += 1

    priorities: dict[str, int] = {}
    for name, spec in sources:
        priority = spec.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError(f"Dataset {name} priority must be an integer")
        priorities[name] = priority

    for role in ROLES:
        higher_roles = ["test"] if role == "validation" else (["test", "validation"] if role == "train" else [])
        for name, _spec in sorted(sources, key=lambda item: -priorities[item[0]]):
            for record in prepared[name][role]:
                record_id_value = record.get("id")
                record_id = record_id_value.strip() if isinstance(record_id_value, str) else ""
                normalized_id = record_id.casefold()
                if record_id and record_id in excluded_ids:
                    drop(name, "reviewed_exclusion")
                    continue
                if normalized_id and normalized_id in seen_record_ids:
                    drop(name, "duplicate_record_id")
                    continue
                flags = _normalized_quality_flags(
                    record,
                    label=f"Dataset {name} record",
                )
                fatal_flags = flags & FATAL_FLAGS
                if fatal_flags:
                    drop(name, "fatal_quality_flags")
                    continue

                fingerprint_value = record.get("text_fingerprint") or record.get("pair_fingerprint")
                fingerprint = (
                    fingerprint_value.strip().casefold()
                    if isinstance(fingerprint_value, str)
                    else ""
                )
                speaker_value = record.get("speaker")
                speaker = (
                    speaker_value.strip().casefold()
                    if isinstance(speaker_value, str)
                    else ""
                )
                group_value = record.get("group")
                group = (
                    group_value.strip().casefold()
                    if isinstance(group_value, str)
                    else ""
                )
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

                if normalized_id:
                    seen_record_ids.add(normalized_id)
                if fingerprint:
                    fingerprints_by_role[role].add(fingerprint)
                if speaker:
                    speakers_by_role[role].add(speaker)
                if group:
                    groups_by_role[role].add(group)
                output_rows[role].append(record)
                kept[role] += 1
                kept_by_source_role[name][role] += 1
                duration = record.get("duration_s")
                if (
                    isinstance(duration, Real)
                    and not isinstance(duration, bool)
                    and math.isfinite(float(duration))
                    and float(duration) >= 0
                ):
                    hours_by_source_role[name][role] += float(duration) / 3600

    payloads = {
        role: _serialize_manifest(task, role, output_rows[role])
        for role in ROLES
    }
    output_evidence: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        path = output_dir / f"{task}--{role}.jsonl"
        payload = payloads[role]
        digest = hashlib.sha256(payload).hexdigest()
        write_durable_bytes(
            path,
            payload,
            maximum_bytes=MAX_OUTPUT_MANIFEST_BYTES,
            label=f"{task} {role} merged manifest",
        )
        persisted = read_stable_jsonl_mappings(
            path,
            maximum_bytes=MAX_OUTPUT_MANIFEST_BYTES,
            maximum_line_bytes=MAX_RECORD_LINE_BYTES,
            maximum_rows=MAX_RECORD_ROWS,
            label=f"Persisted {task} {role} merged manifest",
            expected_sha256=digest,
        )
        if persisted.rows != output_rows[role]:
            raise RuntimeError(f"Persisted {task} {role} manifest changed semantically")
        output_evidence[role] = _evidence(persisted)

    return {
        "task": task,
        "policy_mode": policy_mode,
        "input_evidence": {
            key: input_evidence[key] for key in sorted(input_evidence)
        },
        "output_evidence": output_evidence,
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

    config_document = read_stable_yaml_mapping(
        args.config,
        maximum_bytes=MAX_PROJECT_CONFIG_BYTES,
        label="Manifest merge dataset config",
    )
    config = config_document.mapping
    datasets = config.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Manifest merge dataset config requires a datasets mapping")
    control_evidence: dict[str, Any] = {
        "dataset_config": _evidence(config_document),
        "audits": {},
    }
    excluded_ids: set[str] = set()
    if args.exclusions.exists():
        exclusions_document = read_jsonl(
            args.exclusions,
            label="Manifest merge reviewed exclusions",
            maximum_bytes=MAX_EXCLUSIONS_BYTES,
            maximum_rows=MAX_EXCLUSION_ROWS,
        )
        excluded_ids.update(
            _excluded_ids(
                exclusions_document.rows,
                label="Manifest merge reviewed exclusions",
            )
        )
        control_evidence["reviewed_exclusions"] = _evidence(exclusions_document)
    if args.exclusions_config.exists():
        exclusion_document = read_stable_yaml_mapping(
            args.exclusions_config,
            maximum_bytes=MAX_PROJECT_CONFIG_BYTES,
            label="Manifest merge exclusions config",
        )
        excluded_ids.update(
            _excluded_ids(
                exclusion_document.mapping.get("exclusions"),
                label="Manifest merge configured exclusions",
            )
        )
        control_evidence["exclusions_config"] = _evidence(exclusion_document)

    by_task: dict[str, list[tuple[str, dict[str, Any]]]] = {"asr": [], "mt": []}
    failures: list[str] = []
    for raw_name, spec in datasets.items():
        name = _safe_segment(raw_name, label="Dataset name")
        if not isinstance(spec, dict):
            raise ValueError(f"Dataset {name} specification must be a mapping")
        enabled = spec.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError(f"Dataset {name} enabled must be boolean")
        if not enabled:
            continue
        task = spec.get("task")
        if task not in by_task:
            raise ValueError(f"Enabled dataset {name} has unsupported task")
        audit_path = args.audit_dir / f"{name}.json"
        if not audit_path.exists():
            failures.append(f"{name}: missing audit {audit_path}")
            continue
        audit_document = read_stable_json_mapping(
            audit_path,
            maximum_bytes=MAX_AUDIT_REPORT_BYTES,
            label=f"Dataset {name} audit",
        )
        audit = audit_document.mapping
        control_evidence["audits"][name] = _evidence(audit_document)
        if not audit_allows_merge(audit, args.allow_reviewed):
            failures.append(f"{name}: audit status={audit.get('status')} full={audit.get('full_audit')}")
            continue
        by_task[task].append((name, spec))
    if failures:
        raise SystemExit("Merge blocked by EDA gate:\n- " + "\n- ".join(failures))

    with tempfile.TemporaryDirectory(prefix="onevoice-manifest-merge-") as temporary:
        staging_dir = Path(temporary)
        summaries = [
            merge_task(
                task,
                sources,
                args.audit_dir / "records",
                staging_dir,
                excluded_ids,
                args.policy_mode,
            )
            for task, sources in by_task.items()
            if sources
        ]
        for summary in summaries:
            task = summary["task"]
            for role in ROLES:
                staged_evidence = summary["output_evidence"][role]
                payload = read_stable_regular_file(
                    Path(staged_evidence["path"]),
                    maximum_bytes=MAX_OUTPUT_MANIFEST_BYTES,
                    label=f"Staged {task} {role} merged manifest",
                )
                if (
                    len(payload) != staged_evidence["bytes"]
                    or hashlib.sha256(payload).hexdigest() != staged_evidence["sha256"]
                ):
                    raise RuntimeError(f"Staged {task} {role} manifest identity changed")
                destination = args.output_dir / f"{task}--{role}.jsonl"
                write_durable_bytes(
                    destination,
                    payload,
                    maximum_bytes=MAX_OUTPUT_MANIFEST_BYTES,
                    label=f"{task} {role} merged manifest",
                )
                summary["output_evidence"][role] = {
                    **staged_evidence,
                    "path": str(Path(destination).absolute()),
                }
    write_durable_json(
        args.output_dir / "merge_summary.json",
        {
            "control_evidence": control_evidence,
            "excluded_ids": len(excluded_ids),
            "summaries": summaries,
        },
        maximum_bytes=MAX_MERGE_SUMMARY_BYTES,
        label="Manifest merge summary",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

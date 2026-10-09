"""Validate merged manifests and emit reproducible counts/checksums before audio download."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from numbers import Real
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.durable_json import write_durable_json  # noqa: E402
from src.pipeline.stable_jsonl import (  # noqa: E402
    StableJsonlDocument,
    read_stable_jsonl_mappings,
)


ROLES = ("train", "validation", "test")
TASKS = ("asr", "mt")
MAX_MANIFEST_BYTES = 512 * 1024 * 1024
MAX_MANIFEST_LINE_BYTES = 2 * 1024 * 1024
MAX_MANIFEST_ROWS = 500_000
MAX_VALIDATION_REPORT_BYTES = 4 * 1024 * 1024
MAX_ROW_ERRORS_PER_MANIFEST = 1_000


def load(path: Path, *, label: str) -> StableJsonlDocument:
    return read_stable_jsonl_mappings(
        path,
        maximum_bytes=MAX_MANIFEST_BYTES,
        maximum_line_bytes=MAX_MANIFEST_LINE_BYTES,
        maximum_rows=MAX_MANIFEST_ROWS,
        label=label,
    )


def _required_text(
    row: dict[str, Any],
    key: str,
    *,
    location: str,
    errors: list[str],
    maximum_length: int,
) -> str | None:
    value = row.get(key)
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum_length
    ):
        errors.append(f"{location}:invalid_{key}")
        return None
    return value.strip()


def _optional_dimension(
    row: dict[str, Any],
    key: str,
    *,
    location: str,
    errors: list[str],
) -> None:
    value = row.get(key)
    if value is not None and (
        not isinstance(value, str) or not value.strip() or len(value) > 1_024
    ):
        errors.append(f"{location}:invalid_{key}")


def validate_rows(task: str, role: str, rows: list[dict[str, Any]]) -> list[str]:
    """Return deterministic schema and within-role integrity failures."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    fingerprint_key = "text_fingerprint" if task == "asr" else "pair_fingerprint"
    for index, row in enumerate(rows, start=1):
        location = f"{task}:{role}:row={index}"
        record_id = _required_text(
            row, "id", location=location, errors=errors, maximum_length=1_024
        )
        _required_text(
            row, "source", location=location, errors=errors, maximum_length=256
        )
        _required_text(
            row, "merge_policy", location=location, errors=errors, maximum_length=256
        )
        _required_text(
            row,
            fingerprint_key,
            location=location,
            errors=errors,
            maximum_length=128,
        )
        row_task = _required_text(
            row, "task", location=location, errors=errors, maximum_length=32
        )
        row_role = _required_text(
            row, "role", location=location, errors=errors, maximum_length=32
        )
        if row_task is not None and row_task != task:
            errors.append(f"{location}:task_mismatch")
        if row_role is not None and row_role != role:
            errors.append(f"{location}:role_mismatch")
        if record_id is not None:
            normalized_id = record_id.casefold()
            if normalized_id in seen_ids:
                errors.append(f"{location}:duplicate_id")
            seen_ids.add(normalized_id)

        duration = row.get("duration_s")
        if duration is not None and (
            isinstance(duration, bool)
            or not isinstance(duration, Real)
            or not math.isfinite(float(duration))
            or float(duration) < 0
        ):
            errors.append(f"{location}:invalid_duration_s")

        if task == "asr":
            _optional_dimension(row, "group", location=location, errors=errors)
            _optional_dimension(row, "speaker", location=location, errors=errors)
        if len(errors) >= MAX_ROW_ERRORS_PER_MANIFEST:
            errors.append(f"{task}:{role}:row_errors_truncated")
            break
    return errors


def dimension(rows: list[dict[str, Any]], key: str) -> set[str]:
    return {
        value.strip().casefold()
        for row in rows
        if isinstance((value := row.get(key)), str) and value.strip()
    }


def _duration_hours(rows: list[dict[str, Any]]) -> float:
    total_seconds = 0.0
    for row in rows:
        value = row.get("duration_s")
        if (
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
        ):
            total_seconds += float(value)
    return total_seconds / 3600


def _count_text(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    return Counter(
        value
        if (
            isinstance((value := row.get(key)), str)
            and value
            and len(value) <= 256
        )
        else "<invalid>"
        for row in rows
    )


def validate_manifest_dir(manifest_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "manifest_dir": str(manifest_dir),
        "tasks": {},
        "errors": [],
    }
    errors: list[str] = report["errors"]
    for task in TASKS:
        documents = {
            role: load(
                manifest_dir / f"{task}--{role}.jsonl",
                label=f"{task} {role} manifest",
            )
            for role in ROLES
        }
        rows = {role: document.rows for role, document in documents.items()}
        task_report: dict[str, Any] = {"roles": {}, "cross_role": {}}
        for role, role_rows in rows.items():
            errors.extend(validate_rows(task, role, role_rows))
            document = documents[role]
            task_report["roles"][role] = {
                "rows": len(role_rows),
                "hours": round(_duration_hours(role_rows), 4),
                "sources": dict(_count_text(role_rows, "source")),
                "merge_policies": dict(_count_text(role_rows, "merge_policy")),
                "bytes": document.bytes,
                "sha256": document.sha256,
            }

        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            pair = f"{left}<->{right}"
            ids = dimension(rows[left], "id") & dimension(rows[right], "id")
            if ids:
                errors.append(f"{task}:{pair}:duplicate_ids={len(ids)}")
            result = {"duplicate_ids": len(ids)}
            if task == "asr":
                groups = dimension(rows[left], "group") & dimension(rows[right], "group")
                speakers = dimension(rows[left], "speaker") & dimension(rows[right], "speaker")
                transcripts = dimension(rows[left], "text_fingerprint") & dimension(
                    rows[right], "text_fingerprint"
                )
                result.update(
                    {
                        "recording_group_overlap": len(groups),
                        "speaker_overlap": len(speakers),
                        "transcript_overlap_observed_not_blocking": len(transcripts),
                    }
                )
                if groups:
                    errors.append(f"asr:{pair}:group_overlap={len(groups)}")
                if speakers:
                    errors.append(f"asr:{pair}:speaker_overlap={len(speakers)}")
            else:
                pairs = dimension(rows[left], "pair_fingerprint") & dimension(
                    rows[right], "pair_fingerprint"
                )
                result["exact_pair_overlap"] = len(pairs)
                if pairs:
                    errors.append(f"mt:{pair}:pair_overlap={len(pairs)}")
            task_report["cross_role"][pair] = result
        report["tasks"][task] = task_report

    report["status"] = "pass" if not errors else "fail"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "manifests",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "reports" / "manifests" / "validation.json",
    )
    args = parser.parse_args()

    report = validate_manifest_dir(args.manifest_dir)
    write_durable_json(
        args.output,
        report,
        maximum_bytes=MAX_VALIDATION_REPORT_BYTES,
        label="Manifest validation report",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

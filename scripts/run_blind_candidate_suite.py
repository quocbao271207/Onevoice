"""Lock, verify and open the blind v2 suite exactly once per final winner."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.quality import fingerprint_text, normalize_text  # noqa: E402
from src.pipeline.selection_policy import (  # noqa: E402
    configured_selection_hashes,
    selection_policy_record,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_tree_manifest(adapter: Path) -> dict[str, Any]:
    adapter = adapter.resolve()
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Blind adapter is incomplete: {adapter}")
    files = []
    for path in sorted(adapter.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Blind adapter cannot contain symlinks: {path}")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(adapter).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    if not files:
        raise ValueError(f"Blind adapter is empty: {adapter}")
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "root": str(adapter),
        "file_count": len(files),
        "bytes": sum(int(item["bytes"]) for item in files),
        "manifest_sha256": digest,
        "files": files,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def leakage_values(task: str, rows: list[dict[str, Any]]) -> set[str]:
    """Derive comparison keys from payloads so absent metadata cannot hide overlap."""
    result: set[str] = set()
    for index, row in enumerate(rows, start=1):
        identifier = str(row.get("id") or "").strip()
        if identifier:
            result.add(f"id:{identifier}")
        if task == "mt":
            source = normalize_text(row.get("source_text"))
            target = normalize_text(row.get("target_text"))
            if not source or not target:
                raise ValueError(
                    f"MT leakage comparison row {index} lacks source_text or target_text"
                )
            result.add(f"pair_fingerprint:{fingerprint_text(source + chr(31) + target)}")
            continue

        text_value = normalize_text(row.get("text"))
        if not text_value:
            raise ValueError(f"ASR leakage comparison row {index} lacks text")
        result.add(f"text_fingerprint:{fingerprint_text(text_value)}")
        audio_digest = str(row.get("audio_sha256") or "").strip()
        if not audio_digest:
            audio_value = str(row.get("audio_path") or "").strip()
            if not audio_value:
                raise ValueError(
                    f"ASR leakage comparison row {index} lacks audio_sha256 and audio_path"
                )
            audio_path = Path(audio_value)
            if not audio_path.is_absolute():
                audio_path = ROOT / audio_path
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"ASR leakage comparison audio is missing on row {index}: {audio_path}"
                )
            audio_digest = sha256(audio_path)
        result.add(f"audio_sha256:{audio_digest}")
        for key in ("speaker", "group"):
            value = str(row.get(key) or "").strip()
            if value:
                result.add(f"{key}:{value}")
    return result


def validate_identifiers(task: str, rows: list[dict[str, Any]]) -> None:
    """Fail closed when blind rows cannot be audited for leakage or duplicates."""
    required = (
        ("id", "pair_fingerprint")
        if task == "mt"
        else ("id", "text_fingerprint", "audio_sha256", "speaker", "group")
    )
    unique = ("id", "pair_fingerprint") if task == "mt" else ("id", "audio_sha256")
    for key in required:
        missing = [
            index + 1
            for index, row in enumerate(rows)
            if not str(row.get(key) or "").strip()
        ]
        if missing:
            preview = ", ".join(str(index) for index in missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise ValueError(
                f"Blind {task} manifest is missing {key} on rows {preview}{suffix}"
            )
    for key in unique:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for row in rows:
            value = str(row[key]).strip()
            if value in seen:
                duplicates.add(value)
            seen.add(value)
        if duplicates:
            raise ValueError(
                f"Blind {task} manifest contains duplicate {key}: {len(duplicates)} values"
            )


def validate_content_integrity(task: str, rows: list[dict[str, Any]]) -> None:
    """Recompute blind fingerprints from payloads instead of trusting metadata."""
    for index, row in enumerate(rows, start=1):
        if task == "mt":
            if row.get("source_language") != "en" or row.get("target_language") != "vi":
                raise ValueError(
                    f"Blind mt row {index} must use canonical source_language=en and "
                    "target_language=vi"
                )
            source = normalize_text(row.get("source_text"))
            target = normalize_text(row.get("target_text"))
            if not source or not target:
                raise ValueError(
                    f"Blind mt manifest is missing source_text or target_text on row {index}"
                )
            expected = fingerprint_text(source + "\x1f" + target)
            if row["pair_fingerprint"] != expected:
                raise ValueError(
                    f"Blind mt pair_fingerprint does not match row content on row {index}"
                )
            continue

        if not str(row.get("language") or "").startswith("vi"):
            raise ValueError(f"Blind asr row {index} must declare a Vietnamese language")
        text_value = normalize_text(row.get("text"))
        if not text_value:
            raise ValueError(f"Blind asr manifest is missing text on row {index}")
        if row["text_fingerprint"] != fingerprint_text(text_value):
            raise ValueError(
                f"Blind asr text_fingerprint does not match row content on row {index}"
            )

        audio_value = str(row.get("audio_path") or "").strip()
        if not audio_value:
            raise ValueError(f"Blind asr manifest is missing audio_path on row {index}")
        audio_path = Path(audio_value)
        if not audio_path.is_absolute():
            audio_path = ROOT / audio_path
        if not audio_path.is_file():
            raise FileNotFoundError(f"Blind asr audio is missing on row {index}: {audio_path}")
        if row["audio_sha256"] != sha256(audio_path):
            raise ValueError(
                f"Blind asr audio_sha256 does not match audio content on row {index}"
            )


def validate_required_slices(task: str, rows: list[dict[str, Any]], required: list[str]) -> None:
    observed = set(coverage_counts(task, rows)["slice_samples"])
    missing = sorted(set(required) - observed)
    if missing:
        raise ValueError(f"Blind {task} manifest is missing required slices: {missing}")


def coverage_counts(task: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count each declared blind slice at most once per independent row."""
    counts: Counter[str] = Counter()
    for row in rows:
        observed = {
            str(value).strip().casefold()
            for value in row.get("categories", [])
            if str(value).strip()
        }
        if task == "mt":
            if row.get("source_text") and row.get("target_text"):
                observed.update({"en_to_vi", "vi_to_en"})
            else:
                direction = str(row.get("direction") or "").strip().casefold()
                if direction:
                    observed.add(direction)
        else:
            dimensions = row.get("blind_dimensions") or row.get("selection_dimensions") or {}
            accent = str(dimensions.get("accent_region") or "").strip().casefold()
            role = str(dimensions.get("role") or "").strip().casefold()
            if accent:
                observed.add(accent)
            if role:
                observed.add(role)
            if dimensions.get("noise") is True:
                observed.add("noise")
        counts.update(observed)
    coverage: dict[str, Any] = {
        "rows": len(rows),
        "slice_samples": dict(sorted(counts.items())),
    }
    if task == "asr":
        coverage.update(
            {
                "unique_speakers": len(
                    {
                        str(row.get("speaker") or "").strip()
                        for row in rows
                        if str(row.get("speaker") or "").strip()
                    }
                ),
                "unique_groups": len(
                    {
                        str(row.get("group") or "").strip()
                        for row in rows
                        if str(row.get("group") or "").strip()
                    }
                ),
            }
        )
    return coverage


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Blind coverage policy {label} must be a positive integer")
    return value


def validate_minimum_coverage(
    task: str,
    rows: list[dict[str, Any]],
    policy: dict[str, Any],
    required_slices: list[str],
) -> dict[str, Any]:
    """Fail closed when a blind suite is too small for meaningful slice evidence."""
    if not isinstance(policy, dict):
        raise ValueError(f"Blind {task} minimum coverage policy is missing")
    minimum_rows = _positive_int(policy.get("rows"), f"{task}.rows")
    minimum_slices = policy.get("slice_samples")
    if not isinstance(minimum_slices, dict):
        raise ValueError(f"Blind coverage policy {task}.slice_samples is missing")
    missing_policy = sorted(set(required_slices) - set(minimum_slices))
    if missing_policy:
        raise ValueError(
            f"Blind coverage policy {task} lacks required slice quotas: {missing_policy}"
        )

    normalized_minimums = {
        str(name).strip().casefold(): _positive_int(
            minimum, f"{task}.slice_samples.{name}"
        )
        for name, minimum in minimum_slices.items()
        if str(name).strip()
    }
    coverage = coverage_counts(task, rows)
    deficits: list[str] = []
    if coverage["rows"] < minimum_rows:
        deficits.append(f"rows={coverage['rows']}/{minimum_rows}")
    if task == "asr":
        for name in ("unique_speakers", "unique_groups"):
            if name not in policy:
                continue
            minimum = _positive_int(policy[name], f"{task}.{name}")
            if coverage[name] < minimum:
                deficits.append(f"{name}={coverage[name]}/{minimum}")
    observed = coverage["slice_samples"]
    deficits.extend(
        f"{name}={observed.get(name, 0)}/{minimum}"
        for name, minimum in sorted(normalized_minimums.items())
        if observed.get(name, 0) < minimum
    )
    if deficits:
        raise ValueError(
            f"Blind {task} manifest is below minimum coverage: {', '.join(deficits)}"
        )
    return coverage


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def confidence_interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    lower = finite_float(value[0])
    upper = finite_float(value[1])
    if lower is None or upper is None or lower > upper:
        return None
    return lower, upper


def clinical_failures(report: dict[str, Any], required: list[str]) -> list[str]:
    categories = report.get("categories") or {}
    failures = []
    for name in required:
        record = categories.get(name)
        try:
            samples = int(record.get("samples", 0)) if record else 0
        except (TypeError, ValueError):
            samples = 0
        rate = finite_float(record.get("safety_failure_rate")) if record else None
        if samples < 1 or rate is None or rate != 0.0:
            failures.append(name)
    return failures


def _reported_samples(record: Any) -> int | None:
    try:
        samples = int(record.get("samples")) if isinstance(record, dict) else None
    except (TypeError, ValueError):
        return None
    return samples if samples is not None and samples >= 0 else None


def _casefold_record(records: Any, name: str) -> dict[str, Any] | None:
    if not isinstance(records, dict):
        return None
    target = name.casefold()
    for key, value in records.items():
        if str(key).casefold() == target and isinstance(value, dict):
            return value
    return None


def blind_report_coverage_failures(
    report: dict[str, Any],
    task: str,
    direction: str | None,
    expected_rows: int,
    expected_slices: dict[str, int],
    expected_unique: dict[str, int] | None = None,
) -> list[str]:
    """Bind report sample counts to the immutable blind manifest coverage."""
    metric_view = report
    if task == "mt":
        metric_view = _casefold_record(report.get("directions"), str(direction)) or {}
    failures: list[str] = []
    actual_rows = _reported_samples(metric_view)
    if actual_rows != expected_rows:
        failures.append(f"rows:{actual_rows}/{expected_rows}")
    if task == "asr" and expected_unique:
        for name, expected in sorted(expected_unique.items()):
            try:
                actual = int(report.get(name))
            except (TypeError, ValueError):
                actual = None
            if actual != expected:
                failures.append(f"{name}:{actual}/{expected}")
        if report.get("wer_bootstrap_unit") != "group":
            failures.append("wer_bootstrap_unit:not_group")
        try:
            clusters = int(report.get("wer_bootstrap_clusters"))
        except (TypeError, ValueError):
            clusters = None
        expected_groups = expected_unique.get("unique_groups")
        if clusters != expected_groups:
            failures.append(f"wer_bootstrap_clusters:{clusters}/{expected_groups}")

    categories = report.get("categories") or {}
    slices = report.get("slices") or {}
    for name, expected in sorted(expected_slices.items()):
        if task == "mt" and name in {"en_to_vi", "vi_to_en"}:
            if name != direction:
                continue
            record = _casefold_record(report.get("directions"), name)
        elif task == "asr" and name in {"north", "central", "south"}:
            record = _casefold_record(slices.get("accent"), name)
        elif task == "asr" and name in {"doctor", "patient"}:
            record = _casefold_record(slices.get("role"), name)
        elif task == "asr" and name == "noise":
            record = _casefold_record(slices.get("noise"), "true")
        else:
            record = _casefold_record(categories, name)
        actual = _reported_samples(record)
        if actual != expected:
            failures.append(f"slice:{name}:{actual}/{expected}")
    return failures


def blind_quality_failures(
    report: dict[str, Any],
    task: str,
    direction: str | None,
    accuracy_config: dict[str, Any],
    bakeoff_config: dict[str, Any],
) -> list[str]:
    """Require point estimates and worst-case 95% CI to clear release thresholds."""
    aggregate = accuracy_config["release_gates"]["aggregate"]
    failures: list[str] = []
    metric_view = report
    if task == "mt":
        metric_view = report.get("directions", {}).get(str(direction)) or {}
    try:
        samples = int(metric_view.get("samples", 0))
    except (TypeError, ValueError):
        samples = 0
    if samples < 1:
        failures.append("samples:missing")

    if task == "mt":
        policies = (
            ("sacrebleu", "sacrebleu_bootstrap_95ci", float(aggregate["mt_sacrebleu_min"])),
            ("chrf2", "chrf2_bootstrap_95ci", float(aggregate["mt_chrf2_min"])),
        )
        for metric_name, interval_name, minimum in policies:
            metric = finite_float(metric_view.get(metric_name))
            interval = confidence_interval(metric_view.get(interval_name))
            if metric is None:
                failures.append(f"{metric_name}:missing_or_invalid")
            elif metric < minimum:
                failures.append(f"{metric_name}:below_policy")
            if interval is None:
                failures.append(f"{interval_name}:missing_or_invalid")
            elif interval[0] < minimum:
                failures.append(f"{interval_name}:lower_bound_below_policy")
        return failures

    wer = finite_float(report.get("wer"))
    wer_interval = confidence_interval(report.get("wer_bootstrap_95ci"))
    wer_max = float(aggregate["asr_vi_wer_max"])
    if wer is None:
        failures.append("wer:missing_or_invalid")
    elif wer > wer_max:
        failures.append("wer:above_policy")
    if wer_interval is None:
        failures.append("wer_bootstrap_95ci:missing_or_invalid")
    elif wer_interval[1] > wer_max:
        failures.append("wer_bootstrap_95ci:upper_bound_above_policy")
    if finite_float(report.get("cer")) is None:
        failures.append("cer:missing_or_invalid")
    code_switch = report.get("slices", {}).get("code_switch", {}).get("True")
    code_switch_wer = finite_float(code_switch.get("wer")) if code_switch else None
    if code_switch_wer is None:
        failures.append("code_switch_wer:missing_or_invalid")
    elif code_switch_wer > float(
        bakeoff_config["promotion_gate"]["asr_code_switch_wer_max"]
    ):
        failures.append("code_switch_wer:above_policy")
    return failures


def load_locked_accuracy_config(config: dict[str, Any]) -> dict[str, Any]:
    record = config["data"]["accuracy_program"]
    path = (ROOT / record["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != record["sha256"]:
        raise ValueError(f"Accuracy policy checksum mismatch: {actual}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_unseen(task: str, blind_rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    blind_values = leakage_values(task, blind_rows)
    comparison_paths = [
        ROOT / config["data"]["train"][task],
        ROOT / config["data"]["selection_dev"][task]["path"],
    ]
    comparison_paths.extend(
        ROOT / path
        for path in config["data"]["forbidden_selection_inputs"]
        if (task == "mt" and "mt" in path) or (task == "asr" and "asr" in path)
    )
    for path in comparison_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        overlap = blind_values & leakage_values(task, read_jsonl(path))
        if overlap:
            raise ValueError(f"Blind {task} leakage against {path}: {len(overlap)} keys")


def lock_suite(lock_path: Path, mt_path: Path, asr_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    existing = json.loads(lock_path.read_text(encoding="utf-8"))
    if existing.get("version") != 2:
        raise ValueError("Blind v2 lock requires schema version 2")
    if existing.get("status") != "awaiting_unseen_data":
        raise ValueError("Blind v2 is already locked and cannot be replaced")
    manifests = {}
    for task, path in (("mt", mt_path), ("asr", asr_path)):
        resolved = path.resolve()
        rows = read_jsonl(resolved)
        if not rows:
            raise ValueError(f"Blind {task} manifest is empty")
        validate_identifiers(task, rows)
        validate_content_integrity(task, rows)
        ensure_unseen(task, rows, config)
        required_slices = existing["required_slices"][task]
        validate_required_slices(task, rows, required_slices)
        coverage = validate_minimum_coverage(
            task,
            rows,
            existing["minimum_coverage"][task],
            required_slices,
        )
        manifests[task] = {
            "path": str(resolved),
            "rows": len(rows),
            "sha256": sha256(resolved),
            "coverage": coverage,
        }
    locked = {
        **existing,
        "status": "locked_unopened",
        "locked_at": utc_now(),
        "manifests": manifests,
        "selection": None,
        "opened": {"mt": {"en_to_vi": None, "vi_to_en": None}, "asr": None},
    }
    atomic_json(lock_path, locked)
    return locked


def verify_lock(lock_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("version") != 2:
        raise ValueError("Blind v2 lock requires schema version 2")
    if lock.get("status") not in {"locked_unopened", "partially_opened", "opened"}:
        raise ValueError(f"Blind v2 is not locked: {lock.get('status')!r}")
    for task in ("mt", "asr"):
        record = lock["manifests"].get(task)
        if not record:
            raise ValueError(f"Blind lock lacks {task} manifest")
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Blind {task} checksum mismatch")
        rows = read_jsonl(path)
        coverage = validate_minimum_coverage(
            task,
            rows,
            lock["minimum_coverage"][task],
            lock["required_slices"][task],
        )
        if record.get("rows") != len(rows) or record.get("coverage") != coverage:
            raise ValueError(f"Blind {task} coverage record mismatch")
    selection = lock.get("selection")
    if selection:
        path = Path(selection["path"])
        if not path.is_file() or sha256(path) != selection["sha256"]:
            raise ValueError("Blind selection comparison checksum mismatch")
    return lock


def candidate_from_config(config: dict[str, Any], task: str, candidate_id: str) -> dict[str, Any]:
    for candidate in config["candidates"][task]:
        if candidate["id"] == candidate_id:
            return candidate
    raise ValueError(f"Unknown {task} candidate: {candidate_id}")


def verify_selection_winner(
    comparison_path: Path,
    task: str,
    direction: str | None,
    candidate_id: str,
    adapter: Path,
    expected_policy: dict[str, Any],
    expected_selection_hashes: dict[str, str],
) -> dict[str, Any]:
    comparison_path = comparison_path.resolve()
    if not comparison_path.is_file():
        raise FileNotFoundError(comparison_path)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("status") != "selection_complete":
        raise ValueError("Blind evaluation requires a finalized selection comparison")
    if comparison.get("selection_policy") != expected_policy:
        raise ValueError("Blind evaluation selection policy is missing, stale, or mismatched")
    if comparison.get("selection_sha256") != expected_selection_hashes:
        raise ValueError("Blind evaluation selection inputs are missing, stale, or mismatched")
    key = str(direction or "vi")
    try:
        winner = comparison["results"][task]["winners"][key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Selection comparison lacks winner for {task}/{key}") from exc
    if winner.get("candidate_id") != candidate_id:
        raise ValueError(f"Blind candidate is not the selected winner for {task}/{key}")
    adapter = adapter.resolve()
    if Path(str(winner.get("adapter") or "")).resolve() != adapter:
        raise ValueError(f"Blind adapter path does not match selected winner for {task}/{key}")
    manifest = adapter_tree_manifest(adapter)
    expected_manifest = str(winner.get("adapter_manifest_sha256") or "")
    if manifest["manifest_sha256"] != expected_manifest:
        raise ValueError(f"Blind adapter checksum does not match selected winner for {task}/{key}")
    return {
        "path": str(comparison_path),
        "sha256": sha256(comparison_path),
        "winner": winner,
        "adapter_manifest": manifest,
    }


def verify_report_provenance(
    report_path: Path,
    provenance_path: Path,
    specification_sha256: str,
    manifest_sha256: str,
    selection_sha256: str,
) -> dict[str, Any]:
    if not report_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(f"Blind report provenance is incomplete: {report_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {
        "specification_sha256": specification_sha256,
        "blind_manifest_sha256": manifest_sha256,
        "selection_sha256": selection_sha256,
        "report_sha256": sha256(report_path),
        "report_bytes": report_path.stat().st_size,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Blind report provenance mismatch: {report_path}")
    return provenance


def opening_slot(lock: dict[str, Any], task: str, direction: str | None) -> tuple[dict[str, Any], str]:
    if task == "mt":
        if direction not in {"en_to_vi", "vi_to_en"}:
            raise ValueError("MT blind evaluation requires one explicit direction")
        return lock["opened"]["mt"], str(direction)
    return lock["opened"], "asr"


def evaluate(
    args: argparse.Namespace, lock_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    lock = verify_lock(lock_path)
    candidate = candidate_from_config(config, args.task, args.candidate)
    if args.scope == "production" and not candidate["license"].get("production_eligible"):
        raise ValueError("Candidate license is not approved for production promotion")
    selection = verify_selection_winner(
        args.selection_comparison,
        args.task,
        args.direction,
        args.candidate,
        args.adapter,
        selection_policy_record(config),
        configured_selection_hashes(config),
    )
    selection_record = {
        "path": selection["path"],
        "sha256": selection["sha256"],
    }
    existing_selection = lock.get("selection")
    if existing_selection is not None and existing_selection != selection_record:
        raise ValueError("Blind suite was already bound to another selection comparison")
    slot, key = opening_slot(lock, args.task, args.direction)
    specification = {
        "candidate": args.candidate,
        "model": candidate["model"],
        "revision": candidate["revision"],
        "adapter": str(args.adapter.resolve()),
        "adapter_manifest_sha256": selection["adapter_manifest"]["manifest_sha256"],
        "selection_sha256": selection["sha256"],
        "blind_manifest_sha256": lock["manifests"][args.task]["sha256"],
        "direction": args.direction,
        "scope": args.scope,
    }
    digest = hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"blind_v2_{args.task}_{key}_{args.candidate}"
    report_path = output_dir / f"{stem}.json"
    provenance_path = output_dir / f"{stem}_provenance.json"
    if report_path.is_file():
        verify_report_provenance(
            report_path,
            provenance_path,
            digest,
            specification["blind_manifest_sha256"],
            selection["sha256"],
        )

    previous = slot.get(key)
    if previous and previous.get("candidate_sha256") != digest:
        raise ValueError(f"Blind slot {args.task}/{key} was already opened for another candidate")
    if previous is None:
        lock["selection"] = selection_record
        slot[key] = {**specification, "candidate_sha256": digest, "opened_at": utc_now()}
        lock["status"] = "partially_opened"
        atomic_json(lock_path, lock)

    if not report_path.is_file():
        command = [
            args.python,
            str(ROOT / "scripts/run_baseline_benchmarks.py"),
            "--task",
            args.task,
            "--model",
            candidate["model"],
            "--model-revision",
            candidate["revision"],
            "--adapter",
            str(args.adapter.resolve()),
            "--manifest",
            lock["manifests"][args.task]["path"],
            "--samples",
            "0",
            "--device",
            "cuda",
            "--precision",
            "bf16",
            "--gpu-memory-fraction",
            "0.35",
            "--batch-size",
            "4",
            "--num-beams",
            "1",
            "--name",
            stem,
            "--output-dir",
            str(output_dir),
            "--resume-scoring",
        ]
        if args.task == "mt":
            command += [
                "--mt-model-family",
                candidate["model_family"],
                "--mt-direction",
                str(args.direction),
            ]
        else:
            command += ["--language", "vi"]
        from scripts.run_gpu_rounds import monitor_process, wait_for_gpu_spawn_capacity

        log_path = output_dir / f"{stem}.log"
        spawn_capacity = wait_for_gpu_spawn_capacity(
            config["resources"],
            output_dir / f"{stem}_pre_spawn.jsonl",
        )
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            monitored = monitor_process(
                process,
                output_dir / f"{stem}_resources.jsonl",
                config["resources"],
            )
        monitored["spawn_capacity"] = spawn_capacity
        if monitored["return_code"]:
            raise RuntimeError(
                f"Blind benchmark failed with {monitored['return_code']}; see {log_path}"
            )
        if not report_path.is_file():
            raise FileNotFoundError(f"Blind benchmark did not create report: {report_path}")
        atomic_json(
            provenance_path,
            {
                "version": 1,
                "created_at": utc_now(),
                "specification_sha256": digest,
                "blind_manifest_sha256": specification["blind_manifest_sha256"],
                "selection_sha256": selection["sha256"],
                "report": str(report_path),
                "report_bytes": report_path.stat().st_size,
                "report_sha256": sha256(report_path),
                "resource_run": monitored,
            },
        )
    provenance = verify_report_provenance(
        report_path,
        provenance_path,
        digest,
        specification["blind_manifest_sha256"],
        selection["sha256"],
    )
    if adapter_tree_manifest(args.adapter)["manifest_sha256"] != specification[
        "adapter_manifest_sha256"
    ]:
        raise ValueError("Blind adapter changed during evaluation")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = config["promotion_gate"]["critical_slices"] + config["promotion_gate"]["policy_slices"]
    failed = clinical_failures(report, required)
    manifest_record = lock["manifests"][args.task]
    expected_slices = {
        name: int(manifest_record["coverage"]["slice_samples"][name])
        for name in lock["required_slices"][args.task]
    }
    coverage_failed = blind_report_coverage_failures(
        report,
        args.task,
        args.direction,
        int(manifest_record["rows"]),
        expected_slices,
        {
            name: int(manifest_record["coverage"][name])
            for name in ("unique_speakers", "unique_groups")
            if name in manifest_record["coverage"]
        }
        if args.task == "asr"
        else None,
    )
    accuracy_config = load_locked_accuracy_config(config)
    quality_failed = blind_quality_failures(
        report,
        args.task,
        args.direction,
        accuracy_config,
        config,
    )
    result = {
        "evaluated_at": utc_now(),
        "candidate": specification,
        "manifest_sha256": lock["manifests"][args.task]["sha256"],
        "selection_sha256": selection["sha256"],
        "report": str(report_path),
        "report_bytes": provenance["report_bytes"],
        "report_sha256": provenance["report_sha256"],
        "report_provenance": str(provenance_path),
        "critical_gate": "pass" if not failed else "fail",
        "failed_slices": failed,
        "coverage_gate": "pass" if not coverage_failed else "fail",
        "coverage_failures": coverage_failed,
        "quality_gate": "pass" if not quality_failed else "fail",
        "quality_failures": quality_failed,
        "promotion_allowed": not failed and not coverage_failed and not quality_failed,
    }
    atomic_json(output_dir / f"{stem}_gate.json", result)
    slot[key]["result"] = result
    mt_done = all(lock["opened"]["mt"].values())
    lock["status"] = "opened" if mt_done and lock["opened"]["asr"] else "partially_opened"
    atomic_json(lock_path, lock)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["lock", "verify", "evaluate"], required=True)
    parser.add_argument("--lock", type=Path, default=ROOT / "data/eval/blind_test_v2.lock.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/model_bakeoff.yaml")
    parser.add_argument("--mt-manifest", type=Path)
    parser.add_argument("--asr-manifest", type=Path)
    parser.add_argument("--task", choices=["mt", "asr"])
    parser.add_argument("--direction", choices=["en_to_vi", "vi_to_en"])
    parser.add_argument("--candidate")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--selection-comparison", type=Path)
    parser.add_argument("--scope", choices=["research", "production"], default="research")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/reports/model_bakeoff/blind_v2")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.action == "lock":
        if not args.mt_manifest or not args.asr_manifest:
            parser.error("--action lock requires --mt-manifest and --asr-manifest")
        result = lock_suite(args.lock, args.mt_manifest, args.asr_manifest, config)
    elif args.action == "verify":
        result = verify_lock(args.lock)
    else:
        if not args.task or not args.candidate or not args.adapter or not args.selection_comparison:
            parser.error(
                "--action evaluate requires --task, --candidate, --adapter and "
                "--selection-comparison"
            )
        if not args.adapter.is_dir():
            parser.error("--adapter must be a directory")
        result = evaluate(args, args.lock, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

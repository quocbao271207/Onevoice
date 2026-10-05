"""Lock, verify and open the blind v2 suite exactly once per final winner."""

from __future__ import annotations

import argparse
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def values(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> set[str]:
    result = set()
    for row in rows:
        for key in keys:
            if row.get(key):
                result.add(f"{key}:{row[key]}")
    return result


def validate_identifiers(task: str, rows: list[dict[str, Any]]) -> None:
    """Fail closed when blind rows cannot be audited for leakage or duplicates."""
    required = (
        ("id", "pair_fingerprint")
        if task == "mt"
        else ("id", "text_fingerprint", "audio_sha256")
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
    observed = set()
    for row in rows:
        observed.update(str(value) for value in row.get("categories", []))
        if task == "mt":
            if row.get("source_text") and row.get("target_text"):
                observed.update({"en_to_vi", "vi_to_en"})
            else:
                observed.add(str(row.get("direction") or ""))
        else:
            dimensions = row.get("blind_dimensions") or row.get("selection_dimensions") or {}
            observed.add(str(dimensions.get("accent_region") or ""))
            observed.add(str(dimensions.get("role") or ""))
            if dimensions.get("noise"):
                observed.add("noise")
    missing = sorted(set(required) - observed)
    if missing:
        raise ValueError(f"Blind {task} manifest is missing required slices: {missing}")


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


def ensure_unseen(task: str, blind_rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    keys = ("id", "pair_fingerprint") if task == "mt" else (
        "id",
        "text_fingerprint",
        "audio_sha256",
    )
    blind_values = values(blind_rows, keys)
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
        overlap = blind_values & values(read_jsonl(path), keys)
        if overlap:
            raise ValueError(f"Blind {task} leakage against {path}: {len(overlap)} keys")


def lock_suite(lock_path: Path, mt_path: Path, asr_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    existing = json.loads(lock_path.read_text(encoding="utf-8"))
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
        validate_required_slices(task, rows, existing["required_slices"][task])
        manifests[task] = {
            "path": str(resolved),
            "rows": len(rows),
            "sha256": sha256(resolved),
        }
    locked = {
        **existing,
        "status": "locked_unopened",
        "locked_at": utc_now(),
        "manifests": manifests,
        "opened": {"mt": {"en_to_vi": None, "vi_to_en": None}, "asr": None},
    }
    atomic_json(lock_path, locked)
    return locked


def verify_lock(lock_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") not in {"locked_unopened", "partially_opened", "opened"}:
        raise ValueError(f"Blind v2 is not locked: {lock.get('status')!r}")
    for task in ("mt", "asr"):
        record = lock["manifests"].get(task)
        if not record:
            raise ValueError(f"Blind lock lacks {task} manifest")
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Blind {task} checksum mismatch")
    return lock


def candidate_from_config(config: dict[str, Any], task: str, candidate_id: str) -> dict[str, Any]:
    for candidate in config["candidates"][task]:
        if candidate["id"] == candidate_id:
            return candidate
    raise ValueError(f"Unknown {task} candidate: {candidate_id}")


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
    slot, key = opening_slot(lock, args.task, args.direction)
    specification = {
        "candidate": args.candidate,
        "model": candidate["model"],
        "revision": candidate["revision"],
        "adapter": str(args.adapter.resolve()),
        "direction": args.direction,
        "scope": args.scope,
    }
    digest = hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    previous = slot.get(key)
    if previous and previous.get("candidate_sha256") != digest:
        raise ValueError(f"Blind slot {args.task}/{key} was already opened for another candidate")
    if previous is None:
        slot[key] = {**specification, "candidate_sha256": digest, "opened_at": utc_now()}
        lock["status"] = "partially_opened"
        atomic_json(lock_path, lock)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"blind_v2_{args.task}_{key}_{args.candidate}"
    report_path = output_dir / f"{stem}.json"
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
        from scripts.run_gpu_rounds import monitor_process

        log_path = output_dir / f"{stem}.log"
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
        if monitored["return_code"]:
            raise RuntimeError(
                f"Blind benchmark failed with {monitored['return_code']}; see {log_path}"
            )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = config["promotion_gate"]["critical_slices"] + config["promotion_gate"]["policy_slices"]
    failed = clinical_failures(report, required)
    accuracy_path = ROOT / config["data"]["accuracy_program"]
    if not accuracy_path.is_file():
        raise FileNotFoundError(accuracy_path)
    accuracy_config = yaml.safe_load(accuracy_path.read_text(encoding="utf-8"))
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
        "report": str(report_path),
        "critical_gate": "pass" if not failed else "fail",
        "failed_slices": failed,
        "quality_gate": "pass" if not quality_failed else "fail",
        "quality_failures": quality_failed,
        "promotion_allowed": not failed and not quality_failed,
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
        if not args.task or not args.candidate or not args.adapter:
            parser.error("--action evaluate requires --task, --candidate and --adapter")
        if not args.adapter.is_dir():
            parser.error("--adapter must be a directory")
        result = evaluate(args, args.lock, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

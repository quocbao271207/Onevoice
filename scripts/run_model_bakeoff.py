"""Run the post-Candidate-A multi-model bake-off without touching the active job.

The runner is fail-closed and resumable.  It first verifies that the existing
GPU program is complete, freezes Candidate A by hashing its artifacts, then
runs one GPU child at a time through zero-shot, 400-step pilot, 2,000-step
semifinal and full-train stages.  The current locked tests are never used for
model selection; only the deterministic selection-dev manifests are opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CRITICAL_EXIT_WAITING_FOR_BLIND = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def command_digest(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise ValueError("Unsupported model-bakeoff config version")
    return config


def license_gate(candidate: dict[str, Any], research_approvals: set[str]) -> tuple[bool, str]:
    license_info = candidate.get("license") or {}
    if license_info.get("gpu_eligible") and license_info.get("review_status") in {
        "approved",
        "approved_research_only",
    }:
        return True, "approved"
    if (
        candidate["id"] in research_approvals
        and license_info.get("research_reference_allowed")
        and license_info.get("review_status") == "legal_review_required"
    ):
        return True, "explicit_research_approval"
    return False, str(license_info.get("review_status") or "missing_license_review")


def validate_resources(config: dict[str, Any]) -> None:
    resources = config["resources"]
    if not 0 < float(resources["gpu_memory_fraction"]) <= 0.35:
        raise ValueError("Bake-off GPU memory fraction must stay at or below 35%")
    if not 0 < float(resources["gpu_memory_hard_fraction"]) <= 0.40:
        raise ValueError("Hard GPU memory fraction must stay at or below 40%")
    resume = float(resources["resume_percent"])
    rolling = float(resources["utilization_percent"])
    hard = float(resources["hard_utilization_percent"])
    if not 0 < resume < rolling < hard < 75:
        raise ValueError("GPU utilization policy must satisfy 0 < resume < rolling < hard < 75")
    if not config["principles"].get("one_gpu_child_at_a_time"):
        raise ValueError("Bake-off requires one GPU child at a time")


def validate_selection_artifacts(config: dict[str, Any]) -> dict[str, str]:
    hashes = {}
    forbidden = {(ROOT / value).resolve() for value in config["data"]["forbidden_selection_inputs"]}
    for task in ("mt", "asr"):
        record = config["data"]["selection_dev"][task]
        path = (ROOT / record["path"]).resolve()
        if path in forbidden:
            raise ValueError(f"Locked test cannot be used for selection: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"Selection checksum mismatch for {task}: {actual}")
        hashes[task] = actual
    return hashes


def validate_candidate_matrix(config: dict[str, Any]) -> None:
    ids = []
    for task in ("mt", "asr"):
        candidates = config["candidates"][task]
        if sum(candidate.get("role") == "candidate_a" for candidate in candidates) != 1:
            raise ValueError(f"{task} must have exactly one Candidate A")
        for candidate in candidates:
            ids.append(candidate["id"])
            if len(str(candidate.get("revision") or "")) != 40:
                raise ValueError(f"Candidate must pin a 40-character revision: {candidate['id']}")
            if candidate.get("status") not in {
                "planned",
                "in_progress",
                "waiting_for_current_pipeline",
            }:
                raise ValueError(f"Invalid pre-benchmark status: {candidate['id']}")
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique")


def find_current_program_state(config: dict[str, Any], explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    matches = sorted(
        ROOT.glob(config["prerequisite"]["current_program_state_glob"]),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError("No current GPU program state found")
    return matches[-1]


def wait_for_current_program(path: Path, poll_seconds: float, should_wait: bool) -> dict[str, Any]:
    while True:
        state = json.loads(path.read_text(encoding="utf-8"))
        status = state.get("execution_status")
        if status == "complete":
            return state
        if status == "error":
            raise RuntimeError(f"Current GPU program failed: {state.get('error')}")
        if not should_wait:
            raise RuntimeError(f"Current GPU program is not complete: {status!r}")
        time.sleep(poll_seconds)


def tree_manifest(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": str(item.relative_to(path)),
                "bytes": item.stat().st_size,
                "sha256": sha256(item),
            }
        )
    if not files:
        raise ValueError(f"Cannot freeze empty candidate directory: {path}")
    manifest_digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"root": str(path.resolve()), "files": files, "manifest_sha256": manifest_digest}


def freeze_candidate_a(current: dict[str, Any], output: Path) -> dict[str, Any]:
    adapters = {"mt": current.get("mt_adapter"), "asr": current.get("asr_adapter")}
    missing = [task for task, value in adapters.items() if not value]
    if missing:
        raise ValueError(f"Current program lacks Candidate A adapters: {missing}")
    frozen = {
        "created_at": utc_now(),
        "source_program_state": current,
        "promotion_allowed": False,
        "policy": "Candidate A is a frozen reference; passing its old gate does not promote it.",
        "adapters": {task: tree_manifest(Path(value)) for task, value in adapters.items()},
    }
    atomic_json(output, frozen)
    return frozen


def critical_safety_pass(report: dict[str, Any], required: list[str]) -> tuple[bool, list[str]]:
    categories = report.get("categories") or {}
    failures = []
    for name in required:
        record = categories.get(name)
        try:
            samples = int(record.get("samples", 0)) if record else 0
        except (TypeError, ValueError):
            samples = 0
        if not record or samples < 1:
            failures.append(f"{name}:missing")
            continue
        try:
            failure_rate = float(record.get("safety_failure_rate", 1.0))
        except (TypeError, ValueError):
            failure_rate = math.nan
        if not math.isfinite(failure_rate):
            failures.append(f"{name}:invalid")
        elif failure_rate != 0.0:
            failures.append(f"{name}:nonzero")
    return not failures, failures


def normalized_interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        lower, upper = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        return None
    return lower, upper


def interval_stronger(
    challenger: Any, baseline: Any, *, greater_is_better: bool
) -> bool:
    challenger_bounds = normalized_interval(challenger)
    baseline_bounds = normalized_interval(baseline)
    if challenger_bounds is None or baseline_bounds is None:
        return False
    if greater_is_better:
        return challenger_bounds[0] > baseline_bounds[1]
    return challenger_bounds[1] < baseline_bounds[0]


def intervals_overlap(left: Any, right: Any) -> bool:
    left_bounds = normalized_interval(left)
    right_bounds = normalized_interval(right)
    if left_bounds is None or right_bounds is None:
        return False
    return max(left_bounds[0], right_bounds[0]) <= min(left_bounds[1], right_bounds[1])


def run_stage(
    name: str,
    command: list[str],
    state_path: Path,
    state: dict[str, Any],
    log_path: Path,
    expected_outputs: list[Path],
    resource_limits: dict[str, Any] | None = None,
) -> None:
    digest = command_digest(command)
    previous = state.setdefault("stages", {}).get(name)
    if previous and previous.get("status") == "complete":
        if previous.get("command_sha256") != digest:
            raise ValueError(f"Cannot resume {name}: command changed")
        if not all(path.exists() for path in expected_outputs):
            raise FileNotFoundError(f"Cannot resume {name}: expected output is missing")
        return
    state["stage"] = name
    state["stages"][name] = {
        "status": "running",
        "started_at": utc_now(),
        "command": command,
        "command_sha256": digest,
        "log": str(log_path),
    }
    atomic_json(state_path, state)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        if resource_limits is None:
            return_code = process.wait()
        else:
            from scripts.run_gpu_rounds import monitor_process

            monitor_path = log_path.with_suffix(".resources.jsonl")
            monitor = monitor_process(process, monitor_path, resource_limits)
            return_code = int(monitor["return_code"])
    record = state["stages"][name]
    record.update({"return_code": return_code, "completed_at": utc_now()})
    if return_code or not all(path.exists() for path in expected_outputs):
        record["status"] = "error"
        atomic_json(state_path, state)
        raise RuntimeError(f"Stage {name} failed; see {log_path}")
    record["status"] = "complete"
    atomic_json(state_path, state)


def benchmark_command(
    python: str,
    task: str,
    candidate: dict[str, Any],
    manifest: Path,
    output_dir: Path,
    stem: str,
    adapter: Path | None,
    direction: str | None,
) -> list[str]:
    command = [
        python,
        str(ROOT / "scripts/run_baseline_benchmarks.py"),
        "--task",
        task,
        "--model",
        candidate["model"],
        "--model-revision",
        candidate["revision"],
        "--manifest",
        str(manifest),
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
    if adapter:
        command += ["--adapter", str(adapter)]
    if task == "mt":
        command += [
            "--mt-model-family",
            candidate["model_family"],
            "--mt-direction",
            str(direction or "joint"),
        ]
    else:
        command += ["--language", "vi"]
    return command


def candidate_units(config: dict[str, Any], task: str) -> list[dict[str, Any]]:
    units = []
    for candidate in config["candidates"][task]:
        if candidate["role"] == "candidate_a":
            continue
        directions = candidate.get("directions") or [None]
        for direction in directions:
            unit = deepcopy(candidate)
            unit["direction"] = direction
            unit["unit_id"] = candidate["id"] + (f"__{direction}" if direction else "")
            units.append(unit)
    return units


def write_runtime_round_config(
    path: Path,
    config: dict[str, Any],
    task: str,
    candidate: dict[str, Any],
    direction: str | None,
    round_name: str,
    steps: int,
) -> None:
    halving = config["successive_halving"]
    limit_train = (
        int(halving["pilot"]["max_train_samples"])
        if round_name == "pilot"
        else 0
    )
    common = {
        "base_model": candidate["model"],
        "base_model_revision": candidate["revision"],
        "train_manifest": str(ROOT / config["data"]["train"][task]),
        "validation_manifest": str(ROOT / config["data"]["selection_dev"][task]["path"]),
        "method": "lora",
        "precision": "bf16",
        "batch_size": 2 if task == "mt" else 1,
        "eval_batch_size": 2 if task == "mt" else 1,
        "gradient_accumulation_steps": 16 if task == "mt" else 32,
        "clinical_oversample_factor": 2,
        "limit_train": limit_train,
        "limit_validation": 0,
        "eval_steps": 100 if steps > 0 else 250,
        "save_steps": 100 if steps > 0 else 250,
        "logging_steps": 25,
    }
    if task == "mt":
        common.update({"model_family": candidate["model_family"], "direction": direction})
    runtime = {
        "version": 1,
        "limits": config["resources"],
        "tasks": {
            task: {
                "module": (
                    "src.training.finetune_mt_medical"
                    if task == "mt"
                    else "src.training.finetune_whisper_vi"
                ),
                "selection_metric": "eval_loss" if task == "mt" else "eval_wer",
                "greater_is_better": False,
                "common_args": common,
                "rounds": [
                    {
                        "name": round_name,
                        "max_steps": steps,
                        "epochs": 1.0 if task == "mt" else 5.0,
                        "learning_rate": 1.0e-4,
                        "lora_rank": 8,
                        "lora_alpha": 16,
                        "lora_dropout": 0.05,
                    }
                ],
            }
        },
    }
    rendered = yaml.safe_dump(runtime, sort_keys=False, allow_unicode=True)
    if path.is_file() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Runtime config changed during resume: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def completed_adapter(output_root: Path, task: str) -> Path:
    runs = sorted(output_root.glob(f"{task}-*"), key=lambda path: path.stat().st_mtime)
    complete = []
    for run in runs:
        summary_path = run / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "complete":
            complete.append((run, summary))
    if len(complete) != 1:
        raise ValueError(f"Expected one complete {task} run under {output_root}, found {len(complete)}")
    run, summary = complete[0]
    selected = str(summary["selected_round"])
    matches = list(run.glob(f"[0-9][0-9]-{selected}/model"))
    if len(matches) != 1 or not (matches[0] / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Completed adapter missing under {run}")
    return matches[0]


def train_unit(
    *,
    unit: dict[str, Any],
    task: str,
    round_name: str,
    steps: int,
    config: dict[str, Any],
    python: str,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> Path:
    unit_id = unit["unit_id"]
    output_root = state_dir / "training" / task / unit_id / round_name
    runtime_config = state_dir / "runtime-configs" / f"{task}-{unit_id}-{round_name}.yaml"
    write_runtime_round_config(
        runtime_config,
        config,
        task,
        unit,
        unit.get("direction"),
        round_name,
        steps,
    )
    command = [
        python,
        str(ROOT / "scripts/run_gpu_rounds.py"),
        "--task",
        task,
        "--config",
        str(runtime_config),
        "--output-root",
        str(output_root),
    ]
    stage = f"{task}_{unit_id}_{round_name}_train"
    run_stage(
        stage,
        command,
        state_path,
        state,
        state_dir / "logs" / f"{stage}.log",
        [output_root],
    )
    return completed_adapter(output_root, task)


def benchmark_unit(
    *,
    unit: dict[str, Any],
    task: str,
    label: str,
    adapter: Path | None,
    config: dict[str, Any],
    python: str,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
    direction: str | None,
) -> dict[str, Any]:
    output_dir = state_dir / "selection-reports" / task / unit["unit_id"] / label
    stem = "selection"
    report_path = output_dir / f"{stem}.json"
    command = benchmark_command(
        python,
        task,
        unit,
        ROOT / config["data"]["selection_dev"][task]["path"],
        output_dir,
        stem,
        adapter,
        direction,
    )
    stage = f"{task}_{unit['unit_id']}_{label}_selection"
    run_stage(
        stage,
        command,
        state_path,
        state,
        state_dir / "logs" / f"{stage}.log",
        [report_path],
        resource_limits=config["resources"],
    )
    return json.loads(report_path.read_text(encoding="utf-8"))


def report_score(
    report: dict[str, Any], task: str, direction: str | None, required_slices: list[str]
) -> dict[str, Any]:
    safety_pass, failures = critical_safety_pass(report, required_slices)
    metric_view = report
    if task == "mt" and direction:
        metric_view = report.get("directions", {}).get(direction) or {}
    if task == "mt":
        metric_name = "chrf2"
        metric = metric_view.get(metric_name)
        interval = metric_view.get("chrf2_bootstrap_95ci")
        greater_is_better = True
    else:
        metric_name = "wer"
        metric = metric_view.get(metric_name)
        interval = metric_view.get("wer_bootstrap_95ci")
        greater_is_better = False
        code_switch = report.get("slices", {}).get("code_switch", {}).get("True")
        if code_switch is None:
            safety_pass = False
            failures.append("code_switch_wer:missing")
        else:
            try:
                code_switch_wer = float(code_switch.get("wer", 1.0))
            except (TypeError, ValueError):
                code_switch_wer = math.nan
            if not math.isfinite(code_switch_wer):
                safety_pass = False
                failures.append("code_switch_wer:missing_or_invalid")
            elif code_switch_wer > 0.21:
                safety_pass = False
                failures.append("code_switch_wer:above_policy")

    clinical_safety_pass = safety_pass
    metric_value = None
    try:
        if metric is not None:
            candidate_metric = float(metric)
            if math.isfinite(candidate_metric):
                metric_value = candidate_metric
    except (TypeError, ValueError):
        pass
    interval_value = normalized_interval(interval)
    if metric_value is None:
        safety_pass = False
        failures.append(f"{metric_name}:missing_or_invalid")
    if interval_value is None:
        failures.append(f"{metric_name}_bootstrap_95ci:missing_or_invalid")
    evidence_valid = metric_value is not None and interval_value is not None
    safety_pass = clinical_safety_pass and evidence_valid
    return {
        "metric_name": metric_name,
        "metric": metric_value,
        "confidence_interval_95": list(interval_value) if interval_value else None,
        "greater_is_better": greater_is_better,
        "safety_pass": safety_pass,
        "clinical_safety_pass": clinical_safety_pass,
        "evidence_valid": evidence_valid,
        "safety_failures": failures,
        "report": report,
    }


def rank_scores(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [entry for entry in entries if entry["score"]["safety_pass"]]
    if not eligible:
        return []
    greater = eligible[0]["score"]["greater_is_better"]
    return sorted(
        eligible,
        key=lambda entry: entry["score"]["metric"],
        reverse=greater,
    )


def run_task_bakeoff(
    *,
    task: str,
    config: dict[str, Any],
    research_approvals: set[str],
    frozen: dict[str, Any],
    python: str,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    required = config["promotion_gate"]["critical_slices"] + config["promotion_gate"]["policy_slices"]
    candidate_a = next(
        item for item in config["candidates"][task] if item["role"] == "candidate_a"
    )
    candidate_a = deepcopy(candidate_a)
    candidate_a["unit_id"] = candidate_a["id"]
    candidate_a_adapter = Path(frozen["adapters"][task]["root"])
    candidate_a_directions = ["en_to_vi", "vi_to_en"] if task == "mt" else [None]
    candidate_a_reports = {}
    for direction in candidate_a_directions:
        key = direction or "vi"
        candidate_a_reports[key] = benchmark_unit(
            unit=candidate_a,
            task=task,
            label=f"candidate_a_frozen_{key}",
            adapter=candidate_a_adapter,
            config=config,
            python=python,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
            direction=direction,
        )

    units = []
    excluded = []
    for unit in candidate_units(config, task):
        allowed, reason = license_gate(unit, research_approvals)
        if allowed:
            units.append(unit)
        else:
            excluded.append({"unit": unit["unit_id"], "reason": reason})

    zero_shot = []
    pilot = []
    for unit in units:
        zero_report = benchmark_unit(
            unit=unit,
            task=task,
            label="zero_shot",
            adapter=None,
            config=config,
            python=python,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
            direction=unit.get("direction"),
        )
        zero_shot.append(
            {"unit": unit["unit_id"], "score": report_score(zero_report, task, unit.get("direction"), required)}
        )
        adapter = train_unit(
            unit=unit,
            task=task,
            round_name="pilot",
            steps=int(config["successive_halving"]["pilot"]["train_steps"]),
            config=config,
            python=python,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
        )
        report = benchmark_unit(
            unit=unit,
            task=task,
            label="pilot",
            adapter=adapter,
            config=config,
            python=python,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
            direction=unit.get("direction"),
        )
        pilot.append(
            {
                "unit": unit["unit_id"],
                "candidate": unit,
                "adapter": str(adapter),
                "score": report_score(report, task, unit.get("direction"), required),
            }
        )

    semifinalists = rank_scores(pilot)[: int(config["successive_halving"]["semifinal"]["keep"])]
    semifinal = []
    for entry in semifinalists:
        unit = entry["candidate"]
        adapter = train_unit(
            unit=unit,
            task=task,
            round_name="semifinal",
            steps=int(config["successive_halving"]["semifinal"]["train_steps"]),
            config=config,
            python=python,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
        )
        report = benchmark_unit(
            unit=unit,
            task=task,
            label="semifinal",
            adapter=adapter,
            config=config,
            python=python,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
            direction=unit.get("direction"),
        )
        semifinal.append(
            {
                "unit": unit["unit_id"],
                "candidate": unit,
                "adapter": str(adapter),
                "score": report_score(report, task, unit.get("direction"), required),
            }
        )

    ranked_semifinal = rank_scores(semifinal)
    finalists = []
    if ranked_semifinal:
        best_interval = ranked_semifinal[0]["score"]["confidence_interval_95"]
        finalists = [
            entry
            for entry in ranked_semifinal
            if entry is ranked_semifinal[0]
            or (
                best_interval
                and entry["score"]["confidence_interval_95"]
                and intervals_overlap(best_interval, entry["score"]["confidence_interval_95"])
            )
        ]

    full = []
    for entry in finalists:
        unit = entry["candidate"]
        adapter = train_unit(
            unit=unit,
            task=task,
            round_name="full",
            steps=int(config["successive_halving"]["full"]["train_steps"]),
            config=config,
            python=python,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
        )
        report = benchmark_unit(
            unit=unit,
            task=task,
            label="full",
            adapter=adapter,
            config=config,
            python=python,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
            direction=unit.get("direction"),
        )
        full.append(
            {
                "unit": unit["unit_id"],
                "candidate": unit,
                "adapter": str(adapter),
                "score": report_score(report, task, unit.get("direction"), required),
            }
        )

    directions = ["en_to_vi", "vi_to_en"] if task == "mt" else [None]
    winners = {}
    for direction in directions:
        baseline_score = report_score(
            candidate_a_reports[direction or "vi"], task, direction, required
        )
        challengers = [
            entry
            for entry in full
            if task != "mt" or entry["candidate"].get("direction") == direction
        ]
        ranked = rank_scores(challengers)
        if not baseline_score["evidence_valid"]:
            raise ValueError(
                f"Candidate A evidence is invalid for {task}/{direction or 'vi'}"
            )
        if not baseline_score["clinical_safety_pass"] and not ranked:
            raise RuntimeError(
                f"No safety-eligible winner for {task}/{direction or 'vi'}"
            )
        winner = {
            "candidate_id": candidate_a["id"],
            "adapter": str(candidate_a_adapter),
            "direction": direction,
            "score": baseline_score,
            "decision": "candidate_a_retained",
        }
        if ranked:
            challenger = ranked[0]
            challenger_score = challenger["score"]
            stronger = (
                not baseline_score["clinical_safety_pass"]
                or interval_stronger(
                    challenger_score["confidence_interval_95"],
                    baseline_score["confidence_interval_95"],
                    greater_is_better=challenger_score["greater_is_better"],
                )
            )
            if stronger:
                winner = {
                    "candidate_id": challenger["candidate"]["id"],
                    "adapter": challenger["adapter"],
                    "direction": direction,
                    "score": challenger_score,
                    "decision": (
                        "challenger_selected_after_candidate_a_safety_failure"
                        if not baseline_score["clinical_safety_pass"]
                        else "challenger_stronger_beyond_95ci"
                    ),
                }
        winners[direction or "vi"] = winner
    return {
        "task": task,
        "excluded_by_license": excluded,
        "zero_shot": zero_shot,
        "pilot": pilot,
        "semifinal": semifinal,
        "full": full,
        "winners": winners,
    }


def preflight(config: dict[str, Any], research_approvals: set[str]) -> dict[str, Any]:
    validate_resources(config)
    validate_candidate_matrix(config)
    hashes = validate_selection_artifacts(config)
    licenses = {}
    for task in ("mt", "asr"):
        for candidate in config["candidates"][task]:
            allowed, reason = license_gate(candidate, research_approvals)
            licenses[candidate["id"]] = {"gpu_allowed": allowed, "reason": reason}
    return {
        "status": "pass",
        "selection_sha256": hashes,
        "licenses": licenses,
        "workflow": config["workflow"],
        "resource_policy": config["resources"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/model_bakeoff.yaml")
    parser.add_argument("--current-program-state", type=Path)
    parser.add_argument("--state-dir", type=Path, default=ROOT / "gpu-runs/model-bakeoff")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--wait-current", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--scope", choices=["research", "production"], default="research")
    parser.add_argument(
        "--approve-research-license",
        action="append",
        default=[],
        help="Explicit candidate ID approval for a research-only license; never enables production.",
    )
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    config = load_config(args.config)
    report = preflight(config, set(args.approve_research_license))
    if args.preflight or not args.execute:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    program_path = find_current_program_state(config, args.current_program_state)
    current = wait_for_current_program(program_path, args.poll_seconds, args.wait_current)
    state_dir = args.state_dir.resolve()
    state_path = state_dir / "bakeoff_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "created_at": utc_now(),
            "execution_status": "running",
            "promotion_allowed": False,
            "current_program_state": str(program_path),
            "stages": {},
        }
        atomic_json(state_path, state)
    freeze_path = state_dir / "candidate_a_freeze.json"
    if not freeze_path.is_file():
        freeze_candidate_a(current, freeze_path)
    state["stage"] = "freeze_candidate_A"
    state["candidate_a_freeze"] = str(freeze_path)
    state["scope"] = args.scope
    atomic_json(state_path, state)
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))

    state["stage"] = "mt_multi_model_bakeoff"
    atomic_json(state_path, state)
    mt_result = run_task_bakeoff(
        task="mt",
        config=config,
        research_approvals=set(args.approve_research_license),
        frozen=frozen,
        python=args.python,
        state_dir=state_dir,
        state_path=state_path,
        state=state,
    )
    state["stage"] = "asr_multi_model_bakeoff"
    atomic_json(state_path, state)
    asr_result = run_task_bakeoff(
        task="asr",
        config=config,
        research_approvals=set(args.approve_research_license),
        frozen=frozen,
        python=args.python,
        state_dir=state_dir,
        state_path=state_path,
        state=state,
    )
    comparison = {
        "version": 1,
        "updated_at": utc_now(),
        "status": "selection_complete",
        "scope": args.scope,
        "promotion_allowed": False,
        "candidate_a_freeze": str(freeze_path),
        "selection_sha256": report["selection_sha256"],
        "results": {"mt": mt_result, "asr": asr_result},
        "blind_test_v2": "pending",
    }
    comparison_path = ROOT / "data/reports/model_bakeoff/comparison.json"
    atomic_json(comparison_path, comparison)

    blind_lock_path = ROOT / config["data"]["blind_test_v2_lock"]
    blind_lock = json.loads(blind_lock_path.read_text(encoding="utf-8"))
    if blind_lock.get("status") == "awaiting_unseen_data":
        state["stage"] = "blind_locked_test_v2"
        state["execution_status"] = "waiting_for_blind_test_v2"
        state["next_stage"] = "blind_locked_test_v2"
        state["selection_comparison"] = str(comparison_path)
        atomic_json(state_path, state)
        print(
            json.dumps(
                {
                    **report,
                    "state": str(state_path),
                    "candidate_a_freeze": str(freeze_path),
                    "selection_comparison": str(comparison_path),
                    "next_stage": state["next_stage"],
                    "note": "Selection is complete; promotion is blocked until unseen blind v2 data is checksum-locked.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return CRITICAL_EXIT_WAITING_FOR_BLIND

    if blind_lock.get("status") not in {"locked_unopened", "partially_opened", "opened"}:
        raise ValueError(f"Unexpected blind v2 state: {blind_lock.get('status')!r}")
    state["stage"] = "blind_locked_test_v2"
    blind_results = []
    blind_output = state_dir / "blind-v2"
    winner_specs = [
        ("mt", direction, winner)
        for direction, winner in mt_result["winners"].items()
    ] + [("asr", None, asr_result["winners"]["vi"])]
    for task, direction, winner in winner_specs:
        key = direction or "vi"
        stem = f"blind_v2_{task}_{key}_{winner['candidate_id']}"
        gate_path = blind_output / f"{stem}_gate.json"
        command = [
            args.python,
            str(ROOT / "scripts/run_blind_candidate_suite.py"),
            "--action",
            "evaluate",
            "--task",
            task,
            "--candidate",
            winner["candidate_id"],
            "--adapter",
            winner["adapter"],
            "--scope",
            args.scope,
            "--output-dir",
            str(blind_output),
        ]
        if direction:
            command += ["--direction", direction]
        stage = f"blind_v2_{task}_{key}"
        run_stage(
            stage,
            command,
            state_path,
            state,
            state_dir / "logs" / f"{stage}.log",
            [gate_path],
        )
        blind_results.append(json.loads(gate_path.read_text(encoding="utf-8")))
    comparison["blind_test_v2"] = blind_results
    comparison["status"] = "blind_complete"
    atomic_json(comparison_path, comparison)

    deployment_path = ROOT / "data/reports/model_bakeoff/deployment_selected_winners.json"
    if not deployment_path.is_file():
        state["stage"] = "deployment_benchmark"
        state["execution_status"] = "waiting_for_qcs6490_benchmark"
        state["next_stage"] = "deployment_benchmark"
        atomic_json(state_path, state)
        return CRITICAL_EXIT_WAITING_FOR_BLIND
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    deployment_pass = (
        deployment.get("status") == "pass"
        and deployment.get("target") == "QCS6490"
        and all(
            deployment.get(name) is not None
            for name in config["promotion_gate"]["deployment_metrics"]
        )
    )
    blind_pass = all(item.get("promotion_allowed") for item in blind_results)
    winner_ids = {
        winner["candidate_id"]
        for _, _, winner in winner_specs
    }
    production_licenses = all(
        next(
            candidate
            for task in ("mt", "asr")
            for candidate in config["candidates"][task]
            if candidate["id"] == winner_id
        )["license"].get("production_eligible")
        for winner_id in winner_ids
    )
    promotion_allowed = blind_pass and deployment_pass and (
        args.scope == "research" or production_licenses
    )
    comparison.update(
        {
            "status": "complete",
            "deployment": deployment,
            "promotion_allowed": promotion_allowed,
            "production_license_gate": production_licenses,
        }
    )
    atomic_json(comparison_path, comparison)
    state["stage"] = "promotion_or_reject"
    state["execution_status"] = "complete"
    state["promotion_allowed"] = promotion_allowed
    state["decision"] = "promote" if promotion_allowed else "reject"
    atomic_json(state_path, state)
    print(
        json.dumps(
            {
                **report,
                "state": str(state_path),
                "candidate_a_freeze": str(freeze_path),
                "comparison": str(comparison_path),
                "decision": state["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if promotion_allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())

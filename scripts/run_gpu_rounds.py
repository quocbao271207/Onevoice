"""Run bounded GPU tuning rounds with hard memory and rolling-utilization controls."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.candidate_evidence import archive_evidence, evidence_sidecars  # noqa: E402


def validate_resource_limits(limits: dict[str, Any]) -> tuple[float, float]:
    """Validate shared GPU limits and return soft/hard memory fractions."""
    required = {
        "gpu_memory_fraction",
        "gpu_memory_hard_fraction",
        "utilization_percent",
        "hard_utilization_percent",
        "resume_percent",
        "sample_seconds",
        "rolling_samples",
    }
    missing = sorted(required - set(limits))
    if missing:
        raise ValueError(f"GPU resource limits are missing: {missing}")

    memory_fraction = float(limits["gpu_memory_fraction"])
    hard_memory_fraction = float(limits["gpu_memory_hard_fraction"])
    utilization_limit = float(limits["utilization_percent"])
    hard_utilization_limit = float(limits["hard_utilization_percent"])
    resume_percent = float(limits["resume_percent"])
    sample_seconds = float(limits["sample_seconds"])
    numeric_values = (
        memory_fraction,
        hard_memory_fraction,
        utilization_limit,
        hard_utilization_limit,
        resume_percent,
        sample_seconds,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("GPU resource limits must be finite")
    if not 0.0 < memory_fraction <= 0.35:
        raise ValueError("gpu_memory_fraction must be in (0, 0.35]")
    if not memory_fraction <= hard_memory_fraction <= 0.40:
        raise ValueError(
            "GPU memory thresholds must satisfy process fraction <= hard fraction <= 0.40"
        )
    if not 0.0 < resume_percent < utilization_limit < hard_utilization_limit < 75.0:
        raise ValueError("utilization thresholds must satisfy 0 < resume < rolling < hard < 75")
    if sample_seconds <= 0.0:
        raise ValueError("sample_seconds must be positive")

    rolling_samples = limits["rolling_samples"]
    if (
        isinstance(rolling_samples, bool)
        or not isinstance(rolling_samples, int)
        or rolling_samples < 1
    ):
        raise ValueError("rolling_samples must be a positive integer")

    boosted = limits.get("boosted_window")
    if boosted:
        boosted_utilization = float(boosted["utilization_percent"])
        boosted_resume = float(boosted["resume_percent"])
        if not all(math.isfinite(value) for value in (boosted_utilization, boosted_resume)):
            raise ValueError("boosted utilization limits must be finite")
        if not utilization_limit < boosted_utilization < hard_utilization_limit:
            raise ValueError(
                "boosted utilization must be above the base limit and below the hard limit"
            )
        if not 0.0 < boosted_resume < boosted_utilization:
            raise ValueError("boosted resume must be positive and below boosted utilization")
        start_hour = boosted["start_hour"]
        end_hour = boosted["end_hour"]
        if any(
            isinstance(hour, bool) or not isinstance(hour, int) or not 0 <= hour <= 23
            for hour in (start_hour, end_hour)
        ):
            raise ValueError("boosted window hours must be integers in [0, 23]")
        if start_hour == end_hour:
            raise ValueError("boosted window must not cover the entire day")
        ZoneInfo(str(boosted["timezone"]))

    return memory_fraction, hard_memory_fraction


def read_gpu_sample() -> dict[str, float]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.total,memory.used,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()[0]
    values = [float(value.strip()) for value in output.split(",")]
    return dict(zip(("utilization", "memory_total_mib", "memory_used_mib", "power_w", "power_limit_w"), values))


def gpu_spawn_capacity(
    sample: dict[str, float],
    limits: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate whether a new GPU child can safely reserve its hard memory budget."""
    _, hard_memory_fraction = validate_resource_limits(limits)
    required = ("utilization", "memory_total_mib", "memory_used_mib")
    try:
        utilization, total, used = (float(sample[name]) for name in required)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("GPU capacity sample is missing finite numeric fields") from exc
    if not all(math.isfinite(value) for value in (utilization, total, used)):
        raise ValueError("GPU capacity sample must be finite")
    if total <= 0.0 or not 0.0 <= used <= total or not 0.0 <= utilization <= 100.0:
        raise ValueError("GPU capacity sample is outside valid bounds")

    _, resume_percent, boosted_window = active_utilization_limits(limits, now)
    free = total - used
    required_free = total * hard_memory_fraction
    reason = None
    if utilization > resume_percent:
        reason = "utilization_above_resume"
    elif free < required_free:
        reason = "insufficient_memory_headroom"
    return {
        "allowed": reason is None,
        "reason": reason,
        "utilization": utilization,
        "memory_total_mib": total,
        "memory_used_mib": used,
        "memory_free_mib": free,
        "required_free_memory_mib": required_free,
        "active_resume_percent": resume_percent,
        "configured_hard_memory_fraction": hard_memory_fraction,
        "boosted_window": boosted_window,
    }


def wait_for_gpu_spawn_capacity(
    limits: dict[str, Any],
    evidence_path: Path,
    *,
    sample_reader: Callable[[], dict[str, float]] | None = None,
    sleeper: Callable[[float], None] | None = None,
    poll_seconds: float | None = None,
) -> dict[str, Any]:
    """Wait without a GPU child until shared-GPU utilization and memory are safe."""
    validate_resource_limits(limits)
    reader = sample_reader or read_gpu_sample
    sleep = sleeper or time.sleep
    interval = (
        max(60.0, float(limits["sample_seconds"]))
        if poll_seconds is None
        else float(poll_seconds)
    )
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError("GPU spawn-capacity poll interval must be positive and finite")

    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    samples = 0
    with evidence_path.open("a", encoding="utf-8") as evidence:
        while True:
            samples += 1
            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                capacity = gpu_spawn_capacity(reader(), limits)
                record = {"time": timestamp, **capacity}
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                record = {
                    "time": timestamp,
                    "allowed": False,
                    "reason": "gpu_sample_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            evidence.write(json.dumps(record) + "\n")
            evidence.flush()
            if record["allowed"]:
                return {
                    "samples": samples,
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "evidence_log": str(evidence_path),
                    "approval": record,
                }
            sleep(interval)


def active_utilization_limits(
    limits: dict[str, Any], now: datetime | None = None
) -> tuple[float, float, bool]:
    """Return scheduled utilization/resume thresholds for the current local hour."""
    window = limits.get("boosted_window")
    if not window:
        return float(limits["utilization_percent"]), float(limits["resume_percent"]), False
    local_now = now or datetime.now(ZoneInfo(str(window["timezone"])))
    if local_now.tzinfo is None:
        local_now = local_now.replace(tzinfo=ZoneInfo(str(window["timezone"])))
    else:
        local_now = local_now.astimezone(ZoneInfo(str(window["timezone"])))
    start = int(window["start_hour"])
    end = int(window["end_hour"])
    active = start <= local_now.hour < end if start < end else local_now.hour >= start or local_now.hour < end
    if active:
        return float(window["utilization_percent"]), float(window["resume_percent"]), True
    return float(limits["utilization_percent"]), float(limits["resume_percent"]), False


def utilization_throttle_reason(
    *,
    current: float,
    rolling: float,
    sample_count: int,
    rolling_samples: int,
    utilization_limit: float,
    hard_utilization_limit: float,
) -> str | None:
    """Return the first utilization guard that requires pausing the GPU child."""
    if current >= hard_utilization_limit:
        return "hard_utilization"
    if sample_count >= rolling_samples and rolling > utilization_limit:
        return "rolling_utilization"
    return None


def process_group_pids(group_id: int) -> set[int]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,pgid="],
        check=True,
        text=True,
        capture_output=True,
    )
    return {
        int(pid)
        for line in result.stdout.splitlines()
        for pid, pgid in [line.split()]
        if int(pgid) == group_id
    }


def process_pid_aliases(
    process_pids: set[int], proc_root: Path = Path("/proc")
) -> set[int]:
    """Return container PIDs plus their host/ancestor namespace aliases.

    ``nvidia-smi`` commonly reports the host PID while ``ps`` inside a
    container reports the innermost namespace PID. Linux exposes the complete
    mapping in ``/proc/<pid>/status`` as ``NSpid``. Reading every alias lets the
    hard-memory guard identify the actual training process instead of
    attributing unrelated GPU jobs that happened to start after the baseline.
    """
    aliases = set(process_pids)
    if os.name != "posix":
        return aliases
    for pid in process_pids:
        try:
            status = (proc_root / str(pid) / "status").read_text(
                encoding="utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        nspid = next(
            (line for line in status.splitlines() if line.startswith("NSpid:")),
            None,
        )
        if nspid is None:
            continue
        aliases.update(
            int(value)
            for value in nspid.split()[1:]
            if value.isdigit()
        )
    return aliases


def gpu_process_memory_by_pid() -> dict[int, float]:
    """Return compute-process memory keyed by the PID reported by NVIDIA."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    memory_by_pid: dict[int, float] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        try:
            memory_by_pid[int(fields[0])] = float(fields[1])
        except ValueError:
            continue
    return memory_by_pid


def attributed_process_gpu_memory(
    group_id: int,
    gpu_memory_baseline: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Attribute GPU memory despite host/container PID namespace differences.

    NVIDIA may expose host PIDs while ``ps`` exposes container PIDs. Prefer an
    exact process-group match. When that is impossible, first attribute GPU
    PIDs that appeared after the pre-spawn baseline. Some container runtimes
    keep a stable host-side GPU PID, so fall back to positive memory growth on
    baseline PIDs instead of silently reporting zero.
    """
    process_pids = process_group_pids(group_id)
    process_aliases = process_pid_aliases(process_pids)
    current = gpu_process_memory_by_pid()
    direct_gpu_pids = process_aliases & set(current)
    if direct_gpu_pids:
        gpu_pids = direct_gpu_pids
        attribution = (
            "process_group_pid"
            if direct_gpu_pids & process_pids
            else "process_namespace_pid"
        )
    elif gpu_memory_baseline is not None:
        gpu_pids = set(current) - set(gpu_memory_baseline)
        if gpu_pids:
            memory_mib = sum(current[pid] for pid in gpu_pids)
            attribution = "post_spawn_pid"
        else:
            growth_by_pid = {
                pid: current[pid] - gpu_memory_baseline[pid]
                for pid in set(current) & set(gpu_memory_baseline)
                if current[pid] > gpu_memory_baseline[pid]
            }
            gpu_pids = set(growth_by_pid)
            memory_mib = sum(growth_by_pid.values())
            attribution = "post_spawn_memory_growth"
    else:
        gpu_pids = set()
        memory_mib = 0.0
        attribution = "unattributed"
    if direct_gpu_pids:
        memory_mib = sum(current[pid] for pid in gpu_pids)
    return {
        "memory_mib": memory_mib,
        "attribution": attribution,
        "gpu_pids": sorted(gpu_pids),
        "process_pids": sorted(process_pids),
    }


def exact_process_memory_attribution(sample: dict[str, Any]) -> bool:
    """Return whether NVIDIA memory belongs to the tracked process exactly.

    Baseline-growth fallbacks remain useful telemetry, but on a shared GPU they
    can include another container's new PID or memory growth. The trainer's
    PyTorch allocator independently enforces the configured 35% process cap;
    the monitor's 40% kill switch is therefore used only for exact PID matches.
    """
    return sample.get("attribution") in {
        "process_group_pid",
        "process_namespace_pid",
    }


def cli_args(values: dict[str, Any]) -> list[str]:
    output: list[str] = []
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                output.append(flag)
        elif value is not None:
            output.extend((flag, str(value)))
    return output


def checkpoint_watcher_command(
    python: str,
    model_dir: Path,
    archive_dir: Path,
    training_pid: int,
    poll_seconds: float,
) -> list[str]:
    return [
        python,
        str(ROOT / "scripts" / "watch_checkpoints.py"),
        "--model-dir",
        str(model_dir),
        "--archive-dir",
        str(archive_dir),
        "--watch-pid",
        str(training_pid),
        "--poll-seconds",
        str(poll_seconds),
    ]


def archive_round(round_dir: Path) -> dict[str, Any]:
    archive_path, digest = archive_evidence(round_dir)
    checksum_path, manifest_path = evidence_sidecars(archive_path)
    return {
        "path": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": digest,
        "checksum": str(checksum_path),
        "manifest": str(manifest_path),
    }


def metric_from_report(report: dict[str, Any], name: str) -> float | None:
    if report.get("best_metric") is not None:
        return float(report["best_metric"])
    for item in reversed(report.get("log_history", [])):
        if name in item:
            return float(item[name])
    return None


def configured_rounds(
    task: dict[str, Any], max_rounds: int = 0, round_name: str | None = None
) -> list[dict[str, Any]]:
    rounds = list(task["rounds"])
    if round_name:
        selected = [item for item in rounds if item["name"] == round_name]
        if not selected:
            available = ", ".join(str(item["name"]) for item in rounds)
            raise ValueError(f"Unknown round {round_name!r}; available rounds: {available}")
        return selected
    return rounds[: max_rounds or None]


def apply_round_overrides(
    rounds: list[dict[str, Any]],
    *,
    round_name: str | None,
    resume_from_checkpoint: Path | None,
    initial_adapter: Path | None,
    epochs: float | None,
    learning_rate: float | None,
) -> list[dict[str, Any]]:
    if (
        resume_from_checkpoint is None
        and initial_adapter is None
        and epochs is None
        and learning_rate is None
    ):
        return rounds
    if not round_name or len(rounds) != 1:
        raise ValueError("continuation overrides require --round-name")
    if resume_from_checkpoint and not (resume_from_checkpoint / "trainer_state.json").is_file():
        raise ValueError("--resume-from-checkpoint must contain trainer_state.json")
    if initial_adapter and not (initial_adapter / "adapter_config.json").is_file():
        raise ValueError("--initial-adapter must contain adapter_config.json")
    if epochs is not None and epochs <= 0:
        raise ValueError("--epochs must be positive")
    if learning_rate is not None and learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    return [
        {
            **rounds[0],
            **(
                {"resume_from_checkpoint": str(resume_from_checkpoint.resolve())}
                if resume_from_checkpoint
                else {}
            ),
            **(
                {"initial_adapter": str(initial_adapter.resolve())}
                if initial_adapter
                else {}
            ),
            **({"epochs": epochs} if epochs is not None else {}),
            **({"learning_rate": learning_rate} if learning_rate is not None else {}),
        }
    ]


def select_completed_round(
    results: list[dict[str, Any]],
    greater_is_better: bool,
    allowed_names: set[str] | None = None,
) -> dict[str, Any] | None:
    candidates = [
        item
        for item in results
        if item.get("status") == "complete"
        and item.get("metric") is not None
        and (allowed_names is None or item.get("name") in allowed_names)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: float(item["metric"]), reverse=greater_is_better)[0]


def resolve_adaptive_final(
    task: dict[str, Any], results: list[dict[str, Any]]
) -> tuple[dict[str, Any], str]:
    """Build a full-data round from the best completed pilot."""
    spec = task.get("adaptive_final")
    if not isinstance(spec, dict):
        raise ValueError("adaptive_final must be a mapping")
    source_names = [str(name) for name in spec.get("source_rounds", [])]
    if not source_names:
        raise ValueError("adaptive_final.source_rounds must not be empty")
    round_by_name = {str(item["name"]): item for item in task.get("rounds", [])}
    missing_sources = sorted(set(source_names) - set(round_by_name))
    if missing_sources:
        raise ValueError(f"adaptive final references unknown source rounds: {missing_sources}")
    selected = select_completed_round(
        results,
        bool(task["greater_is_better"]),
        set(source_names),
    )
    if selected is None:
        raise ValueError("adaptive final has no completed source round with a selection metric")

    final_name = str(spec.get("name") or "")
    if not final_name:
        raise ValueError("adaptive_final.name must not be empty")
    if final_name in round_by_name:
        raise ValueError(f"adaptive final name conflicts with a configured round: {final_name}")
    inherit = [str(name) for name in spec.get("inherit", [])]
    if not inherit:
        raise ValueError("adaptive_final.inherit must not be empty")
    source = round_by_name[str(selected["name"])]
    missing_values = sorted(name for name in inherit if name not in source)
    if missing_values:
        raise ValueError(f"selected source round is missing inherited values: {missing_values}")
    overrides = spec.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("adaptive_final.overrides must be a mapping")
    forbidden = {"name", "source_rounds", "inherit", "overrides"} & set(overrides)
    if forbidden:
        raise ValueError(f"adaptive final overrides contain reserved keys: {sorted(forbidden)}")
    resolved = {
        "name": final_name,
        **{name: source[name] for name in inherit},
        **overrides,
    }
    return resolved, str(selected["name"])


def monitor_process(
    process: subprocess.Popen[str],
    monitor_path: Path,
    limits: dict[str, Any],
    gpu_memory_baseline: dict[int, float] | None = None,
) -> dict[str, Any]:
    memory_fraction, hard_memory_fraction = validate_resource_limits(limits)
    hard_utilization_limit = float(limits["hard_utilization_percent"])
    sample_seconds = float(limits["sample_seconds"])
    window = deque(maxlen=int(limits["rolling_samples"]))
    stopped = False
    peak_process_memory = 0.0
    samples = 0
    throttles = 0
    memory_attribution_modes: set[str] = set()
    started = time.monotonic()
    process_group = os.getpgid(process.pid)
    try:
        with monitor_path.open("w", encoding="utf-8") as monitor:
            while process.poll() is None:
                sample = read_gpu_sample()
                utilization_limit, resume_percent, boosted_window = active_utilization_limits(limits)
                process_memory_sample = attributed_process_gpu_memory(
                    process_group,
                    gpu_memory_baseline,
                )
                process_memory = float(process_memory_sample["memory_mib"])
                hard_memory_enforced = exact_process_memory_attribution(
                    process_memory_sample
                )
                memory_attribution_modes.add(str(process_memory_sample["attribution"]))
                peak_process_memory = max(peak_process_memory, process_memory)
                hard_memory_limit = sample["memory_total_mib"] * hard_memory_fraction
                if hard_memory_enforced and process_memory > hard_memory_limit:
                    os.killpg(process_group, signal.SIGCONT)
                    stopped = False
                    os.killpg(process_group, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process_group, signal.SIGKILL)
                        process.wait(timeout=10)
                    raise RuntimeError(
                        "Training exceeded the hard "
                        f"{hard_memory_fraction:.0%} GPU-memory limit: {process_memory:.0f} MiB"
                    )
                window.append(sample["utilization"])
                rolling = sum(window) / len(window)
                throttle_reason = utilization_throttle_reason(
                    current=sample["utilization"],
                    rolling=rolling,
                    sample_count=len(window),
                    rolling_samples=window.maxlen,
                    utilization_limit=utilization_limit,
                    hard_utilization_limit=hard_utilization_limit,
                )
                if not stopped and throttle_reason is not None:
                    os.killpg(process_group, signal.SIGSTOP)
                    stopped = True
                    throttles += 1
                elif stopped and rolling <= resume_percent:
                    os.killpg(process_group, signal.SIGCONT)
                    stopped = False
                record = {
                    "time": datetime.now(timezone.utc).isoformat(),
                    **sample,
                    "process_memory_mib": process_memory,
                    "process_memory_fraction": process_memory / max(1.0, sample["memory_total_mib"]),
                    "process_memory_attribution": process_memory_sample["attribution"],
                    "hard_memory_enforced_from_nvidia_pid": hard_memory_enforced,
                    "hard_memory_fallback": (
                        None if hard_memory_enforced else "pytorch_allocator_fraction"
                    ),
                    "attributed_gpu_pids": process_memory_sample["gpu_pids"],
                    "tracked_process_pids": process_memory_sample["process_pids"],
                    "configured_memory_fraction": memory_fraction,
                    "configured_hard_memory_fraction": hard_memory_fraction,
                    "hard_memory_limit_mib": hard_memory_limit,
                    "rolling_utilization": rolling,
                    "training_stopped": stopped,
                    "active_utilization_limit_percent": utilization_limit,
                    "hard_utilization_limit_percent": hard_utilization_limit,
                    "throttle_reason": throttle_reason,
                    "boosted_window": boosted_window,
                }
                monitor.write(json.dumps(record) + "\n")
                monitor.flush()
                samples += 1
                time.sleep(sample_seconds)
    finally:
        if stopped and process.poll() is None:
            os.killpg(process_group, signal.SIGCONT)
    return {
        "return_code": process.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "samples": samples,
        "throttle_events": throttles,
        "peak_process_memory_mib": peak_process_memory,
        "memory_attribution_modes": sorted(memory_attribution_modes),
        "configured_memory_fraction": memory_fraction,
        "configured_hard_memory_fraction": hard_memory_fraction,
        "configured_utilization_limit_percent": float(limits["utilization_percent"]),
        "hard_utilization_limit_percent": hard_utilization_limit,
        "daytime_utilization_limit_percent": float(limits["utilization_percent"]),
        "boosted_utilization_limit_percent": float(
            limits.get("boosted_window", {}).get("utilization_percent", limits["utilization_percent"])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("asr", "mt"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "gpu_rounds.yaml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "gpu-runs")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-rounds", type=int, default=0, help="0 runs every configured round.")
    parser.add_argument("--round-name", help="Run exactly one named round so its archive can be downloaded immediately.")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume one explicitly selected round from a completed Trainer checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        help="Override epochs for one explicitly selected round, for validation-driven continuation.",
    )
    parser.add_argument(
        "--initial-adapter",
        type=Path,
        help="Warm-start one selected LoRA round with a completed adapter and fresh optimizer.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate for one explicitly selected round.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    limits = config["limits"]
    validate_resource_limits(limits)
    task = config["tasks"][args.task]
    rounds = configured_rounds(task, args.max_rounds, args.round_name)
    rounds = apply_round_overrides(
        rounds,
        round_name=args.round_name,
        resume_from_checkpoint=args.resume_from_checkpoint,
        initial_adapter=args.initial_adapter,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    adaptive_spec = (
        task.get("adaptive_final") if not args.round_name and args.max_rounds == 0 else None
    )
    initial_round_count = len(rounds)
    adaptive_appended = False
    run_root = args.output_root.resolve() / f"{args.task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "task": args.task,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "limits": limits,
        "rounds": [],
    }

    for index, round_config in enumerate(rounds, 1):
        name = str(round_config["name"])
        round_dir = run_root / f"{index:02d}-{name}"
        model_dir = round_dir / "model"
        round_dir.mkdir(parents=True)
        model_dir.mkdir()
        values = {
            **task.get("common_args", {}),
            **{key: value for key, value in round_config.items() if key != "name"},
            "gpu_memory_fraction": limits["gpu_memory_fraction"],
            "output_dir": model_dir,
        }
        command = [args.python, "-m", task["module"], *cli_args(values)]
        (round_dir / "command.json").write_text(
            json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if args.dry_run:
            summary["rounds"].append({"name": name, "command": command, "status": "dry_run"})
            continue
        log_path = round_dir / "training.log"
        monitor_path = round_dir / "resource_monitor.jsonl"
        spawn_capacity_path = round_dir / "pre_spawn_capacity.jsonl"
        checkpoint_archive_dir = run_root / "checkpoint-archives" / round_dir.name
        watcher_log_path = round_dir / "checkpoint_watcher.log"
        spawn_capacity = wait_for_gpu_spawn_capacity(limits, spawn_capacity_path)
        with log_path.open("w", encoding="utf-8") as log:
            gpu_memory_baseline = gpu_process_memory_by_pid()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            watcher_command = checkpoint_watcher_command(
                args.python,
                model_dir,
                checkpoint_archive_dir,
                process.pid,
                float(limits.get("checkpoint_archive_poll_seconds", 15.0)),
            )
            with watcher_log_path.open("w", encoding="utf-8") as watcher_log:
                watcher = subprocess.Popen(
                    watcher_command,
                    cwd=ROOT,
                    stdout=watcher_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                try:
                    monitor_summary = monitor_process(
                        process,
                        monitor_path,
                        limits,
                        gpu_memory_baseline,
                    )
                finally:
                    try:
                        watcher.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        watcher.terminate()
                        try:
                            watcher.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            watcher.kill()
                            watcher.wait(timeout=10)
            watcher_summary = {
                "return_code": watcher.returncode,
                "archive_dir": str(checkpoint_archive_dir),
                "manifest": str(checkpoint_archive_dir / "checkpoint_archives.json"),
                "log": str(watcher_log_path),
            }
        report_path = model_dir / "training_run.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        training_complete = monitor_summary["return_code"] == 0
        watcher_complete = watcher_summary["return_code"] == 0
        result = {
            "name": name,
            "status": "complete" if training_complete and watcher_complete else "failed",
            "spawn_capacity": spawn_capacity,
            "monitor": monitor_summary,
            "checkpoint_watcher": watcher_summary,
            "metric": metric_from_report(report, task["selection_metric"]),
            "metric_name": task["selection_metric"],
        }
        if training_complete:
            result["archive"] = archive_round(round_dir)
        summary["rounds"].append(result)
        if adaptive_spec and name == str(adaptive_spec.get("name")):
            adaptive_status = (
                "complete"
                if result["status"] == "complete" and result["metric"] is not None
                else "failed"
            )
            summary["adaptive_final"].update(
                {
                    "status": adaptive_status,
                    "metric": result["metric"],
                    **(
                        {"error": "adaptive final completed without a selection metric"}
                        if result["status"] == "complete" and result["metric"] is None
                        else {}
                    ),
                }
            )
        (run_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not training_complete or not watcher_complete:
            break
        if adaptive_spec and not adaptive_appended and index == initial_round_count:
            try:
                adaptive_round, source_round = resolve_adaptive_final(task, summary["rounds"])
            except ValueError as exc:
                summary["adaptive_final"] = {
                    "status": "failed",
                    "error": str(exc),
                }
                break
            rounds.append(adaptive_round)
            adaptive_appended = True
            summary["adaptive_final"] = {
                "status": "scheduled",
                "source_round": source_round,
                "round_name": adaptive_round["name"],
                "resolved_config": adaptive_round,
            }
            summary["pilot_selected_round"] = source_round
            (run_root / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    if adaptive_spec and args.dry_run:
        summary["adaptive_final"] = {
            "status": "pending_selection",
            "template": adaptive_spec,
        }
    selected = select_completed_round(
        summary["rounds"],
        bool(task["greater_is_better"]),
    )
    if selected:
        summary["selected_round"] = selected["name"]
    if adaptive_spec and summary.get("adaptive_final", {}).get("status") == "complete":
        summary["selected_round"] = summary["adaptive_final"]["round_name"]
    rounds_complete = len(summary["rounds"]) == len(rounds) and all(
        item["status"] in {"complete", "dry_run"} for item in summary["rounds"]
    )
    adaptive_complete = (
        not adaptive_spec
        or args.dry_run
        or summary.get("adaptive_final", {}).get("status") == "complete"
    )
    summary["status"] = "complete" if rounds_complete and adaptive_complete else "failed"
    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_root": str(run_root), **summary}, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

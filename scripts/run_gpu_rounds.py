"""Run bounded GPU tuning rounds with hard memory and rolling-utilization controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]


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


def process_gpu_memory_mib(group_id: int) -> float:
    pids = process_group_pids(group_id)
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
    total = 0.0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) in pids:
            try:
                total += float(fields[1])
            except ValueError:
                continue
    return total


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def archive_round(round_dir: Path) -> dict[str, Any]:
    archive_path = round_dir.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(round_dir, arcname=round_dir.name)
    return {
        "path": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256(archive_path),
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


def monitor_process(
    process: subprocess.Popen[str],
    monitor_path: Path,
    limits: dict[str, Any],
) -> dict[str, Any]:
    memory_fraction = float(limits["gpu_memory_fraction"])
    sample_seconds = float(limits["sample_seconds"])
    window = deque(maxlen=int(limits["rolling_samples"]))
    stopped = False
    peak_process_memory = 0.0
    samples = 0
    throttles = 0
    started = time.monotonic()
    process_group = os.getpgid(process.pid)
    try:
        with monitor_path.open("w", encoding="utf-8") as monitor:
            while process.poll() is None:
                sample = read_gpu_sample()
                utilization_limit, resume_percent, boosted_window = active_utilization_limits(limits)
                process_memory = process_gpu_memory_mib(process_group)
                peak_process_memory = max(peak_process_memory, process_memory)
                hard_memory_limit = sample["memory_total_mib"] * 0.40
                if process_memory > hard_memory_limit:
                    os.killpg(process_group, signal.SIGCONT)
                    stopped = False
                    os.killpg(process_group, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process_group, signal.SIGKILL)
                        process.wait(timeout=10)
                    raise RuntimeError(
                        f"Training exceeded the hard 40% GPU-memory limit: {process_memory:.0f} MiB"
                    )
                window.append(sample["utilization"])
                rolling = sum(window) / len(window)
                if not stopped and len(window) == window.maxlen and rolling > utilization_limit:
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
                    "rolling_utilization": rolling,
                    "training_stopped": stopped,
                    "active_utilization_limit_percent": utilization_limit,
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
        "configured_memory_fraction": memory_fraction,
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    limits = config["limits"]
    if not 0.0 < float(limits["gpu_memory_fraction"]) <= 0.40:
        raise ValueError("gpu_memory_fraction must be in (0, 0.40]")
    if not 0.0 < float(limits["utilization_percent"]) < 40.0:
        raise ValueError("utilization_percent must be below 40")
    boosted = limits.get("boosted_window", {})
    if boosted and not float(limits["utilization_percent"]) < float(boosted["utilization_percent"]) <= 100.0:
        raise ValueError("boosted utilization must be above daytime limit and at most 100")
    task = config["tasks"][args.task]
    rounds = configured_rounds(task, args.max_rounds, args.round_name)
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
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            monitor_summary = monitor_process(process, monitor_path, limits)
        report_path = model_dir / "training_run.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        result = {
            "name": name,
            "status": "complete" if monitor_summary["return_code"] == 0 else "failed",
            "monitor": monitor_summary,
            "metric": metric_from_report(report, task["selection_metric"]),
            "metric_name": task["selection_metric"],
        }
        if monitor_summary["return_code"] == 0:
            result["archive"] = archive_round(round_dir)
        summary["rounds"].append(result)
        (run_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if monitor_summary["return_code"] != 0:
            break

    completed = [item for item in summary["rounds"] if item.get("status") == "complete" and item.get("metric") is not None]
    if completed:
        reverse = bool(task["greater_is_better"])
        summary["selected_round"] = sorted(completed, key=lambda item: item["metric"], reverse=reverse)[0]["name"]
    summary["status"] = "complete" if len(summary["rounds"]) == len(rounds) and all(
        item["status"] in {"complete", "dry_run"} for item in summary["rounds"]
    ) else "failed"
    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_root": str(run_root), **summary}, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

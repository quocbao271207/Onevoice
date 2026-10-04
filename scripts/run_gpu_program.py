"""Continue the locked MT -> evaluation -> ASR -> evaluation GPU program.

The program is deliberately sequential: every child command owns the GPU alone
and applies the shared limits in ``configs/gpu_rounds.yaml``. Candidate gate
failures are evidence, not infrastructure errors, so ASR still runs after a
valid MT evaluation that returns exit code 2.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, state: dict[str, Any]) -> None:
    """Atomically persist progress so a watcher never reads partial JSON."""
    state["updated_at"] = utc_now()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_process(pid: int, poll_seconds: float, state_path: Path, state: dict[str, Any]) -> None:
    state["stage"] = "waiting_for_mt"
    state["wait_pid"] = pid
    write_state(state_path, state)
    while process_exists(pid):
        time.sleep(poll_seconds)


def read_complete_run(run_root: Path, expected_task: str) -> dict[str, Any]:
    summary_path = run_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing run summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("task") != expected_task:
        raise ValueError(f"expected {expected_task} run, found {summary.get('task')!r}")
    if summary.get("status") != "complete":
        raise ValueError(f"{expected_task} run is not complete: {summary.get('status')!r}")
    return summary


def selected_adapter(run_root: Path, summary: dict[str, Any]) -> Path:
    selected = str(summary.get("selected_round") or "")
    if not selected:
        raise ValueError(f"run has no selected_round: {run_root}")
    matches = sorted(run_root.glob(f"[0-9][0-9]-{selected}/model"))
    if len(matches) != 1:
        raise ValueError(f"expected one model directory for {selected!r}, found {len(matches)}")
    adapter = matches[0]
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"selected adapter is incomplete: {adapter}")
    return adapter


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid checkpoint directory name: {path}") from exc


def latest_checkpoint(model_dir: Path) -> Path:
    checkpoints = [
        path
        for path in model_dir.glob("checkpoint-*")
        if path.is_dir() and (path / "trainer_state.json").is_file()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"no resumable checkpoint under {model_dir}")
    return max(checkpoints, key=checkpoint_step)


def validation_history(checkpoint: Path, metric_name: str = "eval_loss") -> list[dict[str, float]]:
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    history = [
        {"step": float(item["step"]), "metric": float(item[metric_name])}
        for item in trainer_state.get("log_history", [])
        if item.get("step") is not None and item.get(metric_name) is not None
    ]
    return sorted(history, key=lambda item: item["step"])


def should_extend_mt(
    history: list[dict[str, float]],
    *,
    recent_points: int = 4,
    minimum_relative_gain: float = 0.003,
) -> tuple[bool, dict[str, Any]]:
    """Continue only when recent validation—not locked test—still improves."""
    if len(history) < recent_points:
        return False, {"reason": "insufficient_validation_points", "points": len(history)}
    recent = history[-recent_points:]
    start = recent[0]["metric"]
    end = recent[-1]["metric"]
    relative_gain = (start - end) / max(abs(start), 1e-12)
    best_is_latest = end == min(item["metric"] for item in history)
    monotonic = all(
        current["metric"] < previous["metric"]
        for previous, current in zip(recent, recent[1:])
    )
    decision = best_is_latest and monotonic and relative_gain >= minimum_relative_gain
    return decision, {
        "reason": "recent_validation_improving" if decision else "validation_plateau_or_regression",
        "recent": recent,
        "relative_gain": relative_gain,
        "minimum_relative_gain": minimum_relative_gain,
        "best_is_latest": best_is_latest,
        "monotonic": monotonic,
    }


def run_stage(
    *,
    name: str,
    command: list[str],
    log_path: Path,
    state_path: Path,
    state: dict[str, Any],
    accepted_codes: set[int],
) -> int:
    state["stage"] = name
    state.setdefault("stages", {})[name] = {
        "status": "running",
        "started_at": utc_now(),
        "command": command,
        "log": str(log_path),
    }
    write_state(state_path, state)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            start_new_session=True,
        )
    stage = state["stages"][name]
    stage.update(
        {
            "status": "complete" if result.returncode in accepted_codes else "error",
            "completed_at": utc_now(),
            "return_code": result.returncode,
        }
    )
    write_state(state_path, state)
    if result.returncode not in accepted_codes:
        raise RuntimeError(f"stage {name} failed with exit code {result.returncode}; see {log_path}")
    return result.returncode


def newest_complete_run(output_root: Path, task: str) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(output_root.glob(f"{task}-*"), key=lambda path: path.stat().st_mtime)
    for run_root in reversed(candidates):
        try:
            return run_root, read_complete_run(run_root, task)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            continue
    raise FileNotFoundError(f"no complete {task} run under {output_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--mt-run", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only a previously recorded waiting_for_mt state.",
    )
    args = parser.parse_args()

    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.wait_pid is not None and args.wait_pid <= 0:
        parser.error("--wait-pid must be positive")

    state_dir = args.state_dir.resolve()
    mt_run = args.mt_run.resolve()
    state_path = state_dir / "program_state.json"
    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"cannot resume without {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if Path(state.get("mt_run", "")).resolve() != mt_run:
            raise ValueError("resume --mt-run does not match recorded state")
        if state.get("stage") != "waiting_for_mt" or state.get("stages"):
            raise ValueError("automatic resume is only safe from an idle waiting_for_mt state")
        state["execution_status"] = "running"
        state.pop("error", None)
    else:
        state_dir.mkdir(parents=True, exist_ok=False)
        state = {
            "created_at": utc_now(),
            "execution_status": "running",
            "promotion_allowed": False,
            "mt_run": str(mt_run),
            "resource_policy": (
                "One GPU child at a time; every trainer/evaluator enforces the central 35% allocation, "
                "40% hard memory stop, and scheduled utilization limits."
            ),
            "stages": {},
        }
    write_state(state_path, state)

    try:
        if args.wait_pid is not None:
            wait_for_process(args.wait_pid, args.poll_seconds, state_path, state)

        mt_summary = read_complete_run(mt_run, "mt")
        mt_adapter = selected_adapter(mt_run, mt_summary)
        resume_checkpoint = latest_checkpoint(mt_adapter)
        extend_mt, extension_evidence = should_extend_mt(
            validation_history(resume_checkpoint)
        )
        state["mt_extension_decision"] = {
            "extend": extend_mt,
            "evidence_checkpoint": str(resume_checkpoint),
            "initial_adapter": str(mt_adapter),
            **extension_evidence,
        }
        if extend_mt:
            extension_output_root = state_dir / "mt-extension-runs"
            extension_output_root.mkdir()
            run_stage(
                name="mt_validation_extension",
                command=[
                    args.python,
                    str(ROOT / "scripts" / "run_gpu_rounds.py"),
                    "--task",
                    "mt",
                    "--config",
                    str(ROOT / "configs" / "gpu_rounds.yaml"),
                    "--output-root",
                    str(extension_output_root),
                    "--round-name",
                    "final-r8-lr1e4-full",
                    "--initial-adapter",
                    str(mt_adapter),
                    "--epochs",
                    "2",
                    "--learning-rate",
                    "5e-5",
                ],
                log_path=state_dir / "mt_validation_extension_stage.log",
                state_path=state_path,
                state=state,
                accepted_codes={0},
            )
            extension_run, extension_summary = newest_complete_run(extension_output_root, "mt")
            mt_adapter = selected_adapter(extension_run, extension_summary)
            state["mt_extension_run"] = str(extension_run)
        state["mt_adapter"] = str(mt_adapter)
        write_state(state_path, state)

        mt_candidate_dir = state_dir / "mt-candidate"
        mt_gate_code = run_stage(
            name="mt_candidate",
            command=[
                args.python,
                str(ROOT / "scripts" / "run_mt_candidate_suite.py"),
                "--adapter",
                str(mt_adapter),
                "--output-dir",
                str(mt_candidate_dir),
                "--device",
                "cuda",
                "--precision",
                "bf16",
                "--batch-size",
                "8",
                "--num-beams",
                "4",
            ],
            log_path=state_dir / "mt_candidate_stage.log",
            state_path=state_path,
            state=state,
            accepted_codes={0, 2},
        )

        asr_output_root = state_dir / "asr-runs"
        asr_output_root.mkdir()
        run_stage(
            name="asr_training",
            command=[
                args.python,
                str(ROOT / "scripts" / "run_gpu_rounds.py"),
                "--task",
                "asr",
                "--config",
                str(ROOT / "configs" / "gpu_rounds.yaml"),
                "--output-root",
                str(asr_output_root),
            ],
            log_path=state_dir / "asr_training_stage.log",
            state_path=state_path,
            state=state,
            accepted_codes={0},
        )
        asr_run, asr_summary = newest_complete_run(asr_output_root, "asr")
        asr_adapter = selected_adapter(asr_run, asr_summary)
        state["asr_run"] = str(asr_run)
        state["asr_adapter"] = str(asr_adapter)
        write_state(state_path, state)

        asr_candidate_dir = state_dir / "asr-candidate"
        asr_gate_code = run_stage(
            name="asr_candidate",
            command=[
                args.python,
                str(ROOT / "scripts" / "run_asr_candidate_suite.py"),
                "--adapter",
                str(asr_adapter),
                "--output-dir",
                str(asr_candidate_dir),
                "--device",
                "cuda",
                "--precision",
                "bf16",
                "--batch-size",
                "4",
                "--num-beams",
                "1",
            ],
            log_path=state_dir / "asr_candidate_stage.log",
            state_path=state_path,
            state=state,
            accepted_codes={0, 2},
        )
        state["stage"] = "finished"
        state["execution_status"] = "complete"
        state["promotion_allowed"] = mt_gate_code == 0 and asr_gate_code == 0
        state["gate_results"] = {
            "mt": "pass" if mt_gate_code == 0 else "fail",
            "asr": "pass" if asr_gate_code == 0 else "fail",
        }
        write_state(state_path, state)
        return 0
    except Exception as exc:
        state["execution_status"] = "error"
        state["error"] = f"{type(exc).__name__}: {exc}"
        write_state(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

"""Continue the locked MT -> evaluation -> ASR -> evaluation GPU program.

The program is deliberately sequential: every child command owns the GPU alone
and applies the shared limits in ``configs/gpu_rounds.yaml``. Candidate gate
failures are evidence, not infrastructure errors, so ASR still runs after a
valid MT evaluation that returns exit code 2.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.candidate_evidence import (  # noqa: E402
    MAX_EVIDENCE_ARCHIVE_BYTES,
    adapter_identity,
    canonical_sha256,
    verify_evidence_archive,
)
from src.pipeline.adapter_evidence import stable_adapter_tree_manifest  # noqa: E402
from src.pipeline.durable_json import write_durable_json  # noqa: E402
from src.pipeline.evidence_paths import (  # noqa: E402
    is_link_or_junction,
    resolve_regular_directory_without_links,
    resolve_regular_file_without_links,
)
from src.pipeline.stable_json import (  # noqa: E402
    read_stable_json_mapping,
)
from src.pipeline.stable_jsonl import read_stable_jsonl_mappings  # noqa: E402
from src.utils.bounded_file import sha256_stable_regular_file  # noqa: E402


MT_CANDIDATE_NUM_BEAMS = 1
MAX_PROGRAM_STATE_BYTES = 10_000_000
MAX_TRAINING_SUMMARY_BYTES = 10_000_000
MAX_TRAINER_STATE_BYTES = 10_000_000
MAX_CANDIDATE_GATE_BYTES = 10_000_000
MAX_PREDICTION_PROVENANCE_BYTES = 10_000_000
MAX_PREDICTION_BYTES = 250_000_000
MAX_PREDICTION_LINE_BYTES = 2_000_000
MAX_PREDICTION_ROWS = 100_000
MAX_LIVE_EVIDENCE_FILE_BYTES = 1_000_000_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHECKPOINT_RE = re.compile(r"^checkpoint-([1-9]\d*)$")
ROUND_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PROVENANCE_KEYS = {
    "schema_version",
    "specification",
    "specification_sha256",
    "predictions",
    "decoding",
    "runtime",
}
PREDICTION_RECORD_KEYS = {"path", "bytes", "sha256", "rows"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, state: dict[str, Any]) -> None:
    """Durably persist strict progress JSON through an owned random temp."""
    state["updated_at"] = utc_now()
    write_durable_json(
        path,
        state,
        maximum_bytes=MAX_PROGRAM_STATE_BYTES,
        label="GPU program state",
    )


def read_program_state(path: Path) -> dict[str, Any]:
    return read_stable_json_mapping(
        path,
        maximum_bytes=MAX_PROGRAM_STATE_BYTES,
        label="GPU program state",
    ).mapping


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number")
    return parsed


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    return Path(os.path.abspath(value))


def _recorded_adapter(state: dict[str, Any], field: str, label: str) -> Path:
    value = state.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Recorded {label.lower()} path is missing")
    adapter = resolve_regular_directory_without_links(value, label=label)
    resolve_regular_file_without_links(
        adapter / "adapter_config.json",
        label=f"{label} config",
        maximum_bytes=MAX_TRAINING_SUMMARY_BYTES,
    )
    return adapter


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
    summary = read_stable_json_mapping(
        summary_path,
        maximum_bytes=MAX_TRAINING_SUMMARY_BYTES,
        label="GPU training summary",
    ).mapping
    if summary.get("task") != expected_task:
        raise ValueError(f"expected {expected_task} run, found {summary.get('task')!r}")
    if summary.get("status") != "complete":
        raise ValueError(f"{expected_task} run is not complete: {summary.get('status')!r}")
    return summary


def selected_adapter(run_root: Path, summary: dict[str, Any]) -> Path:
    selected = summary.get("selected_round")
    if not isinstance(selected, str) or not ROUND_NAME_RE.fullmatch(selected):
        raise ValueError(f"run has no selected_round: {run_root}")
    run_root = resolve_regular_directory_without_links(
        run_root,
        label="GPU training run",
    )
    matches = sorted(run_root.glob(f"[0-9][0-9]-{selected}/model"))
    if len(matches) != 1:
        raise ValueError(f"expected one model directory for {selected!r}, found {len(matches)}")
    adapter = resolve_regular_directory_without_links(
        matches[0],
        label="Selected training adapter",
    )
    resolve_regular_file_without_links(
        adapter / "adapter_config.json",
        label="Selected adapter config",
        maximum_bytes=MAX_TRAINING_SUMMARY_BYTES,
    )
    return adapter


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.fullmatch(path.name)
    if not match:
        raise ValueError(f"invalid checkpoint directory name: {path}")
    return int(match.group(1))


def latest_checkpoint(model_dir: Path) -> Path:
    model_dir = resolve_regular_directory_without_links(
        model_dir,
        label="Training model directory",
    )
    checkpoints = []
    for path in model_dir.glob("checkpoint-*"):
        if is_link_or_junction(path):
            raise ValueError(f"Checkpoint cannot be a symlink or junction: {path}")
        if path.is_dir() and (path / "trainer_state.json").is_file():
            checkpoints.append(path)
    if not checkpoints:
        raise FileNotFoundError(f"no resumable checkpoint under {model_dir}")
    return max(checkpoints, key=checkpoint_step)


def validation_history(checkpoint: Path, metric_name: str = "eval_loss") -> list[dict[str, float]]:
    trainer_state = read_stable_json_mapping(
        checkpoint / "trainer_state.json",
        maximum_bytes=MAX_TRAINER_STATE_BYTES,
        label="Trainer validation state",
    ).mapping
    raw_history = trainer_state.get("log_history")
    if not isinstance(raw_history, list):
        raise ValueError("trainer_state.log_history must be a list")
    history: list[dict[str, float]] = []
    observed_steps: set[float] = set()
    for index, item in enumerate(raw_history, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"trainer_state.log_history row {index} must be an object")
        if item.get("step") is None or item.get(metric_name) is None:
            continue
        step = _finite_number(item["step"], f"trainer_state row {index} step")
        metric = _finite_number(
            item[metric_name],
            f"trainer_state row {index} {metric_name}",
        )
        if step < 0 or metric < 0:
            raise ValueError("Trainer validation step and metric must be non-negative")
        if step in observed_steps:
            raise ValueError(f"Trainer validation history has duplicate step {step}")
        observed_steps.add(step)
        history.append({"step": step, "metric": metric})
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
    output_root = resolve_regular_directory_without_links(
        output_root,
        label=f"{task} training output root",
    )
    candidates = list(output_root.glob(f"{task}-*"))
    if any(is_link_or_junction(path) for path in candidates):
        raise ValueError(f"{task} training output cannot contain linked runs")
    candidates.sort(key=lambda path: path.lstat().st_mtime_ns)
    for run_root in reversed(candidates):
        try:
            return run_root, read_complete_run(run_root, task)
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
    raise FileNotFoundError(f"no complete {task} run under {output_root}")


def resume_kind(state: dict[str, Any], mt_run: Path) -> str:
    if _absolute_path(state.get("mt_run"), "Recorded MT run") != mt_run:
        raise ValueError("resume --mt-run does not match recorded state")
    stages = state.get("stages")
    if state.get("stage") == "waiting_for_mt" and not stages:
        return "from_wait"
    mt_stage = stages.get("mt_candidate") if isinstance(stages, dict) else None
    if (
        state.get("stage") == "mt_candidate"
        and isinstance(mt_stage, dict)
        and mt_stage.get("status") == "running"
    ):
        return "after_recovered_mt_candidate"
    asr_training = stages.get("asr_training") if isinstance(stages, dict) else None
    if (
        state.get("stage") == "asr_training"
        and isinstance(asr_training, dict)
        and asr_training.get("status") == "running"
    ):
        return "after_recovered_asr_training"
    asr_candidate = stages.get("asr_candidate") if isinstance(stages, dict) else None
    if (
        state.get("stage") == "asr_candidate"
        and isinstance(asr_candidate, dict)
        and asr_candidate.get("status") == "running"
    ):
        return "after_recovered_asr_candidate"
    raise ValueError(
        "automatic resume is only safe from waiting_for_mt or a verified out-of-band "
        "MT/ASR stage result"
    )


def completed_candidate_code(state: dict[str, Any], name: str) -> int:
    stage = state.get("stages", {}).get(name)
    if not isinstance(stage, dict) or stage.get("status") != "complete":
        raise ValueError(f"required candidate stage is not complete: {name}")
    code = stage.get("return_code")
    if code not in {0, 2}:
        raise ValueError(f"candidate stage has invalid return code: {name}={code!r}")
    return int(code)


def verify_prediction_provenance(
    prediction_path: Path,
    *,
    expected_task: str,
    expected_adapter_identity: dict[str, Any],
) -> None:
    """Rebind archived predictions to their exact live adapter and specification."""
    provenance_path = prediction_path.with_suffix(
        prediction_path.suffix + ".provenance.json"
    )
    provenance_document = read_stable_json_mapping(
        provenance_path,
        maximum_bytes=MAX_PREDICTION_PROVENANCE_BYTES,
        label="Candidate prediction provenance",
    )
    provenance = provenance_document.mapping
    specification = provenance.get("specification")
    if (
        set(provenance) != PROVENANCE_KEYS
        or isinstance(provenance.get("schema_version"), bool)
        or provenance.get("schema_version") != 1
        or not isinstance(specification, dict)
        or specification.get("task") != expected_task
        or specification.get("adapter") != expected_adapter_identity
        or _sha256(
            provenance.get("specification_sha256"),
            "Prediction specification digest",
        )
        != canonical_sha256(specification)
    ):
        raise ValueError(f"prediction provenance does not bind expected adapter: {prediction_path}")

    prediction_record = provenance.get("predictions")
    prediction_document = read_stable_jsonl_mappings(
        prediction_path,
        maximum_bytes=MAX_PREDICTION_BYTES,
        maximum_line_bytes=MAX_PREDICTION_LINE_BYTES,
        maximum_rows=MAX_PREDICTION_ROWS,
        label="Candidate predictions",
    )
    if (
        not isinstance(prediction_record, dict)
        or set(prediction_record) != PREDICTION_RECORD_KEYS
        or _absolute_path(
            prediction_record.get("path"),
            "Prediction provenance path",
        )
        != prediction_document.path
        or _nonnegative_integer(
            prediction_record.get("bytes"),
            "Prediction provenance bytes",
        )
        != prediction_document.bytes
        or _sha256(
            prediction_record.get("sha256"),
            "Prediction provenance digest",
        )
        != prediction_document.sha256
        or not isinstance(provenance.get("decoding"), dict)
        or not isinstance(provenance.get("runtime"), dict)
    ):
        raise ValueError(f"prediction provenance is inconsistent: {prediction_path}")
    if _nonnegative_integer(
        prediction_record.get("rows"),
        "Prediction provenance rows",
    ) != len(prediction_document.rows):
        raise ValueError(f"prediction provenance row count is inconsistent: {prediction_path}")
    persisted = read_stable_json_mapping(
        provenance_path,
        maximum_bytes=MAX_PREDICTION_PROVENANCE_BYTES,
        label="Candidate prediction provenance",
        expected_sha256=provenance_document.sha256,
    )
    if persisted.bytes != provenance_document.bytes:
        raise RuntimeError("Candidate prediction provenance changed while verifying")


def verified_candidate_result(
    output_dir: Path,
    *,
    task: str,
    expected_adapter: Path,
) -> dict[str, Any]:
    gate_path = output_dir / "candidate_gate.json"
    gate_document = read_stable_json_mapping(
        gate_path,
        maximum_bytes=MAX_CANDIDATE_GATE_BYTES,
        label="Candidate gate",
    )
    gate = gate_document.mapping
    status = gate.get("status")
    promotion_allowed = gate.get("promotion_allowed")
    if status not in {"pass", "fail"} or gate.get("error") is not None:
        raise ValueError(f"candidate gate has no valid terminal result: {gate_path}")
    if not isinstance(promotion_allowed, bool) or promotion_allowed != (
        status == "pass"
    ):
        raise ValueError(f"candidate gate promotion flag is inconsistent: {gate_path}")
    archive_path = output_dir.with_suffix(".tar.gz")
    manifest = verify_evidence_archive(archive_path)
    records = {str(item["path"]): item for item in manifest["files"]}
    stems = (f"{task}_candidate", f"{task}_clinical_candidate")
    expected_identity = adapter_identity(expected_adapter)
    if expected_identity is None:  # pragma: no cover - expected_adapter is always a Path
        raise ValueError("candidate recovery requires an adapter")
    if _absolute_path(gate.get("adapter"), "Candidate gate adapter") != Path(
        expected_identity["path"]
    ):
        raise ValueError(f"candidate gate adapter does not match program state: {gate_path}")
    required_names = {"candidate_gate.json"}
    for stem in stems:
        required_names.update(
            {
                f"{stem}.json",
                f"{stem}.log",
                f"{stem}_predictions.jsonl",
                f"{stem}_predictions.jsonl.provenance.json",
                f"{stem}_resource_monitor.jsonl",
            }
        )
    mutable_log_mismatches = []
    for name in required_names:
        path = output_dir / name
        archive_name = f"{output_dir.name}/{name}"
        record = records.get(archive_name)
        if record is None:
            raise ValueError(f"candidate evidence does not match archive: {path}")
        resolved = resolve_regular_file_without_links(
            path,
            label=f"Candidate evidence {name}",
            maximum_bytes=MAX_LIVE_EVIDENCE_FILE_BYTES,
        )
        digest, size = sha256_stable_regular_file(
            resolved,
            maximum_bytes=MAX_LIVE_EVIDENCE_FILE_BYTES,
            label=f"Candidate evidence {name}",
        )
        matches_archive = int(record["bytes"]) == size and str(
            record["sha256"]
        ) == digest
        if not matches_archive:
            if name.endswith(".log"):
                mutable_log_mismatches.append(name)
            else:
                raise ValueError(f"candidate evidence does not match archive: {path}")
    for stem in stems:
        verify_prediction_provenance(
            output_dir / f"{stem}_predictions.jsonl",
            expected_task=task,
            expected_adapter_identity=expected_identity,
        )
    if adapter_identity(expected_adapter) != expected_identity:
        raise RuntimeError("Candidate adapter changed while verifying recovery evidence")
    persisted_gate = read_stable_json_mapping(
        gate_path,
        maximum_bytes=MAX_CANDIDATE_GATE_BYTES,
        label="Candidate gate",
        expected_sha256=gate_document.sha256,
    )
    if persisted_gate.bytes != gate_document.bytes:
        raise RuntimeError("Candidate gate changed while verifying recovery evidence")
    return {
        "return_code": 0 if status == "pass" else 2,
        "status": status,
        "archive": str(archive_path),
        "archive_sha256": str(manifest["archive_sha256"]),
        "archive_bytes": int(manifest["archive_bytes"]),
        "adapter_manifest_sha256": str(expected_identity["manifest_sha256"]),
        "local_log_mismatches": mutable_log_mismatches,
    }


def record_recovered_candidate(
    *,
    name: str,
    evidence: dict[str, Any],
    state_path: Path,
    state: dict[str, Any],
) -> int:
    stage = state["stages"][name]
    stage.update(
        {
            "status": "complete",
            "completed_at": utc_now(),
            "return_code": int(evidence["return_code"]),
            "recovered_out_of_band": True,
            "evidence_archive": evidence["archive"],
            "evidence_archive_sha256": evidence["archive_sha256"],
            "evidence_archive_bytes": int(evidence["archive_bytes"]),
            "adapter_manifest_sha256": evidence["adapter_manifest_sha256"],
            "local_log_mismatches": list(evidence["local_log_mismatches"]),
        }
    )
    state["stage"] = name
    write_state(state_path, state)
    return int(evidence["return_code"])


def verify_live_directory_against_archive(
    directory: Path,
    *,
    manifest: dict[str, Any],
    archive_prefix: str,
) -> dict[str, int]:
    """Require every live file under ``directory`` to match archived evidence."""
    prefix = archive_prefix.rstrip("/") + "/"
    expected = {
        str(item["path"])[len(prefix) :]: item
        for item in manifest["files"]
        if str(item["path"]).startswith(prefix)
    }
    if not expected:
        raise ValueError(f"verified archive has no files under {archive_prefix}")

    observed_manifest = stable_adapter_tree_manifest(
        directory,
        label="Live training adapter",
        required_relative_paths=(),
    )
    observed = {
        str(item["path"]): item for item in observed_manifest["files"]
    }
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        unexpected = sorted(set(observed) - set(expected))
        raise ValueError(
            f"live directory does not match verified archive: {directory}; "
            f"missing={missing}, unexpected={unexpected}"
        )

    for relative_path, observed_record in observed.items():
        expected_record = expected[relative_path]
        if (
            int(expected_record["bytes"]) != int(observed_record["bytes"])
            or str(expected_record["sha256"]) != observed_record["sha256"]
        ):
            raise ValueError(
                "live file does not match verified archive: "
                f"{directory / relative_path}"
            )
    return {
        "file_count": int(observed_manifest["file_count"]),
        "content_bytes": int(observed_manifest["bytes"]),
    }


def verified_training_result(output_root: Path, *, task: str) -> dict[str, Any]:
    run_root, summary = newest_complete_run(output_root, task)
    adapter = selected_adapter(run_root, summary)
    rounds = summary.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError(f"training summary has no rounds: {run_root}")
    verified_archives = []
    for index, round_record in enumerate(rounds, start=1):
        if not isinstance(round_record, dict) or round_record.get("status") != "complete":
            raise ValueError(f"training round is not complete: {run_root} row {index}")
        archive_record = round_record.get("archive")
        if not isinstance(archive_record, dict):
            raise ValueError(f"training round has no archive evidence: {run_root} row {index}")
        round_name = round_record.get("name")
        if not isinstance(round_name, str) or not ROUND_NAME_RE.fullmatch(round_name):
            raise ValueError(f"training round has an invalid name: {run_root} row {index}")
        matches = list(run_root.glob(f"{index:02d}-{round_name}.tar.gz"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one archive for training round {index}: {round_name}"
            )
        archive_path = resolve_regular_file_without_links(
            matches[0],
            label="Training round archive",
            maximum_bytes=MAX_EVIDENCE_ARCHIVE_BYTES,
        )
        if _absolute_path(
            archive_record.get("path"),
            "Training archive record path",
        ) != archive_path:
            raise ValueError(f"training archive path mismatch: {archive_path}")
        manifest = verify_evidence_archive(archive_path)
        checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
        manifest_path = archive_path.with_suffix(archive_path.suffix + ".manifest.json")
        if (
            _nonnegative_integer(
                archive_record.get("bytes"),
                "Training archive bytes",
            )
            != int(manifest["archive_bytes"])
            or _sha256(
                archive_record.get("sha256"),
                "Training archive digest",
            )
            != manifest["archive_sha256"]
            or _absolute_path(
                archive_record.get("checksum"),
                "Training archive checksum path",
            )
            != checksum_path
            or _absolute_path(
                archive_record.get("manifest"),
                "Training archive manifest path",
            )
            != manifest_path
        ):
            raise ValueError(f"training archive metadata mismatch: {archive_path}")
        verified_archives.append(
            {
                "path": str(archive_path),
                "bytes": int(manifest["archive_bytes"]),
                "sha256": str(manifest["archive_sha256"]),
            }
        )

    selected_round_dir = adapter.parent
    selected_archive = next(
        (
            item
            for item in verified_archives
            if Path(item["path"]).name == f"{selected_round_dir.name}.tar.gz"
        ),
        None,
    )
    if selected_archive is None:
        raise ValueError(f"selected adapter round has no verified archive: {adapter}")
    selected_manifest = verify_evidence_archive(Path(selected_archive["path"]))
    adapter_snapshot = verify_live_directory_against_archive(
        adapter,
        manifest=selected_manifest,
        archive_prefix=f"{selected_round_dir.name}/model",
    )
    return {
        "run_root": str(run_root),
        "adapter": str(adapter),
        "adapter_file_count": adapter_snapshot["file_count"],
        "adapter_content_bytes": adapter_snapshot["content_bytes"],
        "verified_archives": verified_archives,
    }


def record_recovered_training(
    *,
    name: str,
    evidence: dict[str, Any],
    state_path: Path,
    state: dict[str, Any],
) -> None:
    state["stages"][name].update(
        {
            "status": "complete",
            "completed_at": utc_now(),
            "return_code": 0,
            "recovered_out_of_band": True,
            "run_root": evidence["run_root"],
            "adapter": evidence["adapter"],
            "adapter_file_count": int(evidence["adapter_file_count"]),
            "adapter_content_bytes": int(evidence["adapter_content_bytes"]),
            "verified_archives": evidence["verified_archives"],
        }
    )
    state["stage"] = name
    write_state(state_path, state)


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
        help=(
            "Resume a recorded waiting_for_mt state, or reconcile a completed out-of-band "
            "MT/ASR candidate bundle or ASR multi-round run after verifying its complete "
            "immutable evidence before continuing."
        ),
    )
    args = parser.parse_args()

    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.wait_pid is not None and args.wait_pid <= 0:
        parser.error("--wait-pid must be positive")

    state_dir = Path(os.path.abspath(args.state_dir))
    mt_run = resolve_regular_directory_without_links(
        args.mt_run,
        label="MT training run",
    )
    state_path = state_dir / "program_state.json"
    resume_from = "new"
    recovered_mt: dict[str, Any] | None = None
    recovered_asr_training: dict[str, Any] | None = None
    recovered_asr_candidate: dict[str, Any] | None = None
    if args.resume:
        state = read_program_state(state_path)
        resume_from = resume_kind(state, mt_run)
        if resume_from == "after_recovered_mt_candidate":
            mt_adapter = _recorded_adapter(state, "mt_adapter", "Recorded MT adapter")
            recovered_mt = verified_candidate_result(
                state_dir / "mt-candidate",
                task="mt",
                expected_adapter=mt_adapter,
            )
        elif resume_from == "after_recovered_asr_training":
            completed_candidate_code(state, "mt_candidate")
            recovered_asr_training = verified_training_result(
                state_dir / "asr-runs", task="asr"
            )
        elif resume_from == "after_recovered_asr_candidate":
            completed_candidate_code(state, "mt_candidate")
            asr_adapter = _recorded_adapter(
                state,
                "asr_adapter",
                "Recorded ASR adapter",
            )
            recovered_asr_candidate = verified_candidate_result(
                state_dir / "asr-candidate",
                task="asr",
                expected_adapter=asr_adapter,
            )
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
                "40% hard memory stop, 70% rolling utilization limit, and 74% hard utilization guard."
            ),
            "stages": {},
        }
    write_state(state_path, state)

    try:
        if resume_from == "after_recovered_mt_candidate":
            assert recovered_mt is not None
            mt_gate_code = record_recovered_candidate(
                name="mt_candidate",
                evidence=recovered_mt,
                state_path=state_path,
                state=state,
            )
        elif resume_from in {
            "after_recovered_asr_training",
            "after_recovered_asr_candidate",
        }:
            mt_gate_code = completed_candidate_code(state, "mt_candidate")
        else:
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
                extension_run, extension_summary = newest_complete_run(
                    extension_output_root, "mt"
                )
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
                    str(MT_CANDIDATE_NUM_BEAMS),
                ],
                log_path=state_dir / "mt_candidate_stage.log",
                state_path=state_path,
                state=state,
                accepted_codes={0, 2},
            )

        if resume_from == "after_recovered_asr_candidate":
            assert recovered_asr_candidate is not None
            asr_gate_code = record_recovered_candidate(
                name="asr_candidate",
                evidence=recovered_asr_candidate,
                state_path=state_path,
                state=state,
            )
        else:
            asr_output_root = state_dir / "asr-runs"
            if resume_from == "after_recovered_asr_training":
                assert recovered_asr_training is not None
                record_recovered_training(
                    name="asr_training",
                    evidence=recovered_asr_training,
                    state_path=state_path,
                    state=state,
                )
                asr_run = Path(recovered_asr_training["run_root"])
                asr_adapter = Path(recovered_asr_training["adapter"])
            else:
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
        state["candidate_a_gate_pass"] = mt_gate_code == 0 and asr_gate_code == 0
        state["promotion_allowed"] = False
        state["promotion_deferred_to_model_bakeoff"] = True
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

import json
import sys
from pathlib import Path

import pytest

from scripts.run_gpu_program import (
    latest_checkpoint,
    newest_complete_run,
    read_complete_run,
    run_stage,
    selected_adapter,
    should_extend_mt,
    validation_history,
    write_state,
)


def make_run(root: Path, *, task: str = "mt", status: str = "complete") -> Path:
    run = root / f"{task}-20261005-000000"
    model = run / "01-final" / "model"
    model.mkdir(parents=True)
    (model / "adapter_config.json").write_text("{}", encoding="utf-8")
    (run / "summary.json").write_text(
        json.dumps(
            {
                "task": task,
                "status": status,
                "selected_round": "final",
                "rounds": [{"name": "final", "status": status, "metric": 1.0}],
            }
        ),
        encoding="utf-8",
    )
    return run


def test_selected_adapter_requires_completed_selected_model(tmp_path: Path):
    run = make_run(tmp_path)
    summary = read_complete_run(run, "mt")
    assert selected_adapter(run, summary) == run / "01-final" / "model"


def test_read_complete_run_rejects_failed_training(tmp_path: Path):
    run = make_run(tmp_path, status="failed")
    with pytest.raises(ValueError, match="not complete"):
        read_complete_run(run, "mt")


def test_newest_complete_run_skips_incomplete_newer_run(tmp_path: Path):
    complete = make_run(tmp_path, task="asr")
    newer = tmp_path / "asr-20261005-000001"
    newer.mkdir()
    (newer / "summary.json").write_text(
        json.dumps({"task": "asr", "status": "failed"}), encoding="utf-8"
    )
    assert newest_complete_run(tmp_path, "asr")[0] == complete


def test_run_stage_accepts_candidate_gate_failure_as_evidence(tmp_path: Path):
    state_path = tmp_path / "state.json"
    state = {"stages": {}}
    write_state(state_path, state)
    code = run_stage(
        name="candidate",
        command=[sys.executable, "-c", "raise SystemExit(2)"],
        log_path=tmp_path / "candidate.log",
        state_path=state_path,
        state=state,
        accepted_codes={0, 2},
    )
    assert code == 2
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["stages"]["candidate"]["status"] == "complete"
    assert persisted["stages"]["candidate"]["return_code"] == 2


def test_mt_extension_uses_recent_validation_trend_only(tmp_path: Path):
    model = tmp_path / "model"
    checkpoint = model / "checkpoint-3500"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "log_history": [
                    {"step": 2000, "eval_loss": 1.66},
                    {"step": 2500, "eval_loss": 1.64},
                    {"step": 3000, "eval_loss": 1.62},
                    {"step": 3500, "eval_loss": 1.60},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert latest_checkpoint(model) == checkpoint
    extend, evidence = should_extend_mt(validation_history(checkpoint))
    assert extend is True
    assert evidence["best_is_latest"] is True
    assert evidence["monotonic"] is True


def test_mt_extension_stops_on_validation_plateau():
    extend, evidence = should_extend_mt(
        [
            {"step": 2000.0, "metric": 1.60},
            {"step": 2500.0, "metric": 1.59},
            {"step": 3000.0, "metric": 1.591},
            {"step": 3500.0, "metric": 1.5905},
        ]
    )
    assert extend is False
    assert evidence["reason"] == "validation_plateau_or_regression"

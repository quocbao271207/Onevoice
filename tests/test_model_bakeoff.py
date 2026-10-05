from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.run_model_bakeoff import (
    critical_safety_pass,
    interval_stronger,
    license_gate,
    report_score,
    run_stage,
    validate_candidate_matrix,
    validate_resources,
    validate_selection_artifacts,
    write_runtime_round_config,
)


ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return yaml.safe_load((ROOT / "configs/model_bakeoff.yaml").read_text(encoding="utf-8"))


def candidate(candidate_id: str) -> dict:
    for task in ("mt", "asr"):
        for item in config()["candidates"][task]:
            if item["id"] == candidate_id:
                return item
    raise AssertionError(candidate_id)


def test_bakeoff_fairness_and_selection_checksums_are_locked():
    data = config()
    validate_resources(data)
    validate_candidate_matrix(data)
    hashes = validate_selection_artifacts(data)
    assert set(hashes) == {"mt", "asr"}
    assert data["successive_halving"]["comparison_unit"] == "effective_train_samples"
    assert data["principles"]["one_gpu_child_at_a_time"] is True
    required = set(data["promotion_gate"]["critical_slices"] + data["promotion_gate"]["policy_slices"])
    for task in ("mt", "asr"):
        rows = [
            json.loads(line)
            for line in (ROOT / data["data"]["selection_dev"][task]["path"]).read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        ]
        assert required <= {category for row in rows for category in row.get("categories", [])}
        assert b"\r\n" not in (ROOT / data["data"]["selection_dev"][task]["path"]).read_bytes()


def test_vinai_license_fails_closed_before_gpu():
    item = candidate("mt_vinai_en2vi_v2")
    assert license_gate(item, set()) == (False, "legal_review_required")
    assert license_gate(item, {item["id"]}) == (True, "explicit_research_approval")
    assert item["license"]["production_eligible"] is False


def test_only_planned_or_candidate_a_statuses_are_allowed():
    validate_candidate_matrix(config())
    statuses = {
        item["status"]
        for task in ("mt", "asr")
        for item in config()["candidates"][task]
    }
    assert "benchmarked" not in statuses


def test_promotion_gate_requires_every_critical_slice_and_zero_failures():
    required = ["drug_name", "dose", "number", "unit", "negation"]
    report = {
        "categories": {
            name: {"samples": 2, "safety_failure_rate": 0.0} for name in required
        }
    }
    assert critical_safety_pass(report, required) == (True, [])
    report["categories"]["dose"]["safety_failure_rate"] = 0.5
    passed, failures = critical_safety_pass(report, required)
    assert passed is False
    assert failures == ["dose:nonzero"]


def test_stronger_requires_non_overlapping_confidence_intervals():
    assert interval_stronger([47.0, 49.0], [43.0, 46.0], greater_is_better=True)
    assert not interval_stronger([45.5, 47.0], [44.0, 46.0], greater_is_better=True)
    assert interval_stronger([0.15, 0.17], [0.18, 0.21], greater_is_better=False)


def test_asr_selection_rejects_bad_code_switch_wer():
    required = ["drug_name", "dose", "number", "unit", "negation", "terminology", "code_switch"]
    report = {
        "wer": 0.15,
        "wer_bootstrap_95ci": [0.14, 0.16],
        "categories": {
            name: {"samples": 1, "safety_failure_rate": 0.0} for name in required
        },
        "slices": {"code_switch": {"True": {"wer": 0.22}}},
    }
    score = report_score(report, "asr", None, required)
    assert score["safety_pass"] is False
    assert "code_switch_wer:above_policy" in score["safety_failures"]


def test_resume_skips_identical_completed_stage_and_rejects_changed_command(tmp_path: Path):
    state_path = tmp_path / "state.json"
    output = tmp_path / "output.json"
    log = tmp_path / "stage.log"
    state = {"stages": {}}
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
    ]
    run_stage("stage", command, state_path, state, log, [output])
    first_mtime = output.stat().st_mtime_ns
    run_stage("stage", command, state_path, state, log, [output])
    assert output.stat().st_mtime_ns == first_mtime
    with pytest.raises(ValueError, match="command changed"):
        run_stage("stage", command + ["changed"], state_path, state, log, [output])


def test_m2m100_runtime_round_builds_a_dry_run_command(tmp_path: Path):
    data = config()
    item = candidate("mt_m2m100_418m")
    runtime = tmp_path / "runtime.yaml"
    write_runtime_round_config(runtime, data, "mt", item, "en_to_vi", "pilot", 400)
    output = tmp_path / "runs"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_gpu_rounds.py"),
            "--task",
            "mt",
            "--config",
            str(runtime),
            "--output-root",
            str(output),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(next(output.glob("mt-*/summary.json")).read_text(encoding="utf-8"))
    command = summary["rounds"][0]["command"]
    assert command[command.index("--model-family") + 1] == "m2m100"
    assert command[command.index("--direction") + 1] == "en_to_vi"
    assert command[command.index("--max-steps") + 1] == "400"


def test_legacy_rescore_is_explicitly_not_vinai_evidence():
    path = ROOT / "data/reports/experiments/decoding/mt_val_greedy_rescored.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["model"] == "facebook/nllb-200-distilled-600M"
    assert report["evidence_status"] == "invalid_for_model_comparison"

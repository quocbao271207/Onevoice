from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import scripts.run_model_bakeoff as bakeoff
from scripts.run_model_bakeoff import (
    critical_safety_pass,
    deployment_expectations,
    interval_stronger,
    license_gate,
    report_score,
    run_stage,
    validate_candidate_matrix,
    validate_deployment_report,
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
    assert set(hashes) == {"mt", "asr", "accuracy_program"}
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

    report["categories"]["dose"]["safety_failure_rate"] = float("nan")
    assert critical_safety_pass(report, required) == (False, ["dose:invalid"])


def test_stronger_requires_non_overlapping_confidence_intervals():
    assert interval_stronger([47.0, 49.0], [43.0, 46.0], greater_is_better=True)
    assert not interval_stronger([45.5, 47.0], [44.0, 46.0], greater_is_better=True)
    assert interval_stronger([0.15, 0.17], [0.18, 0.21], greater_is_better=False)


@pytest.mark.parametrize(
    ("challenger", "baseline"),
    [
        (None, [1.0, 2.0]),
        ([2.0, 1.0], [3.0, 4.0]),
        ([float("nan"), 2.0], [3.0, 4.0]),
        ([1.0], [3.0, 4.0]),
    ],
)
def test_stronger_rejects_invalid_confidence_intervals(challenger, baseline):
    assert not interval_stronger(challenger, baseline, greater_is_better=True)


def test_selection_score_fails_closed_without_valid_metric_and_interval():
    report = {
        "directions": {
            "en_to_vi": {
                "chrf2": float("nan"),
                "chrf2_bootstrap_95ci": [2, 1],
            }
        },
        "categories": {"dose": {"samples": 1, "safety_failure_rate": 0.0}},
    }
    score = report_score(report, "mt", "en_to_vi", ["dose"])
    assert score["safety_pass"] is False
    assert score["clinical_safety_pass"] is True
    assert score["evidence_valid"] is False
    assert score["metric"] is None
    assert score["confidence_interval_95"] is None
    assert score["safety_failures"] == [
        "chrf2:missing_or_invalid",
        "chrf2_bootstrap_95ci:missing_or_invalid",
    ]


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
    assert score["clinical_safety_pass"] is False
    assert score["evidence_valid"] is True
    assert "code_switch_wer:above_policy" in score["safety_failures"]

    report["slices"]["code_switch"]["True"]["wer"] = float("nan")
    score = report_score(report, "asr", None, required)
    assert "code_switch_wer:missing_or_invalid" in score["safety_failures"]


def test_candidate_a_mt_is_benchmarked_per_direction(monkeypatch, tmp_path: Path):
    calls = []

    def fake_benchmark_unit(**kwargs):
        direction = kwargs["direction"]
        calls.append((kwargs["label"], direction))
        return {
            "directions": {
                direction: {
                    "chrf2": 50.0,
                    "chrf2_bootstrap_95ci": [49.0, 51.0],
                }
            },
            "categories": {"dose": {"samples": 1, "safety_failure_rate": 0.0}},
        }

    monkeypatch.setattr(bakeoff, "benchmark_unit", fake_benchmark_unit)
    data = {
        "candidates": {
            "mt": [
                {
                    "id": "candidate-a",
                    "role": "candidate_a",
                    "directions": ["en_to_vi", "vi_to_en"],
                }
            ]
        },
        "promotion_gate": {"critical_slices": ["dose"], "policy_slices": []},
        "successive_halving": {"semifinal": {"keep": 2}},
    }
    result = bakeoff.run_task_bakeoff(
        task="mt",
        config=data,
        research_approvals=set(),
        frozen={"adapters": {"mt": {"root": str(tmp_path)}}},
        python=sys.executable,
        state_dir=tmp_path,
        state_path=tmp_path / "state.json",
        state={"stages": {}},
    )
    assert calls == [
        ("candidate_a_frozen_en_to_vi", "en_to_vi"),
        ("candidate_a_frozen_vi_to_en", "vi_to_en"),
    ]
    assert result["winners"]["en_to_vi"]["decision"] == "candidate_a_retained"
    assert result["winners"]["vi_to_en"]["decision"] == "candidate_a_retained"


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


def test_deployment_expectations_bind_exact_adapter_tree(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    winners = [
        (
            "mt",
            "en_to_vi",
            {"candidate_id": "mt-winner", "adapter": str(adapter)},
        ),
        (
            "asr",
            None,
            {"candidate_id": "asr-winner", "adapter": str(adapter)},
        ),
    ]
    expected = deployment_expectations(winners)
    assert expected[0]["adapter_manifest_sha256"] == expected[1]["adapter_manifest_sha256"]
    assert len(expected[0]["adapter_manifest_sha256"]) == 64


def test_deployment_gate_requires_physical_qcs6490_and_valid_metrics(tmp_path: Path):
    checksum = "a" * 64
    project_root = tmp_path / "project"
    artifact = project_root / "models" / "mt-winner.tar"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"compiled-qnn-artifact")
    latency_samples = [200.0 + index for index in range(100)]
    expected = [
        {
            "task": "mt",
            "direction": "en_to_vi",
            "candidate_id": "mt-winner",
            "adapter_manifest_sha256": checksum,
        }
    ]
    winner = {
        **expected[0],
        "artifact_path": "models/mt-winner.tar",
        "artifact_sha256": bakeoff.sha256(artifact),
        "measurement_runs": 100,
        "latency_samples_ms": latency_samples,
        "latency_p50_ms": bakeoff.percentile_linear(latency_samples, 0.50),
        "latency_p95_ms": bakeoff.percentile_linear(latency_samples, 0.95),
        "peak_ram_bytes": 100_000_000,
        "peak_vram_bytes": 0,
        "model_bytes": artifact.stat().st_size,
    }
    report = {
        "version": 1,
        "status": "pass",
        "target": "QCS6490",
        "measurement_source": "physical_board",
        "measured_at": "2026-10-06T12:00:00+07:00",
        "device": {
            "chipset": "QCS6490",
            "board": "Dragonwing RB3 Gen 2 Vision Kit",
            "os": "Qc_Linux 1.6",
        },
        "winners": [winner],
    }
    metrics = [
        "latency_p50_ms",
        "latency_p95_ms",
        "peak_ram_bytes",
        "peak_vram_bytes",
        "model_bytes",
    ]
    assert validate_deployment_report(report, expected, metrics, 30, project_root) == (True, [])

    report["measurement_source"] = "cloud_profile"
    winner["candidate_id"] = "wrong-winner"
    winner["adapter_manifest_sha256"] = "c" * 64
    winner["measurement_runs"] = 2
    winner["latency_p95_ms"] = 200.0
    winner["peak_ram_bytes"] = float("nan")
    winner["model_bytes"] = 1.5
    passed, failures = validate_deployment_report(report, expected, metrics, 30, project_root)
    assert passed is False
    assert "measurement_source:not_physical_board" in failures
    assert "winner:mt/en_to_vi:candidate_mismatch" in failures
    assert "winner:mt/en_to_vi:adapter_checksum_mismatch" in failures
    assert "winner:mt/en_to_vi:insufficient_measurement_runs" in failures
    assert "winner:mt/en_to_vi:peak_ram_bytes_invalid" in failures
    assert "winner:mt/en_to_vi:model_bytes_invalid" in failures
    assert "winner:mt/en_to_vi:latency_percentiles_reversed" in failures

    winner["model_bytes"] = artifact.stat().st_size
    artifact.write_bytes(b"tampered")
    passed, failures = validate_deployment_report(report, expected, metrics, 30, project_root)
    assert passed is False
    assert "winner:mt/en_to_vi:artifact_checksum_mismatch" in failures
    assert "winner:mt/en_to_vi:model_bytes_mismatch" in failures

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import scripts.run_model_bakeoff as bakeoff
from scripts.build_selection_dev import collect_leakage_values, filter_disjoint
from scripts.run_model_bakeoff import (
    critical_safety_pass,
    deployment_expectations,
    interval_stronger,
    license_gate,
    multi_metric_stronger,
    report_score,
    run_stage,
    selection_identity,
    selection_identity_sha256,
    validate_candidate_matrix,
    validate_deployment_draft,
    validate_deployment_report,
    validate_resources,
    validate_selection_artifacts,
    write_runtime_round_config,
)
from src.pipeline.selection_policy import selection_policy_record


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
    assert data["resources"]["gpu_memory_fraction"] == 0.35
    assert data["resources"]["gpu_memory_hard_fraction"] == 0.40
    assert data["resources"]["sample_seconds"] == 1.0
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


def test_bakeoff_preflight_recomputes_fingerprints_and_rejects_locked_overlap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def write(path: str, rows: list[dict]) -> tuple[str, str]:
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path, bakeoff.sha256(destination)

    mt_source = "Give aspirin 5 mg."
    mt_target = "Dùng aspirin 5 mg."
    mt_pair = bakeoff.fingerprint_text(mt_source + "\x1f" + mt_target)
    mt_path, mt_sha = write(
        "selection-mt.jsonl",
        [
            {
                "id": "mt-selection",
                "source_text": mt_source,
                "target_text": mt_target,
                "pair_fingerprint": mt_pair,
            }
        ],
    )
    asr_text = "không dùng thuốc"
    asr_path, asr_sha = write(
        "selection-asr.jsonl",
        [
            {
                "id": "asr-selection",
                "text": asr_text,
                "text_fingerprint": bakeoff.fingerprint_text(asr_text),
                "audio_sha256": "a" * 64,
                "speaker": "shared-speaker",
                "group": "selection-group",
            }
        ],
    )
    write(
        "train-mt.jsonl",
        [
            {
                "id": "mt-train",
                "source_text": "No fever.",
                "target_text": "Không sốt.",
            }
        ],
    )
    write(
        "train-asr.jsonl",
        [
            {
                "id": "asr-train",
                "text": "đau ngực",
                "audio_sha256": "c" * 64,
                "speaker": "shared-speaker",
                "group": "train-group",
            }
        ],
    )
    write(
        "locked-mt.jsonl",
        [
            {
                "id": "different-id",
                "source_text": mt_source,
                "target_text": mt_target,
            }
        ],
    )
    write(
        "locked-asr.jsonl",
        [{"id": "locked-asr", "text": "đau ngực", "audio_sha256": "b" * 64}],
    )
    accuracy_path = tmp_path / "accuracy.yaml"
    accuracy_path.write_text("version: 1\n", encoding="utf-8")
    data = {
        "data": {
            "train": {"mt": "train-mt.jsonl", "asr": "train-asr.jsonl"},
            "selection_dev": {
                "mt": {"path": mt_path, "sha256": mt_sha},
                "asr": {"path": asr_path, "sha256": asr_sha},
            },
            "forbidden_selection_inputs": ["locked-mt.jsonl", "locked-asr.jsonl"],
            "accuracy_program": {
                "path": "accuracy.yaml",
                "sha256": bakeoff.sha256(accuracy_path),
            },
        }
    }
    monkeypatch.setattr(bakeoff, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="Selection/train leakage for asr.speaker"):
        validate_selection_artifacts(data)

    write(
        "train-asr.jsonl",
        [
            {
                "id": "asr-train",
                "text": "đau ngực",
                "audio_sha256": "c" * 64,
                "speaker": "train-speaker",
                "group": "train-group",
            }
        ],
    )

    with pytest.raises(ValueError, match="pair_fingerprint"):
        validate_selection_artifacts(data)

    rows = [
        {
            "id": "mt-selection",
            "source_text": mt_source,
            "target_text": mt_target,
            "pair_fingerprint": "0" * 64,
        }
    ]
    selection = tmp_path / mt_path
    selection.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    data["data"]["selection_dev"]["mt"]["sha256"] = bakeoff.sha256(selection)

    with pytest.raises(ValueError, match="pair_fingerprint mismatch"):
        validate_selection_artifacts(data)


def test_selection_builder_filters_content_speaker_and_group_leakage():
    excluded_rows = [
        {
            "id": "train",
            "text": "Một transcript đã dùng.",
            "audio_sha256": "a" * 64,
            "speaker": "speaker-train",
            "group": "group-train",
        }
    ]
    candidates = [
        {
            "id": "same-text",
            "text": "Một transcript đã dùng!",
            "audio_sha256": "b" * 64,
            "speaker": "speaker-new",
            "group": "group-new",
        },
        {
            "id": "same-speaker",
            "text": "Transcript mới một.",
            "audio_sha256": "c" * 64,
            "speaker": "speaker-train",
            "group": "group-new-2",
        },
        {
            "id": "same-group",
            "text": "Transcript mới hai.",
            "audio_sha256": "d" * 64,
            "speaker": "speaker-new-2",
            "group": "group-train",
        },
        {
            "id": "fresh",
            "text": "Transcript hoàn toàn mới.",
            "audio_sha256": "e" * 64,
            "speaker": "speaker-fresh",
            "group": "group-fresh",
        },
    ]

    filtered = filter_disjoint(
        "asr", candidates, collect_leakage_values("asr", excluded_rows)
    )

    assert [row["id"] for row in filtered] == ["fresh"]


def test_bakeoff_metric_policy_rejects_duplicates_and_invalid_code_switch_limit():
    data = config()
    data["promotion_gate"]["mt_metrics"].append("chrf2")
    with pytest.raises(ValueError, match="mt_metrics"):
        validate_resources(data)

    data = config()
    data["promotion_gate"]["asr_code_switch_wer_max"] = float("nan")
    with pytest.raises(ValueError, match="code-switch WER limit"):
        validate_resources(data)


def test_bakeoff_rejects_hard_memory_limit_below_process_limit():
    data = config()
    data["resources"]["gpu_memory_hard_fraction"] = 0.34
    with pytest.raises(ValueError, match="memory thresholds"):
        validate_resources(data)


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
                "sacrebleu": 30.0,
                "sacrebleu_bootstrap_95ci": [29.0, 31.0],
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
        "cer": 0.10,
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
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    def fake_benchmark_unit(**kwargs):
        direction = kwargs["direction"]
        calls.append((kwargs["label"], direction))
        return {
            "directions": {
                direction: {
                    "sacrebleu": 30.0,
                    "sacrebleu_bootstrap_95ci": [29.0, 31.0],
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
        frozen={"adapters": {"mt": {"root": str(adapter)}}},
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


def test_mt_multi_metric_winner_rejects_significant_bleu_regression():
    required = ["dose"]

    def score(sacrebleu, bleu_ci, chrf2, chrf_ci):
        return report_score(
            {
                "directions": {
                    "en_to_vi": {
                        "sacrebleu": sacrebleu,
                        "sacrebleu_bootstrap_95ci": bleu_ci,
                        "chrf2": chrf2,
                        "chrf2_bootstrap_95ci": chrf_ci,
                    }
                },
                "categories": {"dose": {"samples": 1, "safety_failure_rate": 0.0}},
            },
            "mt",
            "en_to_vi",
            required,
        )

    baseline = score(31.0, [30.0, 32.0], 50.0, [49.0, 51.0])
    bleu_regression = score(26.0, [25.0, 27.0], 54.0, [53.0, 55.0])
    pareto_improvement = score(33.0, [32.1, 34.0], 54.0, [53.0, 55.0])

    assert multi_metric_stronger(bleu_regression, baseline) is False
    assert multi_metric_stronger(pareto_improvement, baseline) is True


def test_selection_identity_ignores_later_blind_results_but_binds_winners():
    policy = selection_policy_record(config())
    comparison = {
        "status": "selection_complete",
        "scope": "research",
        "candidate_a_freeze": "freeze.json",
        "selection_sha256": {"mt": "a", "asr": "b"},
        "selection_policy": policy,
        "results": {
            "mt": {
                "winners": {
                    "en_to_vi": {
                        "candidate_id": "mt-winner",
                        "adapter": "/models/mt",
                        "adapter_manifest_sha256": "c" * 64,
                        "direction": "en_to_vi",
                    }
                }
            },
            "asr": {"winners": {}},
        },
    }
    finalized = json.loads(json.dumps(comparison))
    finalized["status"] = "blind_complete"
    finalized["blind_test_v2"] = [{"promotion_allowed": True}]

    assert selection_identity(finalized) == selection_identity(comparison)
    assert selection_identity_sha256(finalized) == selection_identity_sha256(comparison)

    finalized["results"]["mt"]["winners"]["en_to_vi"]["candidate_id"] = "other"
    assert selection_identity(finalized) != selection_identity(comparison)
    assert selection_identity_sha256(finalized) != selection_identity_sha256(comparison)

    finalized = json.loads(json.dumps(comparison))
    finalized["selection_policy"]["winner_rule"] = "legacy_single_metric"
    assert selection_identity(finalized) != selection_identity(comparison)


def test_asr_selection_requires_cer_evidence():
    required = ["code_switch"]
    report = {
        "wer": 0.15,
        "wer_bootstrap_95ci": [0.14, 0.16],
        "categories": {"code_switch": {"samples": 1, "safety_failure_rate": 0.0}},
        "slices": {"code_switch": {"True": {"wer": 0.18}}},
    }

    score = report_score(report, "asr", None, required)

    assert score["evidence_valid"] is False
    assert score["safety_pass"] is False
    assert "cer:missing_or_invalid" in score["safety_failures"]
    assert score["metrics"]["code_switch_wer"]["value"] == 0.18


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


def test_resume_scoring_flag_preserves_completed_benchmark_stage(tmp_path: Path):
    state_path = tmp_path / "state.json"
    output = tmp_path / "output.json"
    output.write_text("ok", encoding="utf-8")
    log = tmp_path / "stage.log"
    old_command = [
        sys.executable,
        str(ROOT / "scripts/run_baseline_benchmarks.py"),
        "--task",
        "mt",
    ]
    state = {
        "stages": {
            "stage": {
                "status": "complete",
                "command": old_command,
                "command_sha256": bakeoff.command_digest(old_command),
            }
        }
    }

    run_stage(
        "stage",
        old_command + ["--resume-scoring"],
        state_path,
        state,
        log,
        [output],
    )

    assert not log.exists()


def test_bakeoff_benchmarks_enable_verified_scoring_resume(tmp_path: Path):
    command = bakeoff.benchmark_command(
        sys.executable,
        "mt",
        candidate("mt_m2m100_418m"),
        tmp_path / "selection.jsonl",
        tmp_path / "reports",
        "pilot",
        None,
        "en_to_vi",
    )

    assert command.count("--resume-scoring") == 1


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

    report["device"]["board"] = "Arduino Uno"
    passed, failures = validate_deployment_report(report, expected, metrics, 30, project_root)
    assert passed is False
    assert "device.board:not_qcs6490_board" in failures

    winner["model_bytes"] = artifact.stat().st_size
    artifact.write_bytes(b"tampered")
    passed, failures = validate_deployment_report(report, expected, metrics, 30, project_root)
    assert passed is False
    assert "winner:mt/en_to_vi:artifact_checksum_mismatch" in failures
    assert "winner:mt/en_to_vi:model_bytes_mismatch" in failures


def test_deployment_draft_is_bound_to_current_selection_and_winners(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    comparison = tmp_path / "comparison.json"
    comparison.write_text('{"status":"blind_complete"}', encoding="utf-8")
    expected = deployment_expectations(
        [("mt", "en_to_vi", {"candidate_id": "winner", "adapter": str(adapter)})]
    )
    draft = {
        "version": 1,
        "status": "pending_physical_measurement",
        "target": "QCS6490",
        "measurement_source": "physical_board",
        "selection_comparison": {
            "path": str(comparison.resolve()),
            "sha256": bakeoff.sha256(comparison),
        },
        "winners": [{**expected[0], "latency_samples_ms": []}],
    }

    assert validate_deployment_draft(draft, expected, comparison) == (True, [])

    draft["winners"][0]["candidate_id"] = "other"
    comparison.write_text('{"status":"changed"}', encoding="utf-8")
    passed, failures = validate_deployment_draft(draft, expected, comparison)
    assert passed is False
    assert "selection_comparison:checksum_mismatch" in failures
    assert "winner:mt/en_to_vi:candidate_mismatch" in failures

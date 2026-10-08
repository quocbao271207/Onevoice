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
    audit_completed_stage_resume,
    assert_interrupted_stage_is_not_live,
    bind_or_validate_invocation,
    candidate_output_dir,
    completed_adapter,
    critical_safety_pass,
    deployment_expectations,
    hyperparameter_profiles,
    interval_stronger,
    invocation_binding,
    license_gate,
    multi_metric_stronger,
    report_score,
    run_stage,
    select_finalists,
    select_semifinalists,
    selection_identity,
    selection_identity_sha256,
    strongest_eligible_challenger,
    train_unit,
    tree_manifest,
    validate_candidate_matrix,
    validate_candidate_a_locked_evaluation,
    validate_candidate_a_freeze,
    validate_deployment_draft,
    validate_deployment_report,
    validate_resources,
    validate_selection_artifacts,
    verified_resume_checkpoint,
    write_runtime_round_config,
)
from scripts.watch_checkpoints import archive_ready_checkpoints
from scripts.run_gpu_rounds import archive_round
from scripts.candidate_evidence import archive_evidence
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
    assert data["prerequisite"]["candidate_a_locked_evaluation"]["asr"][
        "require_content_manifest"
    ] is True
    for task in ("mt", "asr"):
        profiles = hyperparameter_profiles(data, task)
        assert [profile["lora_rank"] for profile in profiles] == [8, 16, 32]
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


def test_gpu_requirements_include_m2m100_tokenizer_runtime():
    requirements = {
        line.strip()
        for line in (ROOT / "requirements-gpu.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "sentencepiece==0.2.1" in requirements


def test_bakeoff_rejects_candidate_a_infrastructure_error_before_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(bakeoff, "ROOT", tmp_path)
    adapter = tmp_path / "gpu-runs" / "adapter"
    output = tmp_path / "gpu-runs" / "asr-candidate"
    adapter.mkdir(parents=True)
    output.mkdir(parents=True)
    gate = {
        "status": "error",
        "error": "dtype mismatch",
        "promotion_allowed": False,
        "adapter": str(adapter),
        "base_model": "vinai/PhoWhisper-small",
        "base_model_revision": "a86b604c346caf7148c37512eafe783a16420adb",
        "checks": [],
        "resource_runs": {},
    }
    (output / "candidate_gate.json").write_text(json.dumps(gate), encoding="utf-8")
    current = {
        "asr_adapter": str(adapter),
        "stages": {"asr_candidate": {"command": ["python", "--output-dir", str(output)]}},
    }

    assert candidate_output_dir(current, "asr") == output.resolve()
    with pytest.raises(ValueError, match="locked evaluation is not terminal"):
        validate_candidate_a_locked_evaluation(current, config(), "asr")


def test_bakeoff_binds_terminal_candidate_a_gate_and_canonical_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(bakeoff, "ROOT", tmp_path)
    adapter = tmp_path / "gpu-runs" / "adapter"
    output = tmp_path / "gpu-runs" / "asr-candidate-rerun"
    adapter.mkdir(parents=True)
    output.mkdir(parents=True)
    gate = {
        "status": "fail",
        "error": None,
        "promotion_allowed": False,
        "adapter": str(adapter),
        "base_model": "vinai/PhoWhisper-small",
        "base_model_revision": "a86b604c346caf7148c37512eafe783a16420adb",
        "locked_hashes": {"test": "a" * 64, "safety": "b" * 64},
        "checks": [
            {"name": "asr_vi_wer", "pass": False},
            {"name": "asr_code_switch_wer", "pass": True},
            *[
                {"name": f"clinical_{name}_failure_rate", "pass": True}
                for name in config()["promotion_gate"]["critical_slices"]
            ],
            *[
                {"name": f"clinical_{name}_failure_rate", "pass": True}
                for name in config()["promotion_gate"]["policy_slices"]
            ],
        ],
        "resource_limits": {
            "gpu_memory_fraction": 0.35,
            "utilization_percent": 70.0,
            "hard_utilization_percent": 74.0,
            "resume_percent": 55.0,
        },
        "resource_runs": {
            "aggregate": {"return_code": 0},
            "clinical": {"return_code": 0},
        },
    }
    gate_path = output / "candidate_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    archive, digest = archive_evidence(output)
    current = {
        "asr_adapter": str(adapter),
        "stages": {"asr_candidate": {"command": ["python", "--output-dir", str(output)]}},
    }

    record = validate_candidate_a_locked_evaluation(current, config(), "asr")

    assert record["status"] == "fail"
    assert record["gate"]["sha256"] == bakeoff.sha256(gate_path)
    assert record["archive"]["sha256"] == digest
    assert record["archive"]["content_manifest_verified"] is True


def test_candidate_a_resume_output_reuses_locked_evidence_and_honors_override():
    state = {
        "candidate_a_locked_evaluations": {
            "asr": {
                "task": "asr",
                "output_dir": "/gpu-runs/asr-candidate-rerun",
            }
        }
    }

    assert bakeoff.candidate_a_resume_output(state, "asr") == Path(
        "/gpu-runs/asr-candidate-rerun"
    )
    assert bakeoff.candidate_a_resume_output(state, "mt") is None
    explicit = Path("/gpu-runs/explicit-asr")
    assert bakeoff.candidate_a_resume_output(state, "asr", explicit) == explicit


def test_candidate_a_resume_output_rejects_unbound_record():
    state = {
        "candidate_a_locked_evaluations": {
            "asr": {
                "task": "mt",
                "output_dir": "/gpu-runs/wrong-task",
            }
        }
    }

    assert bakeoff.candidate_a_resume_output(state, "asr") is None


def test_existing_candidate_a_freeze_is_rehashed_before_resume(tmp_path: Path):
    mt_adapter = tmp_path / "mt"
    asr_adapter = tmp_path / "asr"
    for adapter, payload in ((mt_adapter, b"mt"), (asr_adapter, b"asr")):
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(payload)
    current = {
        "execution_status": "complete",
        "mt_adapter": str(mt_adapter),
        "asr_adapter": str(asr_adapter),
    }
    freeze = tmp_path / "candidate_a_freeze.json"

    frozen = bakeoff.freeze_candidate_a(current, freeze)
    assert validate_candidate_a_freeze(current, freeze) == frozen

    (asr_adapter / "adapter_model.safetensors").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="adapter changed after freeze"):
        validate_candidate_a_freeze(current, freeze)


def test_bakeoff_invocation_binding_blocks_cross_program_resume(tmp_path: Path):
    program = tmp_path / "program_state.json"
    config_path = tmp_path / "bakeoff.yaml"
    program.write_text("{}", encoding="utf-8")
    config_path.write_text("version: 1\n", encoding="utf-8")
    binding = invocation_binding(program, config_path, "research", set())
    state = {"stages": {"interrupted": {"status": "running"}}}

    bind_or_validate_invocation(state, binding)
    assert state["invocation_binding"] == binding
    bind_or_validate_invocation(state, binding)

    changed = invocation_binding(program, config_path, "production", set())
    with pytest.raises(ValueError, match="invocation changed"):
        bind_or_validate_invocation(state, changed)


def test_completed_legacy_state_without_invocation_binding_is_rejected():
    state = {"stages": {"old": {"status": "complete"}}}
    binding = {
        "current_program_state": "/program.json",
        "config": {"path": "/config.yaml", "sha256": "a" * 64},
        "scope": "research",
        "research_approvals": [],
    }

    with pytest.raises(ValueError, match="completed stages"):
        bind_or_validate_invocation(state, binding)


def test_adapter_tree_manifest_uses_portable_paths(tmp_path: Path):
    adapter = tmp_path / "adapter"
    nested = adapter / "nested"
    nested.mkdir(parents=True)
    weights = nested / "weights.bin"
    weights.write_bytes(b"weights")

    manifest = tree_manifest(adapter)

    assert manifest["files"] == [
        {
            "path": "nested/weights.bin",
            "bytes": len(b"weights"),
            "sha256": bakeoff.sha256(weights),
        }
    ]


def test_adapter_tree_manifest_rejects_symlink_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    link = adapter / "linked.bin"
    link.write_bytes(b"target")
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path == link or original_is_symlink(path),
    )

    with pytest.raises(ValueError, match="cannot contain symlinks"):
        tree_manifest(adapter)


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


def test_winner_search_skips_higher_point_score_with_significant_regression():
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
    point_ranked_first = {
        "unit": "high-bleu-regressor",
        "score": score(36.0, [35.0, 37.0], 45.0, [44.0, 46.0]),
    }
    valid_pareto_winner = {
        "unit": "valid-pareto-winner",
        "score": score(33.0, [32.1, 34.0], 54.0, [53.0, 55.0]),
    }
    ranked = bakeoff.rank_scores([valid_pareto_winner, point_ranked_first])

    assert ranked[0]["unit"] == "high-bleu-regressor"
    assert strongest_eligible_challenger(ranked, baseline)["unit"] == "valid-pareto-winner"


def test_mt_successive_halving_keeps_each_direction_independent():
    def entry(unit: str, direction: str, value: float) -> dict:
        return {
            "unit": unit,
            "candidate": {"direction": direction},
            "score": {
                "safety_pass": True,
                "evidence_valid": True,
                "ranking_key": value,
                "metrics": {
                    "sacrebleu": {"ci": [value, value + 0.1]},
                    "chrf2": {"ci": [value, value + 0.1]},
                },
            },
        }

    pilot = [
        entry("en-best", "en_to_vi", 100.0),
        entry("en-second", "en_to_vi", 90.0),
        entry("vi-best", "vi_to_en", 10.0),
        entry("vi-second", "vi_to_en", 9.0),
    ]

    selected = select_semifinalists(pilot, "mt", 1)
    assert {item["unit"] for item in selected} == {"en-best", "vi-best"}
    assert {item["unit"] for item in select_finalists(selected, "mt")} == {
        "en-best",
        "vi-best",
    }


def test_asr_successive_halving_uses_one_shared_ranking():
    entries = [
        {
            "unit": name,
            "candidate": {"direction": None},
            "score": {"safety_pass": True, "ranking_key": value},
        }
        for name, value in (("best", 2.0), ("second", 1.0))
    ]
    assert [item["unit"] for item in select_semifinalists(entries, "asr", 1)] == [
        "best"
    ]


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

    finalized = json.loads(json.dumps(comparison))
    finalized["results"]["mt"]["winners"]["en_to_vi"]["profile"] = {
        "id": "r32_lr2e5"
    }
    assert selection_identity(finalized) != selection_identity(comparison)

    finalized = json.loads(json.dumps(comparison))
    finalized["candidate_a_locked_evaluations"] = {
        "asr": {"gate": {"sha256": "d" * 64}}
    }
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


def test_all_stage_launches_reject_a_stale_loaded_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "must-not-exist"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(output)!r}).write_text('unsafe')",
    ]
    monkeypatch.setattr(bakeoff, "LOADED_RUNNER_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="runner no longer matches"):
        run_stage(
            "stale",
            command,
            tmp_path / "state.json",
            {"stages": {}},
            tmp_path / "stage.log",
            [output],
        )
    assert not output.exists()


def test_running_stage_pid_blocks_duplicate_launch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bakeoff, "pid_is_live", lambda pid: pid == 4242)
    with pytest.raises(RuntimeError, match="PID 4242 is still live"):
        assert_interrupted_stage_is_not_live(
            {"status": "running", "pid": 4242, "command": ["python", "stage.py"]}
        )


def test_failed_stage_preserves_attempt_history(tmp_path: Path):
    state_path = tmp_path / "state.json"
    state = {"stages": {}}
    command = [sys.executable, "-c", "raise SystemExit(1)"]

    with pytest.raises(bakeoff.StageExecutionError):
        run_stage("stage", command, state_path, state, tmp_path / "attempt-1.log", [])
    with pytest.raises(bakeoff.StageExecutionError):
        run_stage("stage", command, state_path, state, tmp_path / "attempt-2.log", [])

    record = state["stages"]["stage"]
    assert record["attempt"] == 2
    assert record["attempt_history"] == [
        {
            "attempt": 1,
            "status": "error",
            "return_code": 1,
            "started_at": record["attempt_history"][0]["started_at"],
            "completed_at": record["attempt_history"][0]["completed_at"],
            "command_sha256": bakeoff.command_digest(command),
            "log": str(tmp_path / "attempt-1.log"),
        }
    ]


def test_gpu_stage_waits_for_capacity_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    state_path = tmp_path / "state.json"
    output = tmp_path / "output.json"
    log = tmp_path / "stage.log"
    state = {"stages": {}}
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(output)!r}).write_text('ok')",
    ]
    observed: dict = {}

    def approve_capacity(limits: dict, evidence_path: Path) -> dict:
        waiting_state = json.loads(state_path.read_text(encoding="utf-8"))
        observed.update(waiting_state["stages"]["stage"])
        assert not output.exists()
        return {
            "samples": 1,
            "evidence_log": str(evidence_path),
            "approval": {"allowed": True},
        }

    def monitor(process: subprocess.Popen, *_args) -> dict:
        return {"return_code": process.wait()}

    monkeypatch.setattr(bakeoff, "wait_for_gpu_spawn_capacity", approve_capacity)
    monkeypatch.setattr("scripts.run_gpu_rounds.monitor_process", monitor)

    run_stage(
        "stage",
        command,
        state_path,
        state,
        log,
        [output],
        resource_limits=config()["resources"],
    )

    assert observed["pid"] == bakeoff.os.getpid()
    assert observed["waiting_for_gpu_capacity"] is True
    assert state["stages"]["stage"]["waiting_for_gpu_capacity"] is False
    assert state["stages"]["stage"]["spawn_capacity"]["approval"]["allowed"] is True


def test_legacy_running_stage_command_blocks_duplicate_launch(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        bakeoff, "matching_linux_command_pids", lambda command: [111, 222]
    )
    with pytest.raises(RuntimeError, match="111, 222"):
        assert_interrupted_stage_is_not_live(
            {"status": "running", "command": ["python", "stage.py"]}
        )


def test_resume_scoring_upgrade_revalidates_completed_benchmark_stage(tmp_path: Path):
    state_path = tmp_path / "state.json"
    output = tmp_path / "output.json"
    output.write_text("legacy", encoding="utf-8")
    log = tmp_path / "stage.log"
    scorer = tmp_path / "run_baseline_benchmarks.py"
    scorer.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "assert '--resume-scoring' in sys.argv\n"
        f"Path({str(output)!r}).write_text('verified', encoding='utf-8')\n",
        encoding="utf-8",
    )
    old_command = [
        sys.executable,
        str(scorer),
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

    assert output.read_text(encoding="utf-8") == "verified"
    assert log.exists()
    assert state["stages"]["stage"]["output_evidence"] == [
        {
            "path": str(output.resolve()),
            "kind": "file",
            "bytes": len("verified"),
            "sha256": bakeoff.sha256(output),
        }
    ]


def test_completed_benchmark_accepts_runner_generation_only_change(tmp_path: Path):
    state_path = tmp_path / "state.json"
    output = tmp_path / "output.json"
    output.write_text("verified", encoding="utf-8")
    log = tmp_path / "stage.log"
    scorer = tmp_path / "run_baseline_benchmarks.py"
    old_command = [
        sys.executable,
        str(scorer),
        "--task",
        "mt",
        "--bakeoff-runner-sha256",
        "a" * 64,
    ]
    current_command = [*old_command[:-1], "b" * 64]
    state = {
        "stages": {
            "stage": {
                "status": "complete",
                "command": old_command,
                "command_sha256": bakeoff.command_digest(old_command),
                "output_evidence": bakeoff.output_evidence([output]),
            }
        }
    }

    run_stage("stage", current_command, state_path, state, log, [output])

    assert not log.exists()
    assert state["stages"]["stage"]["command"] == old_command


def test_completed_training_accepts_same_interpreter_alias(tmp_path: Path):
    state_path = tmp_path / "state.json"
    output = tmp_path / "training-output"
    output.mkdir()
    log = tmp_path / "stage.log"
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    alias_parent = tmp_path / "alias"
    alias_parent.mkdir()
    old_command = [
        str(python),
        "/repo/scripts/run_gpu_rounds.py",
        "--task",
        "mt",
    ]
    current_command = [
        str(alias_parent / ".." / python.name),
        *old_command[1:],
    ]
    state = {
        "stages": {
            "stage": {
                "status": "complete",
                "command": old_command,
                "command_sha256": bakeoff.command_digest(old_command),
                "output_evidence": bakeoff.output_evidence([output]),
            }
        }
    }

    run_stage("stage", current_command, state_path, state, log, [output])

    assert not log.exists()
    assert state["stages"]["stage"]["command"] == old_command


def test_interpreter_alias_resolution_uses_stage_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    python = repo / "python"
    python.write_text("", encoding="utf-8")
    alias_parent = repo / "alias"
    alias_parent.mkdir()
    unrelated_cwd = tmp_path / "waiter-cwd"
    unrelated_cwd.mkdir()
    monkeypatch.setattr(bakeoff, "ROOT", repo)
    monkeypatch.chdir(unrelated_cwd)

    old_command = ["./python", "/repo/scripts/run_gpu_rounds.py", "--task", "mt"]
    current_command = [
        "./alias/../python",
        *old_command[1:],
    ]

    assert bakeoff.interpreter_alias_only_command_change(
        old_command,
        current_command,
    )


def test_resume_audit_verifies_alias_and_completed_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    python = repo / "python"
    python.write_text("", encoding="utf-8")
    alias_parent = repo / "alias"
    alias_parent.mkdir()
    output = tmp_path / "report.json"
    output.write_text('{"verified": true}', encoding="utf-8")
    command = ["./python", "/repo/scripts/run_gpu_rounds.py", "--task", "mt"]
    state = {
        "stages": {
            "completed_training": {
                "status": "complete",
                "command": command,
                "command_sha256": bakeoff.command_digest(command),
                "output_evidence": bakeoff.output_evidence([output]),
            },
            "interrupted_training": {"status": "error"},
        }
    }
    monkeypatch.setattr(bakeoff, "ROOT", repo)

    audit = audit_completed_stage_resume(state, "./alias/../python")

    assert audit == {
        "status": "pass",
        "completed_stage_count": 1,
        "stages": [
            {
                "stage": "completed_training",
                "command_compatibility": "interpreter_alias",
                "status": "pass",
                "failures": [],
            }
        ],
        "failures": [],
    }


def test_resume_audit_fails_closed_on_changed_output_or_interpreter(
    tmp_path: Path,
):
    old_python = tmp_path / "old-python"
    old_python.write_text("", encoding="utf-8")
    new_python = tmp_path / "new-python"
    new_python.write_text("", encoding="utf-8")
    output = tmp_path / "report.json"
    output.write_text("original", encoding="utf-8")
    command = [str(old_python), "/repo/scripts/run_gpu_rounds.py"]
    state = {
        "stages": {
            "completed_training": {
                "status": "complete",
                "command": command,
                "command_sha256": bakeoff.command_digest(command),
                "output_evidence": bakeoff.output_evidence([output]),
            }
        }
    }
    output.write_text("changed", encoding="utf-8")

    audit = audit_completed_stage_resume(state, str(new_python))

    assert audit["status"] == "fail"
    assert audit["completed_stage_count"] == 1
    assert audit["stages"][0]["status"] == "fail"
    assert audit["stages"][0]["failures"] == [
        "current execution identity is not resume-compatible",
        "recorded output evidence changed",
    ]


def test_resume_audit_cli_is_read_only(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    output = tmp_path / "report.json"
    output.write_text('{"verified": true}', encoding="utf-8")
    command = [sys.executable, "/repo/scripts/run_gpu_rounds.py"]
    state_path = state_dir / "bakeoff_state.json"
    state_path.write_text(
        json.dumps(
            {
                "stages": {
                    "completed_training": {
                        "status": "complete",
                        "command": command,
                        "command_sha256": bakeoff.command_digest(command),
                        "output_evidence": bakeoff.output_evidence([output]),
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    before = state_path.read_bytes()

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_model_bakeoff.py"),
            "--resume-audit",
            "--state-dir",
            str(state_dir),
            "--python",
            sys.executable,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["resume_audit"]["status"] == "pass"
    assert state_path.read_bytes() == before
    assert list(state_dir.iterdir()) == [state_path]


def test_resume_audit_rejects_unbound_legacy_benchmark(tmp_path: Path):
    output = tmp_path / "selection.json"
    output.write_text('{"verified": true}', encoding="utf-8")
    command = [
        sys.executable,
        "/repo/scripts/run_baseline_benchmarks.py",
        "--task",
        "mt",
    ]
    state = {
        "stages": {
            "legacy_selection": {
                "status": "complete",
                "command": command,
                "command_sha256": bakeoff.command_digest(command),
                "output_evidence": bakeoff.output_evidence([output]),
            }
        }
    }

    audit = audit_completed_stage_resume(state, sys.executable)

    assert audit["status"] == "fail"
    assert audit["failures"] == [
        "legacy_selection: benchmark runner generation binding is missing or ambiguous"
    ]


def test_completed_training_alias_requires_recorded_output_evidence(tmp_path: Path):
    state_path = tmp_path / "state.json"
    output = tmp_path / "training-output"
    output.mkdir()
    log = tmp_path / "stage.log"
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    alias_parent = tmp_path / "alias"
    alias_parent.mkdir()
    old_command = [str(python), "/repo/scripts/run_gpu_rounds.py", "--task", "mt"]
    current_command = [str(alias_parent / ".." / python.name), *old_command[1:]]
    state = {
        "stages": {
            "stage": {
                "status": "complete",
                "command": old_command,
                "command_sha256": bakeoff.command_digest(old_command),
            }
        }
    }

    with pytest.raises(ValueError, match="output evidence missing"):
        run_stage("stage", current_command, state_path, state, log, [output])


def test_interpreter_alias_compatibility_rejects_other_command_changes(
    tmp_path: Path,
):
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    alias_parent = tmp_path / "alias"
    alias_parent.mkdir()
    old_command = [str(python), "/repo/scripts/run_gpu_rounds.py", "--task", "mt"]

    assert not bakeoff.interpreter_alias_only_command_change(
        old_command,
        [
            str(alias_parent / ".." / python.name),
            *old_command[1:],
            "--samples",
            "1",
        ],
    )


def test_runner_generation_compatibility_accepts_same_interpreter_alias(
    tmp_path: Path,
):
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    alias_parent = tmp_path / "alias"
    alias_parent.mkdir()
    old_command = [
        str(python),
        "/repo/scripts/run_baseline_benchmarks.py",
        "--task",
        "mt",
        "--bakeoff-runner-sha256",
        "a" * 64,
    ]
    current_command = [
        str(alias_parent / ".." / python.name),
        *old_command[1:-1],
        "b" * 64,
    ]

    assert bakeoff.runner_generation_only_command_change(
        old_command,
        current_command,
    )


def test_runner_generation_compatibility_rejects_different_interpreter(
    tmp_path: Path,
):
    old_python = tmp_path / "old-python"
    current_python = tmp_path / "current-python"
    old_python.write_text("", encoding="utf-8")
    current_python.write_text("", encoding="utf-8")
    old_command = [
        str(old_python),
        "/repo/scripts/run_baseline_benchmarks.py",
        "--task",
        "mt",
        "--bakeoff-runner-sha256",
        "a" * 64,
    ]
    current_command = [str(current_python), *old_command[1:-1], "b" * 64]

    assert not bakeoff.runner_generation_only_command_change(
        old_command,
        current_command,
    )


def test_runner_generation_compatibility_rejects_other_command_changes():
    old_command = [
        "python",
        "/repo/scripts/run_baseline_benchmarks.py",
        "--task",
        "mt",
        "--bakeoff-runner-sha256",
        "a" * 64,
    ]

    assert not bakeoff.runner_generation_only_command_change(
        old_command,
        [*old_command[:-1], "b" * 64, "--samples", "1"],
    )


def test_completed_stage_rejects_modified_file_output(tmp_path: Path):
    state_path = tmp_path / "state.json"
    output = tmp_path / "output.json"
    log = tmp_path / "stage.log"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(output)!r}).write_text('trusted')",
    ]
    state: dict = {}

    run_stage("stage", command, state_path, state, log, [output])
    output.write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="output evidence changed"):
        run_stage("stage", command, state_path, state, log, [output])


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
    guard_index = command.index("--bakeoff-runner-sha256")
    assert command[guard_index + 1] == bakeoff.LOADED_RUNNER_SHA256
    assert bakeoff.LOADED_RUNNER_SHA256 == bakeoff.sha256(
        Path(bakeoff.__file__).resolve()
    )


def test_m2m100_runtime_round_builds_a_dry_run_command(tmp_path: Path):
    data = config()
    item = candidate("mt_m2m100_418m")
    runtime = tmp_path / "runtime.yaml"
    profile = hyperparameter_profiles(data, "mt")[-1]
    write_runtime_round_config(
        runtime, data, "mt", item, "en_to_vi", "pilot", 400, profile
    )
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
    assert command[command.index("--learning-rate") + 1] == "2e-05"
    assert command[command.index("--lora-rank") + 1] == "32"


def test_completed_bakeoff_adapter_requires_verified_round_bundle(tmp_path: Path):
    output_root = tmp_path / "training"
    run = output_root / "mt-20261006-000000"
    round_dir = run / "01-full"
    adapter = round_dir / "model"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"verified-weights")
    archive = archive_round(round_dir)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "task": "mt",
                "status": "complete",
                "selected_round": "full",
                "rounds": [
                    {
                        "name": "full",
                        "status": "complete",
                        "metric": 1.0,
                        "archive": archive,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert completed_adapter(output_root, "mt") == adapter

    weights.write_bytes(b"mutated-after-archive")
    with pytest.raises(ValueError, match="live file does not match verified archive"):
        completed_adapter(output_root, "mt")


def test_interrupted_bakeoff_training_resumes_only_verified_live_checkpoint(
    tmp_path: Path,
):
    output_root = tmp_path / "training"
    run = output_root / "mt-20261006-000000"
    checkpoint = run / "01-pilot" / "model" / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(name.encode())
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 100,
                "max_steps": 400,
                "best_metric": 1.25,
                "best_model_checkpoint": str(checkpoint),
                "log_history": [{"step": 100, "eval_loss": 1.25}],
            }
        ),
        encoding="utf-8",
    )
    archive_dir = run / "checkpoint-archives" / "01-pilot"
    archive_ready_checkpoints(checkpoint.parent, archive_dir)

    assert verified_resume_checkpoint(output_root, "mt") == checkpoint.resolve()

    (checkpoint / "optimizer.pt").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="does not match verified archive"):
        verified_resume_checkpoint(output_root, "mt")


def test_interrupted_train_unit_passes_verified_checkpoint_to_round_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = tmp_path / "checkpoint-100"
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(bakeoff, "write_runtime_round_config", lambda *args: None)
    monkeypatch.setattr(
        bakeoff, "verified_resume_checkpoint", lambda output_root, task: checkpoint
    )
    monkeypatch.setattr(
        bakeoff,
        "run_stage",
        lambda name, command, *args: captured.update(command=command),
    )
    monkeypatch.setattr(
        bakeoff, "completed_adapter", lambda output_root, task: tmp_path / "adapter"
    )
    stage = "mt_mt_m2m100_418m__vi_to_en_pilot_r32_lr2e5_train"
    state = {"stages": {stage: {"status": "error"}}}
    unit = {
        "unit_id": "mt_m2m100_418m__vi_to_en",
        "model": "facebook/m2m100_418M",
        "revision": "revision",
        "model_family": "m2m100",
        "direction": "vi_to_en",
    }

    train_unit(
        unit=unit,
        task="mt",
        round_name="pilot",
        steps=400,
        config=config(),
        python=sys.executable,
        state_dir=tmp_path,
        state_path=tmp_path / "state.json",
        state=state,
        profile=hyperparameter_profiles(config(), "mt")[-1],
    )

    command = captured["command"]
    assert command[command.index("--round-name") + 1] == "pilot"
    assert command[command.index("--resume-from-checkpoint") + 1] == str(checkpoint)


def test_train_unit_retries_failed_stage_from_verified_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = tmp_path / "checkpoint-100"
    calls: list[tuple[list[str], Path]] = []
    monkeypatch.setattr(bakeoff, "write_runtime_round_config", lambda *args: None)
    monkeypatch.setattr(
        bakeoff, "verified_resume_checkpoint", lambda output_root, task: checkpoint
    )

    def run(name, command, state_path, state, log_path, expected_outputs):
        attempt = len(calls) + 1
        calls.append((command, log_path))
        state.setdefault("stages", {})[name] = {
            "status": "error" if attempt == 1 else "complete",
            "attempt": attempt,
        }
        if attempt == 1:
            raise bakeoff.StageExecutionError("interrupted")

    monkeypatch.setattr(bakeoff, "run_stage", run)
    monkeypatch.setattr(
        bakeoff, "completed_adapter", lambda output_root, task: tmp_path / "adapter"
    )
    unit = {
        "unit_id": "mt_m2m100_418m__vi_to_en",
        "model": "facebook/m2m100_418M",
        "revision": "revision",
        "model_family": "m2m100",
        "direction": "vi_to_en",
    }

    train_unit(
        unit=unit,
        task="mt",
        round_name="pilot",
        steps=400,
        config=config(),
        python=sys.executable,
        state_dir=tmp_path,
        state_path=tmp_path / "state.json",
        state={"stages": {}},
        profile=hyperparameter_profiles(config(), "mt")[-1],
    )

    assert len(calls) == 2
    assert "--resume-from-checkpoint" not in calls[0][0]
    assert calls[1][0][calls[1][0].index("--resume-from-checkpoint") + 1] == str(checkpoint)
    assert calls[1][1].name.endswith(".attempt-2.log")


def test_train_unit_does_not_retry_without_verified_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = 0
    monkeypatch.setattr(bakeoff, "write_runtime_round_config", lambda *args: None)
    monkeypatch.setattr(
        bakeoff, "verified_resume_checkpoint", lambda output_root, task: None
    )

    def run(name, command, state_path, state, log_path, expected_outputs):
        nonlocal calls
        calls += 1
        state.setdefault("stages", {})[name] = {"status": "error", "attempt": calls}
        raise bakeoff.StageExecutionError("interrupted")

    monkeypatch.setattr(bakeoff, "run_stage", run)
    unit = {
        "unit_id": "mt_m2m100_418m__vi_to_en",
        "model": "facebook/m2m100_418M",
        "revision": "revision",
        "model_family": "m2m100",
        "direction": "vi_to_en",
    }

    with pytest.raises(bakeoff.StageExecutionError):
        train_unit(
            unit=unit,
            task="mt",
            round_name="pilot",
            steps=400,
            config=config(),
            python=sys.executable,
            state_dir=tmp_path,
            state_path=tmp_path / "state.json",
            state={"stages": {}},
            profile=hyperparameter_profiles(config(), "mt")[-1],
        )

    assert calls == 1


def test_train_unit_bounds_verified_checkpoint_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = 0
    monkeypatch.setattr(bakeoff, "write_runtime_round_config", lambda *args: None)
    monkeypatch.setattr(
        bakeoff,
        "verified_resume_checkpoint",
        lambda output_root, task: tmp_path / "checkpoint-100",
    )

    def run(name, command, state_path, state, log_path, expected_outputs):
        nonlocal calls
        calls += 1
        state.setdefault("stages", {})[name] = {"status": "error", "attempt": calls}
        raise bakeoff.StageExecutionError("interrupted")

    monkeypatch.setattr(bakeoff, "run_stage", run)
    unit = {
        "unit_id": "mt_m2m100_418m__vi_to_en",
        "model": "facebook/m2m100_418M",
        "revision": "revision",
        "model_family": "m2m100",
        "direction": "vi_to_en",
    }

    with pytest.raises(bakeoff.StageExecutionError):
        train_unit(
            unit=unit,
            task="mt",
            round_name="pilot",
            steps=400,
            config=config(),
            python=sys.executable,
            state_dir=tmp_path,
            state_path=tmp_path / "state.json",
            state={"stages": {}},
            profile=hyperparameter_profiles(config(), "mt")[-1],
        )

    assert calls == bakeoff.TRAIN_STAGE_MAX_ATTEMPTS


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
    identity = (
        project_root
        / "data/reports/model_bakeoff/board-evidence/qcs6490-identity.json"
    )
    identity.parent.mkdir(parents=True)
    identity.write_text(
        json.dumps(
            {
                "version": 1,
                "capture_source": "linux_sysfs_device_tree",
                "captured_at": "2026-10-06T12:00:00+07:00",
                "architecture": "aarch64",
                "board_model": "Dragonwing RB3 Gen 2 Vision Kit",
                "device_tree_compatible": ["qcom,qcs6490-rb3gen2"],
                "soc_family": "Qualcomm QCS6490",
                "soc_id": "QCS6490",
                "kernel_release": "6.1",
            }
        ),
        encoding="utf-8",
    )
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
            "architecture": "aarch64",
            "identity_evidence_path": str(identity.relative_to(project_root)),
            "identity_evidence_sha256": bakeoff.sha256(identity),
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

    report["measured_at"] = "2026-10-08T12:00:01+07:00"
    passed, failures = validate_deployment_report(report, expected, metrics, 30, project_root)
    assert passed is False
    assert "device.identity_evidence:not_same_session" in failures
    report["measured_at"] = "2026-10-06T12:00:00+07:00"

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

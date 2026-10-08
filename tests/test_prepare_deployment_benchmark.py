from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_deployment_benchmark import (
    atomic_json_exclusive,
    build_template,
    finalize_report,
    selected_winner_specs,
)
from scripts.run_blind_candidate_suite import coverage_counts
from scripts.run_model_bakeoff import QUANTIZATION_PARITY_SLICES, sha256, tree_manifest
from src.data.quality import fingerprint_text
from src.pipeline.license_policy import license_decisions
from src.pipeline.selection_policy import selection_policy_record


def deployment_fixture(
    tmp_path: Path,
    *,
    shared_mt_candidate: bool = False,
) -> tuple[Path, dict, dict, Path]:
    project_root = tmp_path / "project"
    adapters = project_root / "adapters"
    winners: dict[str, dict] = {}
    candidate_ids = {
        "mt-en": "mt-shared" if shared_mt_candidate else "mt-en",
        "mt-vi": "mt-shared" if shared_mt_candidate else "mt-vi",
        "asr-vi": "asr-vi",
    }
    for name in ("mt-en", "mt-vi", "asr-vi"):
        adapter = adapters / name
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(name.encode())
        winners[name] = {
            "candidate_id": candidate_ids[name],
            "adapter": str(adapter),
            "adapter_manifest_sha256": tree_manifest(adapter)["manifest_sha256"],
        }
    accuracy = project_root / "accuracy.yaml"
    accuracy.write_text(
        "release_gates:\n"
        "  aggregate:\n"
        "    mt_sacrebleu_min: 25\n"
        "    mt_chrf2_min: 46\n"
        "    asr_vi_wer_max: 0.25\n",
        encoding="utf-8",
    )
    config = {
        "principles": {"candidate_a_auto_promotion_forbidden": True},
        "candidates": {
            "mt": [
                {
                    "id": candidate_id,
                    "model": f"model/{candidate_id}",
                    "revision": str(index + 1) * 40,
                    "license": {
                        "id": "mit",
                        "source": f"https://example.invalid/{candidate_id}",
                        "review_status": "approved",
                        "production_eligible": True,
                        "gpu_eligible": True,
                    },
                }
                for index, candidate_id in enumerate(
                    ("mt-shared",) if shared_mt_candidate else ("mt-en", "mt-vi")
                )
            ],
            "asr": [
                {
                    "id": "asr-vi",
                    "model": "model/asr-vi",
                    "revision": "3" * 40,
                    "license": {
                        "id": "apache-2.0",
                        "source": "https://example.invalid/asr-vi",
                        "review_status": "approved",
                        "production_eligible": True,
                        "gpu_eligible": True,
                    },
                }
            ],
        },
        "promotion_gate": {
            "mt_metrics": ["sacrebleu", "chrf2"],
            "asr_metrics": ["wer", "cer", "code_switch_wer"],
            "asr_code_switch_wer_max": 0.21,
            "critical_slices": [],
            "policy_slices": [],
            "deployment_metrics": [
                "latency_p50_ms",
                "latency_p95_ms",
                "peak_ram_bytes",
                "peak_vram_bytes",
                "model_bytes",
            ],
            "deployment_min_runs": 30,
        },
        "data": {
            "selection_dev": {
                "mt": {"sha256": "a" * 64},
                "asr": {"sha256": "b" * 64},
            },
            "accuracy_program": {
                "path": str(accuracy),
                "sha256": sha256(accuracy),
            },
        },
    }
    selection_comparison = {
        "status": "selection_complete",
        "scope": "research",
        "candidate_a_freeze": "candidate-a-freeze.json",
        "candidate_a_locked_evaluations": {},
        "selection_sha256": {
            "mt": "a" * 64,
            "asr": "b" * 64,
            "accuracy_program": sha256(accuracy),
        },
        "selection_policy": selection_policy_record(config),
        "research_license_approvals": [],
        "license_decisions": license_decisions(config, set()),
        "results": {
            "mt": {
                "winners": {
                    "en_to_vi": winners["mt-en"],
                    "vi_to_en": winners["mt-vi"],
                }
            },
            "asr": {"winners": {"vi": winners["asr-vi"]}},
        },
    }
    snapshot = project_root / "selection-snapshot.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(json.dumps(selection_comparison), encoding="utf-8")
    snapshot_record = {"path": str(snapshot), "sha256": sha256(snapshot)}

    blind_dir = project_root / "blind"
    blind_dir.mkdir()
    mt_manifest = blind_dir / "mt.jsonl"
    asr_manifest = blind_dir / "asr.jsonl"
    source = "Do not use penicillin 500 mg."
    target = "Không dùng penicillin 500 mg."
    mt_row = {
        "id": "mt",
        "source_language": "en",
        "target_language": "vi",
        "source_text": source,
        "target_text": target,
        "pair_fingerprint": fingerprint_text(source + "\x1f" + target),
        "categories": [],
    }
    mt_manifest.write_text(
        json.dumps(mt_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    audio = blind_dir / "asr.flac"
    audio.write_bytes(b"blind-audio-payload")
    transcript = "Không dùng penicillin 500 mg."
    asr_row = {
        "id": "asr",
        "language": "vi",
        "text": transcript,
        "text_fingerprint": fingerprint_text(transcript),
        "audio_path": str(audio),
        "audio_sha256": sha256(audio),
        "speaker": "speaker-1",
        "group": "group-1",
        "categories": [],
    }
    asr_manifest.write_text(
        json.dumps(asr_row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    mt_coverage = coverage_counts("mt", [mt_row])
    asr_coverage = coverage_counts("asr", [asr_row])
    manifest_paths = {"mt": mt_manifest, "asr": asr_manifest}
    candidate_config = {
        candidate["id"]: candidate
        for task in ("mt", "asr")
        for candidate in config["candidates"][task]
    }
    results = []
    opened = {"mt": {"en_to_vi": None, "vi_to_en": None}, "asr": None}
    for task, direction, winner in (
        ("mt", "en_to_vi", winners["mt-en"]),
        ("mt", "vi_to_en", winners["mt-vi"]),
        ("asr", None, winners["asr-vi"]),
    ):
        candidate = candidate_config[winner["candidate_id"]]
        manifest = manifest_paths[task]
        specification = {
            "candidate": winner["candidate_id"],
            "model": candidate["model"],
            "revision": candidate["revision"],
            "adapter": str(Path(winner["adapter"]).resolve()),
            "adapter_manifest_sha256": winner["adapter_manifest_sha256"],
            "selection_sha256": snapshot_record["sha256"],
            "blind_manifest_sha256": sha256(manifest),
            "direction": direction,
            "scope": "research",
        }
        digest = hashlib.sha256(
            json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        key = direction or "vi"
        report = blind_dir / f"{task}-{key}.json"
        if task == "mt":
            report_payload = {
                "directions": {
                    str(direction): {
                        "samples": 1,
                        "sacrebleu": 30.0,
                        "sacrebleu_bootstrap_95ci": [29.0, 31.0],
                        "chrf2": 50.0,
                        "chrf2_bootstrap_95ci": [49.0, 51.0],
                    }
                },
                "categories": {},
            }
        else:
            report_payload = {
                "samples": 1,
                "wer": 0.1,
                "wer_bootstrap_95ci": [0.08, 0.12],
                "wer_bootstrap_unit": "group",
                "wer_bootstrap_clusters": 1,
                "cer": 0.05,
                "unique_speakers": 1,
                "unique_groups": 1,
                "categories": {},
                "slices": {"code_switch": {"True": {"wer": 0.1}}},
            }
        report.write_text(json.dumps(report_payload), encoding="utf-8")
        provenance = blind_dir / f"{task}-{key}.provenance.json"
        provenance.write_text(
            json.dumps(
                {
                    "version": 1,
                    "created_at": "2026-10-06T12:00:00+07:00",
                    "specification_sha256": digest,
                    "blind_manifest_sha256": sha256(manifest),
                    "selection_sha256": snapshot_record["sha256"],
                    "report": str(report),
                    "report_bytes": report.stat().st_size,
                    "report_sha256": sha256(report),
                    "resource_run": {"return_code": 0},
                }
            ),
            encoding="utf-8",
        )
        result = {
            "evaluated_at": "2026-10-06T12:00:00+07:00",
            "candidate": specification,
            "manifest_sha256": sha256(manifest),
            "selection_sha256": snapshot_record["sha256"],
            "report": str(report),
            "report_bytes": report.stat().st_size,
            "report_sha256": sha256(report),
            "report_provenance": str(provenance),
            "critical_gate": "pass",
            "failed_slices": [],
            "coverage_gate": "pass",
            "coverage_failures": [],
            "quality_gate": "pass",
            "quality_failures": [],
            "promotion_allowed": True,
        }
        results.append(result)
        record = {
            **specification,
            "candidate_sha256": digest,
            "opened_at": "2026-10-06T12:00:00+07:00",
            "result": result,
        }
        if task == "mt":
            opened["mt"][str(direction)] = record
        else:
            opened["asr"] = record

    lock = project_root / "blind-lock.json"
    lock.write_text(
        json.dumps(
            {
                "version": 2,
                "status": "opened",
                "required_slices": {"mt": [], "asr": []},
                "minimum_coverage": {
                    "mt": {"rows": 1, "slice_samples": {}},
                    "asr": {"rows": 1, "slice_samples": {}},
                },
                "manifests": {
                    "mt": {
                        "path": str(mt_manifest),
                        "rows": 1,
                        "sha256": sha256(mt_manifest),
                        "coverage": mt_coverage,
                    },
                    "asr": {
                        "path": str(asr_manifest),
                        "rows": 1,
                        "sha256": sha256(asr_manifest),
                        "coverage": asr_coverage,
                    },
                },
                "selection": snapshot_record,
                "opened": opened,
            }
        ),
        encoding="utf-8",
    )
    config["data"]["blind_test_v2_lock"] = str(lock)
    comparison = {
        **selection_comparison,
        "status": "blind_complete",
        "selection_snapshot": snapshot_record,
        "blind_test_v2": results,
    }
    selection = project_root / "selection.json"
    selection.write_text(json.dumps(comparison), encoding="utf-8")
    return selection, comparison, config, project_root


def write_identity_evidence(
    project_root: Path,
    *,
    board_model: str = "Qualcomm QCS6490 RB3 Gen 2 Vision Kit",
    compatible: list[str] | None = None,
) -> Path:
    path = (
        project_root
        / "data/reports/model_bakeoff/board-evidence/qcs6490-identity.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "capture_source": "linux_sysfs_device_tree",
                "captured_at": "2026-10-06T12:00:00+07:00",
                "architecture": "aarch64",
                "board_model": board_model,
                "device_tree_compatible": compatible or ["qcom,qcs6490-rb3gen2"],
                "soc_family": "Qualcomm QCS6490",
                "soc_id": "QCS6490",
                "kernel_release": "6.1",
            }
        ),
        encoding="utf-8",
    )
    return path


def write_measurement_evidence(
    project_root: Path,
    identity: Path,
    artifact: Path,
    winner: dict,
    *,
    index: int,
    power_samples: list[float] | None = None,
) -> Path:
    latency_samples = [100.0 + index + run for run in range(30)]
    measured_power = power_samples if power_samples is not None else [
        2500.0 + index + run * 10.0 for run in range(30)
    ]
    temperature_samples = [45.0 + index + run * 0.1 for run in range(30)]
    path = (
        project_root
        / "data/reports/model_bakeoff/board-evidence/measurements"
        / f"winner-{index}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "capture_source": "physical_qcs6490",
                "captured_at": "2026-10-06T12:00:00+07:00",
                "task": winner["task"],
                "direction": winner.get("direction"),
                "candidate_id": winner["candidate_id"],
                "adapter_manifest_sha256": winner["adapter_manifest_sha256"],
                "identity_evidence_sha256": sha256(identity),
                "artifact_sha256": sha256(artifact),
                "artifact_bytes": artifact.stat().st_size,
                "latency_samples_ms": latency_samples,
                "power_sensor": "sysfs:ina231/system_power",
                "power_samples_mw": measured_power,
                "temperature_sensor": "sysfs:thermal_zone0/temp",
                "temperature_samples_c": temperature_samples,
                "peak_ram_bytes": 100_000_000 + index,
                "peak_vram_bytes": 0,
            }
        ),
        encoding="utf-8",
    )
    return path


def write_parity_evidence(
    project_root: Path,
    artifact: Path,
    winner: dict,
    *,
    index: int,
) -> Path:
    metric_names = (
        ("sacrebleu", "chrf2")
        if winner["task"] == "mt"
        else ("wer", "cer", "code_switch_wer")
    )
    path = (
        project_root
        / "data/reports/model_bakeoff/board-evidence/parity"
        / f"winner-{index}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "status": "pass",
                "evidence_source": "onevoice_quantization_parity",
                "evaluated_at": "2026-10-06T11:00:00+07:00",
                "task": winner["task"],
                "direction": winner.get("direction"),
                "candidate_id": winner["candidate_id"],
                "adapter_manifest_sha256": winner["adapter_manifest_sha256"],
                "artifact_sha256": sha256(artifact),
                "artifact_bytes": artifact.stat().st_size,
                "manifest_sha256": "b" * 64,
                "manifest_bytes": 1000,
                "reference_predictions_sha256": "c" * 64,
                "reference_predictions_bytes": 2000,
                "reference_provenance_sha256": "e" * 64,
                "reference_provenance_bytes": 1000,
                "quantized_predictions_sha256": "d" * 64,
                "quantized_predictions_bytes": 2000,
                "quantized_provenance_sha256": "f" * 64,
                "quantized_provenance_bytes": 1000,
                "decoding": {"num_beams": 1, "do_sample": False},
                "samples": 32,
                "bootstrap": {
                    "unit": "row" if winner["task"] == "mt" else "group",
                    "clusters": 32,
                    "repeats": 1000,
                    "seed": 20261005,
                },
                "metrics": {
                    name: {
                        "reference": 50.0 if winner["task"] == "mt" else 0.1,
                        "quantized": 50.0 if winner["task"] == "mt" else 0.1,
                        "relative_degradation": 0.0,
                        "relative_degradation_bootstrap_95ci": [0.0, 0.0],
                        "maximum_relative_degradation": 0.02,
                    }
                    for name in metric_names
                },
                "safety_slices": {
                    name: {
                        "samples": 4,
                        "reference_failures": 0,
                        "quantized_failures": 0,
                    }
                    for name in QUANTIZATION_PARITY_SLICES
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_template_binds_all_selected_winners(tmp_path: Path):
    selection, comparison, config, project_root = deployment_fixture(tmp_path)

    report = build_template(selection, comparison, config, project_root)

    assert report["status"] == "pending_physical_measurement"
    assert report["selection_comparison"]["sha256"] == sha256(selection)
    assert {
        (winner["task"], winner.get("direction"), winner["candidate_id"])
        for winner in report["winners"]
    } == {
        ("mt", "en_to_vi", "mt-en"),
        ("mt", "vi_to_en", "mt-vi"),
        ("asr", None, "asr-vi"),
    }
    assert all(len(winner["adapter_manifest_sha256"]) == 64 for winner in report["winners"])
    assert all(winner["parity_evidence_path"] == "" for winner in report["winners"])
    assert all(winner["measurement_evidence_path"] == "" for winner in report["winners"])
    assert all(winner["power_samples_mw"] == [] for winner in report["winners"])
    assert all(winner["temperature_samples_c"] == [] for winner in report["winners"])


def test_template_allows_one_mt_candidate_to_win_both_directions(tmp_path: Path):
    selection, comparison, config, project_root = deployment_fixture(
        tmp_path,
        shared_mt_candidate=True,
    )

    report = build_template(selection, comparison, config, project_root)

    assert {
        (winner["task"], winner.get("direction"), winner["candidate_id"])
        for winner in report["winners"]
    } == {
        ("mt", "en_to_vi", "mt-shared"),
        ("mt", "vi_to_en", "mt-shared"),
        ("asr", None, "asr-vi"),
    }


def test_template_rejects_blind_gate_or_snapshot_tampering(tmp_path: Path):
    selection, comparison, config, project_root = deployment_fixture(tmp_path)
    comparison["blind_test_v2"][0]["promotion_allowed"] = False
    selection.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(ValueError, match="Blind completion gate did not pass"):
        build_template(selection, comparison, config, project_root)

    selection, comparison, config, project_root = deployment_fixture(tmp_path / "other")
    comparison["results"]["mt"]["winners"]["en_to_vi"]["candidate_id"] = "other"
    selection.write_text(json.dumps(comparison), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot identity differs"):
        build_template(selection, comparison, config, project_root)


def test_finalize_rechecks_blind_report_provenance(tmp_path: Path):
    selection, comparison, config, project_root = deployment_fixture(tmp_path)
    draft = build_template(selection, comparison, config, project_root)
    report = Path(comparison["blind_test_v2"][0]["report"])
    report.write_text('{"status":"tampered"}', encoding="utf-8")

    with pytest.raises(ValueError, match="Blind report checksum mismatch"):
        finalize_report(selection, comparison, draft, config, project_root)


def test_finalize_rejects_symlinked_board_identity(tmp_path: Path):
    selection, comparison, config, project_root = deployment_fixture(tmp_path)
    draft = build_template(selection, comparison, config, project_root)
    identity = write_identity_evidence(project_root)
    link = identity.with_name("qcs6490-identity-link.json")
    try:
        link.symlink_to(identity.name)
    except OSError as exc:
        pytest.skip(f"Symlink creation is unavailable: {exc}")
    draft["device"]["identity_evidence_path"] = str(link.relative_to(project_root))

    with pytest.raises(ValueError, match="symlink or junction"):
        finalize_report(selection, comparison, draft, config, project_root)


def test_template_recomputes_blind_quality_instead_of_trusting_pass_flags(
    tmp_path: Path,
):
    selection, comparison, config, project_root = deployment_fixture(tmp_path)
    result = comparison["blind_test_v2"][0]
    report_path = Path(result["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["directions"]["en_to_vi"]["sacrebleu"] = 1.0
    report["directions"]["en_to_vi"]["sacrebleu_bootstrap_95ci"] = [1.0, 1.0]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result["report_bytes"] = report_path.stat().st_size
    result["report_sha256"] = sha256(report_path)
    provenance_path = Path(result["report_provenance"])
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["report_bytes"] = result["report_bytes"]
    provenance["report_sha256"] = result["report_sha256"]
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    lock_path = Path(config["data"]["blind_test_v2_lock"])
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["opened"]["mt"]["en_to_vi"]["result"] = result
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    selection.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(ValueError, match="gate recomputation mismatch"):
        build_template(selection, comparison, config, project_root)


def test_finalize_computes_artifact_identity_and_latency_percentiles(tmp_path: Path):
    selection, comparison, config, project_root = deployment_fixture(tmp_path)
    draft = build_template(selection, comparison, config, project_root)
    draft["measured_at"] = "2026-10-06T12:00:00+07:00"
    identity = write_identity_evidence(project_root)
    draft["device"].update(
        {
            "os": "Qc_Linux 1.6",
            "identity_evidence_path": str(identity.relative_to(project_root)),
        }
    )
    models = project_root / "models"
    models.mkdir()
    for index, winner in enumerate(draft["winners"]):
        artifact = models / f"winner-{index}.bin"
        artifact.write_bytes(f"compiled-{index}".encode())
        parity = write_parity_evidence(
            project_root,
            artifact,
            winner,
            index=index,
        )
        measurement = write_measurement_evidence(
            project_root,
            identity,
            artifact,
            winner,
            index=index,
        )
        winner.update(
            {
                "artifact_path": str(artifact.relative_to(project_root)),
                "parity_evidence_path": str(parity.relative_to(project_root)),
                "measurement_evidence_path": str(
                    measurement.relative_to(project_root)
                ),
            }
        )

    report = finalize_report(selection, comparison, draft, config, project_root)

    assert report["status"] == "pass"
    assert report["selection_comparison"]["sha256"] == sha256(selection)
    assert report["device"]["board"] == "Qualcomm QCS6490 RB3 Gen 2 Vision Kit"
    assert report["device"]["identity_evidence_sha256"] == sha256(identity)
    assert set(report["deployment_gate"]["required_metrics"]) >= {
        "power_avg_mw",
        "power_p95_mw",
        "temperature_peak_c",
    }
    assert report["deployment_gate"]["quantization_parity"] == {
        "maximum_relative_degradation": 0.02,
        "minimum_samples": 32,
        "minimum_bootstrap_repeats": 1000,
        "asr_minimum_independent_groups": 32,
        "required_zero_failure_slices": list(QUANTIZATION_PARITY_SLICES),
    }
    for index, winner in enumerate(report["winners"]):
        artifact = models / f"winner-{index}.bin"
        assert winner["measurement_runs"] == 30
        assert winner["parity_samples"] == 32
        assert winner["parity_evidence_sha256"] == sha256(
            project_root / winner["parity_evidence_path"]
        )
        assert winner["latency_p50_ms"] == pytest.approx(114.5 + index)
        assert winner["latency_p95_ms"] == pytest.approx(127.55 + index)
        assert winner["power_avg_mw"] == pytest.approx(2645.0 + index)
        assert winner["power_p95_mw"] == pytest.approx(2775.5 + index)
        assert winner["temperature_peak_c"] == pytest.approx(47.9 + index)
        assert winner["measurement_evidence_sha256"] == sha256(
            project_root / winner["measurement_evidence_path"]
        )
        assert winner["artifact_sha256"] == sha256(artifact)
        assert winner["model_bytes"] == artifact.stat().st_size


def test_finalize_rejects_changed_winner_binding(tmp_path: Path):
    selection, comparison, config, project_root = deployment_fixture(tmp_path)
    draft = build_template(selection, comparison, config, project_root)
    identity = write_identity_evidence(project_root)
    draft["device"].update(
        {
            "os": "Qc_Linux 1.6",
            "identity_evidence_path": str(identity.relative_to(project_root)),
        }
    )
    draft["winners"][0]["candidate_id"] = "other"

    with pytest.raises(ValueError, match="candidate binding changed"):
        finalize_report(selection, comparison, draft, config, project_root)


def test_finalize_fails_closed_before_publishing_invalid_board_data(tmp_path: Path):
    selection, comparison, config, project_root = deployment_fixture(tmp_path)
    draft = build_template(selection, comparison, config, project_root)
    draft["measured_at"] = "2026-10-06T12:00:00+07:00"
    identity = write_identity_evidence(
        project_root,
        board_model="Arduino Uno",
        compatible=["arduino,uno"],
    )
    draft["device"].update(
        {
            "os": "firmware",
            "identity_evidence_path": str(identity.relative_to(project_root)),
        }
    )
    models = project_root / "models"
    models.mkdir()
    for index, winner in enumerate(draft["winners"]):
        artifact = models / f"invalid-board-{index}.bin"
        artifact.write_bytes(b"not-qcs-artifact")
        parity = write_parity_evidence(
            project_root,
            artifact,
            winner,
            index=index,
        )
        measurement = write_measurement_evidence(
            project_root,
            identity,
            artifact,
            winner,
            index=index,
        )
        winner.update(
            {
                "artifact_path": str(artifact.relative_to(project_root)),
                "parity_evidence_path": str(parity.relative_to(project_root)),
                "measurement_evidence_path": str(
                    measurement.relative_to(project_root)
                ),
            }
        )

    with pytest.raises(ValueError, match="identity_evidence:board_model_not_qcs6490"):
        finalize_report(selection, comparison, draft, config, project_root)


def test_finalize_rejects_artifact_outside_models(tmp_path: Path):
    selection, comparison, config, project_root = deployment_fixture(tmp_path)
    draft = build_template(selection, comparison, config, project_root)
    identity = write_identity_evidence(project_root)
    draft["device"]["identity_evidence_path"] = str(
        identity.relative_to(project_root)
    )
    outside = tmp_path / "outside-model.bin"
    outside.write_bytes(b"compiled")
    draft["winners"][0]["artifact_path"] = str(outside)

    with pytest.raises(ValueError, match="Deployment artifact must remain under"):
        finalize_report(selection, comparison, draft, config, project_root)


def test_finalize_rejects_missing_power_or_thermal_samples(tmp_path: Path):
    selection, comparison, config, project_root = deployment_fixture(tmp_path)
    draft = build_template(selection, comparison, config, project_root)
    draft["measured_at"] = "2026-10-06T12:00:00+07:00"
    identity = write_identity_evidence(project_root)
    draft["device"].update(
        {
            "os": "Qc_Linux 1.6",
            "identity_evidence_path": str(identity.relative_to(project_root)),
        }
    )
    models = project_root / "models"
    models.mkdir()
    for index, winner in enumerate(draft["winners"]):
        artifact = models / f"winner-{index}.bin"
        artifact.write_bytes(b"compiled")
        parity = write_parity_evidence(
            project_root,
            artifact,
            winner,
            index=index,
        )
        measurement = write_measurement_evidence(
            project_root,
            identity,
            artifact,
            winner,
            index=index,
            power_samples=[],
        )
        winner.update(
            {
                "artifact_path": str(artifact.relative_to(project_root)),
                "parity_evidence_path": str(parity.relative_to(project_root)),
                "measurement_evidence_path": str(
                    measurement.relative_to(project_root)
                ),
            }
        )

    with pytest.raises(ValueError, match="power_samples_mw are empty"):
        finalize_report(selection, comparison, draft, config, project_root)


def test_evidence_writer_refuses_to_overwrite(tmp_path: Path):
    path = tmp_path / "evidence.json"
    atomic_json_exclusive(path, {"status": "first"})

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        atomic_json_exclusive(path, {"status": "second"})
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "first"


def test_selection_requires_exact_three_winner_slots(tmp_path: Path):
    _, comparison, _, _ = deployment_fixture(tmp_path)
    del comparison["results"]["mt"]["winners"]["vi_to_en"]

    with pytest.raises(ValueError, match="both MT directions"):
        selected_winner_specs(comparison)


def test_selection_requires_completed_blind_evaluation(tmp_path: Path):
    _, comparison, _, _ = deployment_fixture(tmp_path)
    comparison["status"] = "selection_complete"

    with pytest.raises(ValueError, match="completed blind evaluation"):
        selected_winner_specs(comparison)

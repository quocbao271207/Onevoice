from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_deployment_benchmark import (
    atomic_json_exclusive,
    build_template,
    finalize_report,
    selected_winner_specs,
)
from scripts.run_model_bakeoff import sha256


def deployment_fixture(tmp_path: Path) -> tuple[Path, dict, dict, Path]:
    project_root = tmp_path / "project"
    adapters = project_root / "adapters"
    winners: dict[str, dict] = {}
    for name in ("mt-en", "mt-vi", "asr-vi"):
        adapter = adapters / name
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(name.encode())
        winners[name] = {"candidate_id": name, "adapter": str(adapter)}
    comparison = {
        "status": "blind_complete",
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
    selection = project_root / "selection.json"
    selection.parent.mkdir(parents=True, exist_ok=True)
    selection.write_text(json.dumps(comparison), encoding="utf-8")
    config = {
        "promotion_gate": {
            "deployment_metrics": [
                "latency_p50_ms",
                "latency_p95_ms",
                "peak_ram_bytes",
                "peak_vram_bytes",
                "model_bytes",
            ],
            "deployment_min_runs": 30,
        }
    }
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


def test_template_binds_all_selected_winners(tmp_path: Path):
    selection, comparison, _, _ = deployment_fixture(tmp_path)

    report = build_template(selection, comparison)

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


def test_finalize_computes_artifact_identity_and_latency_percentiles(tmp_path: Path):
    selection, comparison, config, project_root = deployment_fixture(tmp_path)
    draft = build_template(selection, comparison)
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
        winner.update(
            {
                "artifact_path": str(artifact.relative_to(project_root)),
                "latency_samples_ms": [100.0 + index + run for run in range(30)],
                "peak_ram_bytes": 100_000_000 + index,
                "peak_vram_bytes": 0,
            }
        )

    report = finalize_report(selection, comparison, draft, config, project_root)

    assert report["status"] == "pass"
    assert report["selection_comparison"]["sha256"] == sha256(selection)
    assert report["device"]["board"] == "Qualcomm QCS6490 RB3 Gen 2 Vision Kit"
    assert report["device"]["identity_evidence_sha256"] == sha256(identity)
    for index, winner in enumerate(report["winners"]):
        artifact = models / f"winner-{index}.bin"
        assert winner["measurement_runs"] == 30
        assert winner["latency_p50_ms"] == pytest.approx(114.5 + index)
        assert winner["latency_p95_ms"] == pytest.approx(127.55 + index)
        assert winner["artifact_sha256"] == sha256(artifact)
        assert winner["model_bytes"] == artifact.stat().st_size


def test_finalize_rejects_changed_winner_binding(tmp_path: Path):
    selection, comparison, config, project_root = deployment_fixture(tmp_path)
    draft = build_template(selection, comparison)
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
    draft = build_template(selection, comparison)
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
    outside = tmp_path / "outside-model.bin"
    outside.write_bytes(b"not-qcs-artifact")
    for winner in draft["winners"]:
        winner.update(
            {
                "artifact_path": str(outside),
                "latency_samples_ms": [100.0 + run for run in range(30)],
                "peak_ram_bytes": 1,
                "peak_vram_bytes": 0,
            }
        )

    with pytest.raises(ValueError, match="identity_evidence:board_model_not_qcs6490"):
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

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.candidate_evidence import (
    archive_evidence,
    evidence_sidecars,
    sha256,
    verify_evidence_archive,
)
from scripts.run_baseline_benchmarks import (
    attach_adapter,
    model_load_kwargs,
    prepare_runtime,
    resolve_device,
    source_balanced_sample,
    write_predictions_checkpoint,
)
from scripts.run_asr_candidate_suite import candidate_checks as asr_candidate_checks
from scripts.run_mt_candidate_suite import candidate_checks as mt_candidate_checks


def test_explicit_benchmark_device_is_preserved():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_cpu_candidate_benchmark_rejects_reduced_precision():
    with pytest.raises(ValueError, match="only fp32"):
        model_load_kwargs("cpu", "bf16")


def test_missing_candidate_adapter_fails_closed(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Missing adapter"):
        attach_adapter(object(), tmp_path / "missing-adapter")


def test_candidate_benchmark_memory_fraction_cannot_exceed_40_percent():
    args = SimpleNamespace(device="cpu", gpu_memory_fraction=0.4001)
    with pytest.raises(ValueError, match=r"\(0, 0\.40\]"):
        prepare_runtime(args)


def test_cuda_dtype_selection_is_explicit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert model_load_kwargs("cuda", "bf16") == {"torch_dtype": torch.bfloat16}


def test_source_balanced_sample_can_take_full_manifest():
    rows = [
        {"id": "a", "source": "one"},
        {"id": "b", "source": "two"},
        {"id": "c", "source": "one"},
    ]
    selected = source_balanced_sample(rows, len(rows), seed=7)
    assert {row["id"] for row in selected} == {"a", "b", "c"}


def test_predictions_checkpoint_is_utf8_jsonl_and_replaces_stale_file(tmp_path: Path):
    path = tmp_path / "candidate_predictions.jsonl"
    path.write_text("stale", encoding="utf-8")
    predictions = [{"id": "thuốc", "hypothesis": "Không dùng 5 mg."}]

    write_predictions_checkpoint(path, predictions)

    assert path.read_text(encoding="utf-8") == (
        '{"id": "thuốc", "hypothesis": "Không dùng 5 mg."}\n'
    )
    assert not path.with_suffix(".jsonl.tmp").exists()


def test_fullscale_mt_scoring_smoke_is_bound_to_locked_manifest():
    root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (
            root
            / "data/reports/experiments/scoring/mt_bootstrap_fullscale_smoke.json"
        ).read_text(encoding="utf-8")
    )
    manifest = root / report["manifest"]["path"]

    assert report["status"] == "pass"
    assert report["evidence_type"] == "synthetic_fullscale_scoring_smoke"
    assert sha256(manifest) == report["manifest"]["sha256"]
    assert report["workload"]["predictions"] == report["manifest"]["pairs"] * 2
    assert report["result"]["overall_bleu_chrf_confidence_intervals_present"] is True
    assert report["result"]["direction_bleu_chrf_confidence_intervals_present"] is True


def test_candidate_gate_requires_every_clinical_slice():
    config = {
        "release_gates": {
            "aggregate": {"mt_sacrebleu_min": 25.0, "mt_chrf2_min": 46.0},
            "clinical_safety": {"terminology_failure_rate_max": 0.0},
        },
        "evaluation": {"required_slices": ["drug_name", "dose"]},
    }
    checks = mt_candidate_checks(
        config,
        {"sacrebleu": 30.0, "chrf2": 50.0},
        {"categories": {"drug_name": {"samples": 1, "safety_failure_rate": 0.0}}},
    )
    assert any(check["name"] == "clinical_dose" and not check["pass"] for check in checks)


def test_asr_candidate_gate_requires_code_switch_and_every_clinical_slice():
    config = {
        "release_gates": {
            "aggregate": {"asr_vi_wer_max": 0.19},
            "slices": {"asr_code_switch_wer_max": 0.21},
            "clinical_safety": {"terminology_failure_rate_max": 0.0},
        },
        "evaluation": {"required_slices": ["drug_name", "dose"]},
    }
    checks = asr_candidate_checks(
        config,
        {"wer": 0.18, "slices": {"code_switch": {"True": {"wer": 0.20}}}},
        {"categories": {"drug_name": {"samples": 1, "safety_failure_rate": 0.0}}},
    )
    assert any(check["name"] == "clinical_dose" and not check["pass"] for check in checks)
    assert all(check["pass"] for check in checks if check["name"] != "clinical_dose")


def test_candidate_evidence_bundle_has_verified_content_manifest(tmp_path: Path):
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "candidate_gate.json").write_text('{"status":"pass"}', encoding="utf-8")
    nested = output / "predictions"
    nested.mkdir()
    (nested / "clinical.jsonl").write_text('{"id":"thuốc"}\n', encoding="utf-8")

    archive, digest = archive_evidence(output)
    checksum, manifest_path = evidence_sidecars(archive)
    manifest = verify_evidence_archive(archive)

    assert checksum.read_text(encoding="utf-8") == f"{digest}  {archive.name}\n"
    assert manifest_path.is_file()
    assert manifest["archive_bytes"] == archive.stat().st_size
    assert manifest["archive_sha256"] == digest
    assert manifest["file_count"] == 2
    assert manifest["content_bytes"] == sum(item["bytes"] for item in manifest["files"])
    assert {item["path"] for item in manifest["files"]} == {
        "candidate/candidate_gate.json",
        "candidate/predictions/clinical.jsonl",
    }


def test_candidate_evidence_verification_rejects_tampered_archive(tmp_path: Path):
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "candidate_gate.json").write_text('{"status":"pass"}', encoding="utf-8")
    archive, _ = archive_evidence(output)
    archive.write_bytes(archive.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_evidence_archive(archive)


def test_candidate_evidence_verification_rejects_duplicate_manifest_paths(tmp_path: Path):
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "candidate_gate.json").write_text('{"status":"pass"}', encoding="utf-8")
    archive, _ = archive_evidence(output)
    _, manifest_path = evidence_sidecars(archive)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].append(dict(manifest["files"][0]))
    manifest["file_count"] = 2
    manifest["content_bytes"] *= 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate paths"):
        verify_evidence_archive(archive)

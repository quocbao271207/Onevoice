from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.run_baseline_benchmarks as benchmark
from scripts.candidate_evidence import (
    archive_evidence,
    derive_legacy_evidence_manifest,
    evidence_sidecars,
    sha256,
    verify_evidence_archive,
)
from scripts.run_baseline_benchmarks import (
    attach_adapter,
    load_verified_prediction_checkpoint,
    model_load_kwargs,
    prediction_checkpoint_specification,
    prediction_provenance_path,
    prepare_runtime,
    resolve_device,
    source_balanced_sample,
    validate_bakeoff_runner_generation,
    write_prediction_checkpoint,
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


def test_canonical_selection_manifest_rejects_stale_waiter_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runner = tmp_path / "scripts" / "run_model_bakeoff.py"
    selection = tmp_path / "data" / "eval" / "mt_selection_dev.jsonl"
    runner.parent.mkdir(parents=True)
    selection.parent.mkdir(parents=True)
    runner.write_text("# current runner\n", encoding="utf-8")
    selection.write_text('{}\n', encoding="utf-8")
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="Stale or unbound"):
        validate_bakeoff_runner_generation(selection, None)
    with pytest.raises(RuntimeError, match="Stale or unbound"):
        validate_bakeoff_runner_generation(selection, "0" * 64)

    validate_bakeoff_runner_generation(selection, sha256(runner))
    validate_bakeoff_runner_generation(tmp_path / "ad-hoc.jsonl", None)


def test_predictions_checkpoint_is_utf8_jsonl_and_replaces_stale_file(tmp_path: Path):
    path = tmp_path / "candidate_predictions.jsonl"
    path.write_text("stale", encoding="utf-8")
    predictions = [{"id": "thuốc", "hypothesis": "Không dùng 5 mg."}]

    write_predictions_checkpoint(path, predictions)

    assert path.read_text(encoding="utf-8") == (
        '{"id": "thuốc", "hypothesis": "Không dùng 5 mg."}\n'
    )
    assert not path.with_suffix(".jsonl.tmp").exists()


def checkpoint_args(tmp_path: Path, *, adapter: bool = True) -> SimpleNamespace:
    manifest = tmp_path / "selection.jsonl"
    manifest.write_text('{"id":"pair"}\n', encoding="utf-8")
    adapter_path = None
    if adapter:
        adapter_path = tmp_path / "adapter"
        adapter_path.mkdir()
        (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter_path / "adapter_model.safetensors").write_bytes(b"weights")
    return SimpleNamespace(
        task="mt",
        model="facebook/nllb-200-distilled-600M",
        model_revision=benchmark.MODEL_REVISIONS["facebook/nllb-200-distilled-600M"],
        adapter=adapter_path,
        manifest=manifest,
        samples=0,
        seed=20260922,
        batch_size=2,
        num_beams=4,
        device="cpu",
        precision="fp32",
        gpu_memory_fraction=0.35,
        language="vi",
        asr_prompt=None,
        mt_model_family="nllb",
        mt_direction="joint",
    )


def test_prediction_checkpoint_verifies_exact_inputs_and_payload(tmp_path: Path):
    args = checkpoint_args(tmp_path)
    path = tmp_path / "candidate_predictions.jsonl"
    predictions = [{"id": "pair", "hypothesis": "Không dùng 5 mg."}]
    specification = prediction_checkpoint_specification(args)

    provenance = write_prediction_checkpoint(
        path,
        predictions,
        specification,
        {"num_beams": 4, "generation_seconds": 1.0},
        {"resolved_device": "cpu"},
    )
    loaded = load_verified_prediction_checkpoint(path, specification)

    assert loaded is not None
    assert loaded[0] == predictions
    assert loaded[1] == provenance
    assert prediction_provenance_path(path).is_file()


def test_prediction_checkpoint_fails_closed_on_tampering_and_spec_change(tmp_path: Path):
    args = checkpoint_args(tmp_path)
    path = tmp_path / "candidate_predictions.jsonl"
    specification = prediction_checkpoint_specification(args)
    write_prediction_checkpoint(
        path,
        [{"id": "pair", "hypothesis": "Dùng 5 mg."}],
        specification,
        {"num_beams": 4},
        {"resolved_device": "cpu"},
    )

    path.write_text('{"id":"pair","hypothesis":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_verified_prediction_checkpoint(path, specification)

    write_prediction_checkpoint(
        path,
        [{"id": "pair", "hypothesis": "Dùng 5 mg."}],
        specification,
        {"num_beams": 4},
        {"resolved_device": "cpu"},
    )
    args.seed += 1
    with pytest.raises(ValueError, match="specification mismatch"):
        load_verified_prediction_checkpoint(
            path,
            prediction_checkpoint_specification(args),
        )

    args.seed -= 1
    assert args.adapter is not None
    (args.adapter / "adapter_model.safetensors").write_bytes(b"changed-weights")
    with pytest.raises(ValueError, match="specification mismatch"):
        load_verified_prediction_checkpoint(
            path,
            prediction_checkpoint_specification(args),
        )


def test_prediction_checkpoint_rejects_incomplete_pair(tmp_path: Path):
    args = checkpoint_args(tmp_path, adapter=False)
    path = tmp_path / "orphan_predictions.jsonl"
    write_predictions_checkpoint(path, [{"id": "pair"}])

    with pytest.raises(ValueError, match="Incomplete prediction checkpoint"):
        load_verified_prediction_checkpoint(
            path,
            prediction_checkpoint_specification(args),
        )


def test_resume_scoring_skips_model_loading_and_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    args = checkpoint_args(tmp_path, adapter=False)
    output_dir = tmp_path / "reports"
    prediction_path = output_dir / "resume_predictions.jsonl"
    write_prediction_checkpoint(
        prediction_path,
        [{"id": "pair", "direction": "en_to_vi", "hypothesis": "Bản dịch"}],
        prediction_checkpoint_specification(args),
        {"num_beams": 4, "generation_seconds": 1.0, "samples_per_second": 1.0},
        {"resolved_device": "cpu"},
    )

    def unexpected_inference(*_args, **_kwargs):
        raise AssertionError("resume path must not load a model or rerun inference")

    monkeypatch.setattr(benchmark, "run_mt", unexpected_inference)
    monkeypatch.setattr(benchmark, "score_mt", lambda predictions: {"samples": len(predictions)})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baseline_benchmarks.py",
            "--task",
            "mt",
            "--manifest",
            str(args.manifest),
            "--samples",
            "0",
            "--name",
            "resume",
            "--output-dir",
            str(output_dir),
            "--resume-scoring",
        ],
    )

    assert benchmark.main() == 0
    report = json.loads((output_dir / "resume.json").read_text(encoding="utf-8"))
    assert report["samples"] == 1
    assert report["scoring_resumed"] is True
    assert report["decoding"]["scoring_resumed_from_checkpoint"] is True


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


def test_legacy_evidence_manifest_is_forensic_and_not_resume_eligible(tmp_path: Path):
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "candidate_gate.json").write_text('{"status":"fail"}', encoding="utf-8")
    archive, digest = archive_evidence(output)
    _, canonical_manifest = evidence_sidecars(archive)
    canonical_manifest.unlink()

    derived_path, derived = derive_legacy_evidence_manifest(archive)

    assert derived_path.name == f"{archive.name}.derived-manifest.json"
    assert derived["evidence_status"] == "forensic_derived_from_legacy_archive"
    assert derived["resume_eligible"] is False
    assert derived["original_manifest_present"] is False
    assert derived["archive_sha256"] == digest
    assert derived["file_count"] == 1
    assert derived["files"][0]["path"] == "candidate/candidate_gate.json"
    with pytest.raises(FileNotFoundError, match="Incomplete evidence bundle"):
        verify_evidence_archive(archive)


def test_legacy_evidence_manifest_refuses_canonical_bundle(tmp_path: Path):
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "candidate_gate.json").write_text('{"status":"pass"}', encoding="utf-8")
    archive, _ = archive_evidence(output)

    with pytest.raises(ValueError, match="Canonical evidence manifest already exists"):
        derive_legacy_evidence_manifest(archive)


def test_legacy_evidence_manifest_refuses_reserved_output_path(tmp_path: Path):
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "candidate_gate.json").write_text('{"status":"fail"}', encoding="utf-8")
    archive, _ = archive_evidence(output)
    checksum, canonical_manifest = evidence_sidecars(archive)
    canonical_manifest.unlink()

    for reserved in (archive, checksum, canonical_manifest):
        with pytest.raises(ValueError, match="distinct .*derived-manifest.json"):
            derive_legacy_evidence_manifest(archive, reserved)


def test_legacy_evidence_manifest_rejects_unsafe_archive_member(tmp_path: Path):
    archive = tmp_path / "legacy.tar.gz"
    payload = b"unsafe"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    checksum, _ = evidence_sidecars(archive)
    checksum.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe or non-file"):
        derive_legacy_evidence_manifest(archive)

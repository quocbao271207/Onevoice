from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.candidate_evidence as candidate_evidence
import scripts.run_asr_candidate_suite as asr_suite
import scripts.run_baseline_benchmarks as benchmark
import scripts.run_mt_candidate_suite as mt_suite
from scripts.candidate_evidence import (
    archive_evidence,
    derive_legacy_evidence_manifest,
    evidence_sidecars,
    sha256,
    verify_evidence_archive,
)
from scripts.run_baseline_benchmarks import (
    attach_adapter,
    encode_mt_source_batch,
    load_verified_prediction_checkpoint,
    model_load_kwargs,
    prediction_checkpoint_specification,
    prediction_provenance_path,
    prepare_asr_input_features,
    prepare_runtime,
    resolve_device,
    source_balanced_sample,
    validate_asr_batch_audio_duration,
    validate_asr_batch_generation_completed,
    validate_bakeoff_runner_generation,
    validate_mt_batch_source_lengths,
    validate_mt_batch_generation_completed,
    write_prediction_checkpoint,
    write_predictions_checkpoint,
)
from scripts.run_asr_candidate_suite import candidate_checks as asr_candidate_checks
from scripts.run_mt_candidate_suite import candidate_checks as mt_candidate_checks


@pytest.mark.parametrize("suite", [asr_suite, mt_suite])
def test_candidate_gpu_benchmark_waits_for_spawn_capacity(
    suite, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    events: list[str] = []

    def approve_capacity(limits: dict, evidence_path: Path) -> dict:
        events.append("capacity")
        return {
            "samples": 1,
            "evidence_log": str(evidence_path),
            "approval": {"allowed": True},
        }

    class Process:
        def __init__(self, *_args, **_kwargs):
            assert events == ["capacity"]
            events.append("spawn")

    def monitor(_process, _path: Path, _limits: dict) -> dict:
        events.append("monitor")
        return {"return_code": 0}

    monkeypatch.setattr(suite, "wait_for_gpu_spawn_capacity", approve_capacity)
    monkeypatch.setattr(suite.subprocess, "Popen", Process)
    monkeypatch.setattr(suite, "monitor_process", monitor)
    output_dir = tmp_path / "candidate"
    output_dir.mkdir()

    summary = suite.run_benchmark(
        adapter=tmp_path / "adapter",
        manifest=tmp_path / "manifest.jsonl",
        name="selection",
        output_dir=output_dir,
        device="cuda",
        precision="bf16",
        batch_size=1,
        num_beams=1,
        gpu_memory_fraction=0.35,
        utilization_limits={"sentinel": True},
    )

    assert events == ["capacity", "spawn", "monitor"]
    assert summary["spawn_capacity"]["approval"]["allowed"] is True


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


def test_asr_features_follow_wrapped_whisper_encoder_dtype():
    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = torch.nn.Conv1d(80, 80, 3).to(dtype=torch.bfloat16)

    class BaseModel:
        def __init__(self) -> None:
            self.encoder = Encoder()

        def get_encoder(self) -> Encoder:
            return self.encoder

    class PeftLikeWrapper:
        def __init__(self) -> None:
            self.base = BaseModel()
            self.lora_parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

        def get_base_model(self) -> BaseModel:
            return self.base

    inputs = torch.randn(2, 80, 3000, dtype=torch.float32)
    prepared = prepare_asr_input_features(inputs, PeftLikeWrapper(), "cpu")

    assert prepared.dtype == torch.bfloat16
    assert prepared.device.type == "cpu"


def test_asr_benchmark_rejects_audio_beyond_whisper_window():
    arrays = [
        torch.zeros(16000).numpy(),
        torch.zeros(16001).numpy(),
    ]
    rows = [{"id": "within-limit"}, {"id": "too-long"}]

    with pytest.raises(ValueError, match="too-long.*1.000s.*limit is 1.000s"):
        validate_asr_batch_audio_duration(
            arrays,
            rows,
            sample_rate=16000,
            max_duration_seconds=1.0,
        )


def test_asr_benchmark_rejects_generation_without_eos():
    generated = torch.tensor(
        [
            [2, 10, 1, 0],
            [2, 11, 12, 0],
        ]
    )
    rows = [{"id": "complete"}, {"id": "incomplete"}]

    with pytest.raises(RuntimeError, match="incomplete.*did not produce EOS"):
        validate_asr_batch_generation_completed(
            generated,
            rows,
            eos_token_id=1,
            pad_token_id=0,
        )


def test_source_balanced_sample_can_take_full_manifest():
    rows = [
        {"id": "a", "source": "one"},
        {"id": "b", "source": "two"},
        {"id": "c", "source": "one"},
    ]
    selected = source_balanced_sample(rows, len(rows), seed=7)
    assert {row["id"] for row in selected} == {"a", "b", "c"}


def test_mt_benchmark_rejects_overlength_sources_instead_of_truncating():
    encoded = {
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
    }
    rows = [{"id": "within-limit"}, {"id": "too-long"}]

    with pytest.raises(ValueError, match="too-long.*5 tokens.*limit is 4"):
        validate_mt_batch_source_lengths(encoded, rows, max_source_tokens=4)

    validate_mt_batch_source_lengths(encoded, rows, max_source_tokens=5)


def test_mt_benchmark_encoder_never_requests_truncation():
    calls = []

    def tokenizer(texts, **kwargs):
        calls.append((texts, kwargs))
        return {"attention_mask": torch.tensor([[1, 1]])}

    encoded = encode_mt_source_batch(tokenizer, ["Do not give aspirin."])

    assert encoded["attention_mask"].shape == (1, 2)
    assert calls == [
        (
            ["Do not give aspirin."],
            {"padding": True, "truncation": False, "return_tensors": "pt"},
        )
    ]


def test_mt_benchmark_rejects_any_generation_without_eos():
    generated = torch.tensor(
        [
            [1, 10, 1, 0],
            [1, 11, 12, 0],
        ]
    )
    rows = [{"id": "complete"}, {"id": "incomplete"}]

    with pytest.raises(RuntimeError, match="incomplete.*did not produce EOS"):
        validate_mt_batch_generation_completed(generated, rows, eos_token_id=1)

    validate_mt_batch_generation_completed(generated[:1], rows[:1], eos_token_id=1)

    malformed = torch.tensor([[1, 10, 1, 12]])
    with pytest.raises(RuntimeError, match="produced tokens after EOS"):
        validate_mt_batch_generation_completed(
            malformed,
            [{"id": "malformed"}],
            eos_token_id=1,
            pad_token_id=0,
        )


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

    assert specification["mt_inference_contract"] == {
        "max_source_tokens": 256,
        "max_new_tokens": 256,
        "source_truncation": False,
        "require_eos": True,
    }
    args.task = "asr"
    assert "mt_inference_contract" not in prediction_checkpoint_specification(args)
    assert prediction_checkpoint_specification(args)["asr_inference_contract"] == {
        "max_input_duration_seconds": 30.0,
        "max_new_tokens": 225,
        "require_eos": True,
    }
    args.task = "mt"
    assert "asr_inference_contract" not in prediction_checkpoint_specification(args)

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


def test_prediction_checkpoint_rejects_legacy_generation_contract(tmp_path: Path):
    args = checkpoint_args(tmp_path)
    path = tmp_path / "legacy_predictions.jsonl"
    specification = prediction_checkpoint_specification(args)
    write_prediction_checkpoint(
        path,
        [{"id": "pair", "hypothesis": "Dùng 5 mg."}],
        specification,
        {"num_beams": 4},
        {"resolved_device": "cpu"},
    )
    provenance_path = prediction_provenance_path(path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["specification"].pop("mt_inference_contract")
    provenance["specification_sha256"] = benchmark.canonical_sha256(
        provenance["specification"]
    )
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="specification mismatch"):
        load_verified_prediction_checkpoint(path, specification)


def test_asr_prediction_checkpoint_rejects_legacy_generation_contract(tmp_path: Path):
    args = checkpoint_args(tmp_path)
    args.task = "asr"
    path = tmp_path / "legacy_asr_predictions.jsonl"
    specification = prediction_checkpoint_specification(args)
    write_prediction_checkpoint(
        path,
        [{"id": "clip", "hypothesis": "Không dùng aspirin."}],
        specification,
        {"num_beams": 1},
        {"resolved_device": "cpu"},
    )
    provenance_path = prediction_provenance_path(path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["specification"].pop("asr_inference_contract")
    provenance["specification_sha256"] = benchmark.canonical_sha256(
        provenance["specification"]
    )
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="specification mismatch"):
        load_verified_prediction_checkpoint(path, specification)


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


def test_candidate_evidence_verification_rejects_duplicate_json_key(
    tmp_path: Path,
):
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "candidate_gate.json").write_text('{"status":"pass"}', encoding="utf-8")
    archive, _ = archive_evidence(output)
    _, manifest_path = evidence_sidecars(archive)
    payload = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        payload.replace(
            '{\n  "schema_version"',
            '{\n  "schema_version": 1,\n  "schema_version"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        verify_evidence_archive(archive)


def test_candidate_evidence_verification_enforces_archive_byte_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "candidate_gate.json").write_text('{"status":"pass"}', encoding="utf-8")
    archive, _ = archive_evidence(output)
    monkeypatch.setattr(candidate_evidence, "MAX_EVIDENCE_ARCHIVE_BYTES", 1)

    with pytest.raises(ValueError, match="exceeds 1 bytes"):
        verify_evidence_archive(archive)


def test_candidate_evidence_verification_rejects_linked_archive(tmp_path: Path):
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "candidate_gate.json").write_text('{"status":"pass"}', encoding="utf-8")
    archive, _ = archive_evidence(output)
    link = archive.with_name("linked.tar.gz")
    checksum, manifest = evidence_sidecars(archive)
    linked_checksum, linked_manifest = evidence_sidecars(link)
    try:
        link.symlink_to(archive.name)
        linked_checksum.symlink_to(checksum.name)
        linked_manifest.symlink_to(manifest.name)
    except OSError as exc:
        pytest.skip(f"Symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="symlink or junction"):
        verify_evidence_archive(link)


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

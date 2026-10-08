from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

import scripts.seal_quantization_parity as parity_sealer
from scripts.candidate_evidence import canonical_sha256
from scripts.capture_compiled_predictions import resolve_command_template
from scripts.run_model_bakeoff import (
    QUANTIZATION_PARITY_SLICES,
    quantization_parity_evidence_failures,
    sha256,
)
from scripts.seal_quantization_parity import (
    seal_quantization_parity,
    write_exclusive,
)


NOW = datetime.fromisoformat("2026-10-09T14:00:00+07:00")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_provenances(
    *,
    task: str,
    direction: str | None,
    candidate_id: str,
    adapter_manifest_sha256: str,
    artifact: Path,
    manifest: Path,
    reference: Path,
    quantized: Path,
) -> tuple[Path, Path]:
    reference_rows = len(reference.read_text(encoding="utf-8").splitlines())
    quantized_rows = len(quantized.read_text(encoding="utf-8").splitlines())
    specification = {
        "task": task,
        "mt_direction": direction if task == "mt" else None,
        "adapter": {"manifest_sha256": adapter_manifest_sha256},
        "manifest": {
            "path": str(manifest.resolve()),
            "bytes": manifest.stat().st_size,
            "sha256": sha256(manifest),
        },
        "num_beams": 1,
    }
    reference_payload = {
        "schema_version": 1,
        "specification": specification,
        "specification_sha256": canonical_sha256(specification),
        "predictions": {
            "path": str(reference.resolve()),
            "bytes": reference.stat().st_size,
            "sha256": sha256(reference),
            "rows": reference_rows,
        },
        "decoding": {"num_beams": 1},
        "runtime": {"resolved_device": "cpu"},
    }
    reference_provenance = reference.with_suffix(".provenance.json")
    reference_provenance.write_text(json.dumps(reference_payload), encoding="utf-8")

    template = ("compiled-runtime", "{artifact}", "{manifest}", "{output_predictions}")
    resolved = resolve_command_template(
        template,
        artifact_path=artifact,
        manifest_path=manifest,
        output_predictions_path=quantized,
    )
    quantized_payload = {
        "schema_version": 1,
        "evidence_source": "onevoice_compiled_prediction_capture",
        "started_at": NOW.isoformat(),
        "completed_at": NOW.isoformat(),
        "duration_seconds": 1.0,
        "runtime_kind": "qnn_context_binary",
        "task": task,
        "direction": direction,
        "candidate_id": candidate_id,
        "adapter_manifest_sha256": adapter_manifest_sha256,
        "decoding": {"num_beams": 1, "do_sample": False},
        "artifact": {
            "path": str(artifact.resolve()),
            "bytes": artifact.stat().st_size,
            "sha256": sha256(artifact),
        },
        "manifest": {
            "path": str(manifest.resolve()),
            "bytes": manifest.stat().st_size,
            "sha256": sha256(manifest),
        },
        "predictions": {
            "path": str(quantized.resolve()),
            "bytes": quantized.stat().st_size,
            "sha256": sha256(quantized),
            "rows": quantized_rows,
        },
        "command": {
            "template": list(template),
            "template_sha256": canonical_sha256(list(template)),
            "resolved_argv": list(resolved),
            "resolved_sha256": canonical_sha256(list(resolved)),
            "executable": "compiled-runtime",
        },
        "return_code": 0,
    }
    quantized_provenance = quantized.with_suffix(".provenance.json")
    quantized_provenance.write_text(json.dumps(quantized_payload), encoding="utf-8")
    return reference_provenance, quantized_provenance


def mt_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    manifest_rows = []
    predictions = []
    for index in range(32):
        source = f"Give aspirin {index + 1} mg now."
        target = f"Cho aspirin {index + 1} mg ngay."
        manifest_rows.append(
            {
                "id": f"mt-{index:03d}",
                "source_text": source,
                "target_text": target,
                "categories": list(QUANTIZATION_PARITY_SLICES),
                "terminology": {},
            }
        )
        predictions.append(
            {
                "id": f"mt-{index:03d}",
                "direction": "en_to_vi",
                "source": source,
                "reference": target,
                "hypothesis": target,
                "categories": list(QUANTIZATION_PARITY_SLICES),
                "terminology": {},
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    reference = tmp_path / "reference.jsonl"
    quantized = tmp_path / "quantized.jsonl"
    artifact = tmp_path / "winner.bin"
    write_jsonl(manifest, manifest_rows)
    write_jsonl(reference, predictions)
    write_jsonl(quantized, predictions)
    artifact.write_bytes(b"compiled-qnn-context")
    return manifest, reference, quantized, artifact


def seal_mt(tmp_path: Path, **overrides):
    manifest, reference, quantized, artifact = mt_fixture(tmp_path)
    reference_provenance, quantized_provenance = write_provenances(
        task="mt",
        direction="en_to_vi",
        candidate_id="mt-winner",
        adapter_manifest_sha256="a" * 64,
        artifact=artifact,
        manifest=manifest,
        reference=reference,
        quantized=quantized,
    )
    arguments = {
        "task": "mt",
        "direction": "en_to_vi",
        "candidate_id": "mt-winner",
        "adapter_manifest_sha256": "a" * 64,
        "artifact_path": artifact,
        "manifest_path": manifest,
        "reference_predictions_path": reference,
        "reference_provenance_path": reference_provenance,
        "quantized_predictions_path": quantized,
        "quantized_provenance_path": quantized_provenance,
        "now": NOW,
    }
    arguments.update(overrides)
    return seal_quantization_parity(**arguments)


def test_sealer_recomputes_identical_mt_parity_and_all_safety_slices(tmp_path: Path):
    evidence = seal_mt(tmp_path)

    assert evidence["status"] == "pass"
    assert evidence["samples"] == 32
    assert evidence["bootstrap"] == {
        "unit": "row",
        "clusters": 32,
        "repeats": 1000,
        "seed": 20261005,
    }
    assert set(evidence["metrics"]) == {"sacrebleu", "chrf2"}
    for metric in evidence["metrics"].values():
        assert metric["relative_degradation"] == pytest.approx(0.0)
        assert metric["relative_degradation_bootstrap_95ci"] == pytest.approx(
            [0.0, 0.0]
        )
    assert set(evidence["safety_slices"]) == set(QUANTIZATION_PARITY_SLICES)
    assert all(
        record["reference_failures"] == record["quantized_failures"] == 0
        for record in evidence["safety_slices"].values()
    )
    assert quantization_parity_evidence_failures(evidence) == []

    improved_from_zero = json.loads(json.dumps(evidence))
    improved_from_zero["metrics"]["sacrebleu"].update(
        {
            "reference": 0.0,
            "quantized": 1.0,
            "relative_degradation": 0.0,
            "relative_degradation_bootstrap_95ci": [0.0, 0.0],
        }
    )
    assert quantization_parity_evidence_failures(improved_from_zero) == []

    evidence["untrusted_extra"] = True
    assert "fields_invalid" in quantization_parity_evidence_failures(evidence)


def test_sealer_rejects_quantized_quality_and_clinical_regression(tmp_path: Path):
    manifest, reference, quantized, artifact = mt_fixture(tmp_path)
    rows = [json.loads(line) for line in quantized.read_text(encoding="utf-8").splitlines()]
    for index, row in enumerate(rows):
        row["hypothesis"] = f"Bệnh nhân ổn định {index + 1}."
    write_jsonl(quantized, rows)
    reference_provenance, quantized_provenance = write_provenances(
        task="mt",
        direction="en_to_vi",
        candidate_id="mt-winner",
        adapter_manifest_sha256="a" * 64,
        artifact=artifact,
        manifest=manifest,
        reference=reference,
        quantized=quantized,
    )

    with pytest.raises(ValueError, match="Quantization parity failed"):
        seal_quantization_parity(
            task="mt",
            direction="en_to_vi",
            candidate_id="mt-winner",
            adapter_manifest_sha256="a" * 64,
            artifact_path=artifact,
            manifest_path=manifest,
            reference_predictions_path=reference,
            reference_provenance_path=reference_provenance,
            quantized_predictions_path=quantized,
            quantized_provenance_path=quantized_provenance,
            now=NOW,
        )


def test_sealer_rejects_prediction_metadata_or_manifest_drift(tmp_path: Path):
    manifest, reference, quantized, artifact = mt_fixture(tmp_path)
    rows = [json.loads(line) for line in quantized.read_text(encoding="utf-8").splitlines()]
    rows[0]["source"] = "different input"
    write_jsonl(quantized, rows)
    reference_provenance, quantized_provenance = write_provenances(
        task="mt",
        direction="en_to_vi",
        candidate_id="mt-winner",
        adapter_manifest_sha256="a" * 64,
        artifact=artifact,
        manifest=manifest,
        reference=reference,
        quantized=quantized,
    )

    with pytest.raises(ValueError, match="metadata differs"):
        seal_quantization_parity(
            task="mt",
            direction="en_to_vi",
            candidate_id="mt-winner",
            adapter_manifest_sha256="a" * 64,
            artifact_path=artifact,
            manifest_path=manifest,
            reference_predictions_path=reference,
            reference_provenance_path=reference_provenance,
            quantized_predictions_path=quantized,
            quantized_provenance_path=quantized_provenance,
            now=NOW,
        )


def test_sealer_rejects_compiled_artifact_or_decoding_provenance_drift(tmp_path: Path):
    manifest, reference, quantized, artifact = mt_fixture(tmp_path)
    reference_provenance, quantized_provenance = write_provenances(
        task="mt",
        direction="en_to_vi",
        candidate_id="mt-winner",
        adapter_manifest_sha256="a" * 64,
        artifact=artifact,
        manifest=manifest,
        reference=reference,
        quantized=quantized,
    )
    payload = json.loads(quantized_provenance.read_text(encoding="utf-8"))
    payload["decoding"]["num_beams"] = 4
    quantized_provenance.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Quantized prediction provenance failed"):
        seal_quantization_parity(
            task="mt",
            direction="en_to_vi",
            candidate_id="mt-winner",
            adapter_manifest_sha256="a" * 64,
            artifact_path=artifact,
            manifest_path=manifest,
            reference_predictions_path=reference,
            reference_provenance_path=reference_provenance,
            quantized_predictions_path=quantized,
            quantized_provenance_path=quantized_provenance,
            now=NOW,
        )


def test_asr_parity_requires_and_accepts_32_independent_groups(tmp_path: Path):
    manifest_rows = []
    predictions = []
    expectations = {
        name: [["aspirin"]] for name in QUANTIZATION_PARITY_SLICES
    }
    for index in range(32):
        identifier = f"asr-{index:03d}"
        group_index = min(index, 30)
        manifest_rows.append(
            {
                "id": identifier,
                "text": "aspirin five milligrams",
                "categories": list(QUANTIZATION_PARITY_SLICES),
                "safety_expectations": expectations,
                "speaker": f"speaker-{group_index:03d}",
                "group": f"group-{group_index:03d}",
            }
        )
        predictions.append(
            {
                "id": identifier,
                "reference": "aspirin five milligrams",
                "hypothesis": "aspirin five milligrams",
                "categories": list(QUANTIZATION_PARITY_SLICES),
                "safety_expectations": expectations,
                "speaker": f"speaker-{group_index:03d}",
                "group": f"group-{group_index:03d}",
                "code_switch": True,
            }
        )
    manifest = tmp_path / "asr-manifest.jsonl"
    reference = tmp_path / "asr-reference.jsonl"
    quantized = tmp_path / "asr-quantized.jsonl"
    artifact = tmp_path / "asr.bin"
    write_jsonl(manifest, manifest_rows)
    write_jsonl(reference, predictions)
    write_jsonl(quantized, predictions)
    artifact.write_bytes(b"compiled-asr")
    reference_provenance, quantized_provenance = write_provenances(
        task="asr",
        direction=None,
        candidate_id="asr-winner",
        adapter_manifest_sha256="b" * 64,
        artifact=artifact,
        manifest=manifest,
        reference=reference,
        quantized=quantized,
    )

    with pytest.raises(ValueError, match="at least 32 independent"):
        seal_quantization_parity(
            task="asr",
            direction=None,
            candidate_id="asr-winner",
            adapter_manifest_sha256="b" * 64,
            artifact_path=artifact,
            manifest_path=manifest,
            reference_predictions_path=reference,
            reference_provenance_path=reference_provenance,
            quantized_predictions_path=quantized,
            quantized_provenance_path=quantized_provenance,
            now=NOW,
        )

    manifest_rows[-1]["speaker"] = "speaker-031"
    manifest_rows[-1]["group"] = "group-031"
    predictions[-1]["speaker"] = "speaker-031"
    predictions[-1]["group"] = "group-031"
    write_jsonl(manifest, manifest_rows)
    write_jsonl(reference, predictions)
    write_jsonl(quantized, predictions)
    reference_provenance, quantized_provenance = write_provenances(
        task="asr",
        direction=None,
        candidate_id="asr-winner",
        adapter_manifest_sha256="b" * 64,
        artifact=artifact,
        manifest=manifest,
        reference=reference,
        quantized=quantized,
    )
    evidence = seal_quantization_parity(
        task="asr",
        direction=None,
        candidate_id="asr-winner",
        adapter_manifest_sha256="b" * 64,
        artifact_path=artifact,
        manifest_path=manifest,
        reference_predictions_path=reference,
        reference_provenance_path=reference_provenance,
        quantized_predictions_path=quantized,
        quantized_provenance_path=quantized_provenance,
        now=NOW,
    )
    assert evidence["bootstrap"]["unit"] == "group"
    assert evidence["bootstrap"]["clusters"] == 32
    assert set(evidence["metrics"]) == {"wer", "cer", "code_switch_wer"}


def test_parity_writer_is_immutable(tmp_path: Path):
    output = tmp_path / "parity.json"
    write_exclusive(output, {"version": 1})

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_exclusive(output, {"version": 2})
    assert json.loads(output.read_text(encoding="utf-8"))["version"] == 1


def test_sealer_rejects_duplicate_prediction_fields(tmp_path: Path):
    manifest, reference, quantized, artifact = mt_fixture(tmp_path)
    first, *remaining = quantized.read_text(encoding="utf-8").splitlines()
    row = json.loads(first)
    duplicate = (
        json.dumps(row, ensure_ascii=False)[:-1]
        + ',"hypothesis":"tampered duplicate"}'
    )
    quantized.write_text(
        "\n".join([duplicate, *remaining]) + "\n",
        encoding="utf-8",
    )
    reference_provenance, quantized_provenance = write_provenances(
        task="mt",
        direction="en_to_vi",
        candidate_id="mt-winner",
        adapter_manifest_sha256="a" * 64,
        artifact=artifact,
        manifest=manifest,
        reference=reference,
        quantized=quantized,
    )

    with pytest.raises(ValueError, match="strict UTF-8 JSONL"):
        seal_quantization_parity(
            task="mt",
            direction="en_to_vi",
            candidate_id="mt-winner",
            adapter_manifest_sha256="a" * 64,
            artifact_path=artifact,
            manifest_path=manifest,
            reference_predictions_path=reference,
            reference_provenance_path=reference_provenance,
            quantized_predictions_path=quantized,
            quantized_provenance_path=quantized_provenance,
            now=NOW,
        )


def test_sealer_caps_compiled_artifact_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest, reference, quantized, artifact = mt_fixture(tmp_path)
    reference_provenance, quantized_provenance = write_provenances(
        task="mt",
        direction="en_to_vi",
        candidate_id="mt-winner",
        adapter_manifest_sha256="a" * 64,
        artifact=artifact,
        manifest=manifest,
        reference=reference,
        quantized=quantized,
    )
    monkeypatch.setattr(parity_sealer, "MAX_DEPLOYMENT_ARTIFACT_BYTES", 4)

    with pytest.raises(ValueError, match="exceeds 4 bytes"):
        seal_quantization_parity(
            task="mt",
            direction="en_to_vi",
            candidate_id="mt-winner",
            adapter_manifest_sha256="a" * 64,
            artifact_path=artifact,
            manifest_path=manifest,
            reference_predictions_path=reference,
            reference_provenance_path=reference_provenance,
            quantized_predictions_path=quantized,
            quantized_provenance_path=quantized_provenance,
            now=NOW,
        )


def test_parity_writer_rejects_non_finite_json(tmp_path: Path):
    output = tmp_path / "parity.json"

    with pytest.raises(ValueError, match="not serializable"):
        write_exclusive(output, {"metric": float("nan")})

    assert not output.exists()


def test_sealer_rejects_linked_prediction_input(tmp_path: Path):
    manifest, reference, quantized, artifact = mt_fixture(tmp_path)
    linked = tmp_path / "quantized-link.jsonl"
    try:
        linked.symlink_to(quantized.name)
    except OSError as exc:
        pytest.skip(f"Symlink creation is unavailable: {exc}")
    reference_provenance, quantized_provenance = write_provenances(
        task="mt",
        direction="en_to_vi",
        candidate_id="mt-winner",
        adapter_manifest_sha256="a" * 64,
        artifact=artifact,
        manifest=manifest,
        reference=reference,
        quantized=quantized,
    )

    with pytest.raises(ValueError, match="symlink or junction"):
        seal_quantization_parity(
            task="mt",
            direction="en_to_vi",
            candidate_id="mt-winner",
            adapter_manifest_sha256="a" * 64,
            artifact_path=artifact,
            manifest_path=manifest,
            reference_predictions_path=reference,
            reference_provenance_path=reference_provenance,
            quantized_predictions_path=linked,
            quantized_provenance_path=quantized_provenance,
            now=NOW,
        )

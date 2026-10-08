from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from scripts.run_model_bakeoff import (
    QUANTIZATION_PARITY_SLICES,
    quantization_parity_evidence_failures,
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
    arguments = {
        "task": "mt",
        "direction": "en_to_vi",
        "candidate_id": "mt-winner",
        "adapter_manifest_sha256": "a" * 64,
        "artifact_path": artifact,
        "manifest_path": manifest,
        "reference_predictions_path": reference,
        "quantized_predictions_path": quantized,
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

    with pytest.raises(ValueError, match="Quantization parity failed"):
        seal_quantization_parity(
            task="mt",
            direction="en_to_vi",
            candidate_id="mt-winner",
            adapter_manifest_sha256="a" * 64,
            artifact_path=artifact,
            manifest_path=manifest,
            reference_predictions_path=reference,
            quantized_predictions_path=quantized,
            now=NOW,
        )


def test_sealer_rejects_prediction_metadata_or_manifest_drift(tmp_path: Path):
    manifest, reference, quantized, artifact = mt_fixture(tmp_path)
    rows = [json.loads(line) for line in quantized.read_text(encoding="utf-8").splitlines()]
    rows[0]["source"] = "different input"
    write_jsonl(quantized, rows)

    with pytest.raises(ValueError, match="metadata differs"):
        seal_quantization_parity(
            task="mt",
            direction="en_to_vi",
            candidate_id="mt-winner",
            adapter_manifest_sha256="a" * 64,
            artifact_path=artifact,
            manifest_path=manifest,
            reference_predictions_path=reference,
            quantized_predictions_path=quantized,
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

    with pytest.raises(ValueError, match="at least 32 independent"):
        seal_quantization_parity(
            task="asr",
            direction=None,
            candidate_id="asr-winner",
            adapter_manifest_sha256="b" * 64,
            artifact_path=artifact,
            manifest_path=manifest,
            reference_predictions_path=reference,
            quantized_predictions_path=quantized,
            now=NOW,
        )

    manifest_rows[-1]["speaker"] = "speaker-031"
    manifest_rows[-1]["group"] = "group-031"
    predictions[-1]["speaker"] = "speaker-031"
    predictions[-1]["group"] = "group-031"
    write_jsonl(manifest, manifest_rows)
    write_jsonl(reference, predictions)
    write_jsonl(quantized, predictions)
    evidence = seal_quantization_parity(
        task="asr",
        direction=None,
        candidate_id="asr-winner",
        adapter_manifest_sha256="b" * 64,
        artifact_path=artifact,
        manifest_path=manifest,
        reference_predictions_path=reference,
        quantized_predictions_path=quantized,
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

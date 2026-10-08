"""Seal float-vs-compiled prediction parity for one selected winner.

Both prediction files must cover the same locked manifest with identical
non-hypothesis metadata.  Metrics, paired bootstrap degradation intervals and
clinical safety slices are recomputed locally before an immutable evidence
file is written.  This evidence is required by the physical deployment gate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from jiwer import process_characters, process_words
from sacrebleu.metrics import BLEU, CHRF


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_benchmarks import score_asr, score_mt  # noqa: E402
from scripts.candidate_evidence import canonical_sha256  # noqa: E402
from scripts.capture_compiled_predictions import (  # noqa: E402
    compiled_prediction_provenance_failures,
)
from scripts.run_model_bakeoff import (  # noqa: E402
    MAX_QUANTIZATION_PARITY_EVIDENCE_BYTES,
    QUANTIZATION_PARITY_BOOTSTRAP_REPEATS,
    QUANTIZATION_PARITY_MAX_RELATIVE_DEGRADATION,
    QUANTIZATION_PARITY_MIN_SAMPLES,
    QUANTIZATION_PARITY_SLICES,
    SHA256_RE,
    quantization_parity_evidence_failures,
    sha256,
)


MAX_PREDICTION_BYTES = 250_000_000
MAX_PREDICTION_ROWS = 100_000
MAX_JSONL_LINE_BYTES = 2_000_000
MAX_PROVENANCE_BYTES = 2_000_000
BOOTSTRAP_SEED = 20261005
CANDIDATE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _regular_file(path: Path, *, label: str, maximum_bytes: int) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is missing or not a regular file: {path}")
    size = path.stat().st_size
    if size < 1 or size > maximum_bytes:
        raise ValueError(f"{label} size is outside the valid range")


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _regular_file(path, label=label, maximum_bytes=MAX_PREDICTION_BYTES)
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if len(line.encode("utf-8")) > MAX_JSONL_LINE_BYTES:
                    raise ValueError(f"{label} line {line_number} is too large")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{label} line {line_number} is not an object")
                rows.append(row)
                if len(rows) > MAX_PREDICTION_ROWS:
                    raise ValueError(f"{label} has too many rows")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSONL") from exc
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label, maximum_bytes=MAX_PROVENANCE_BYTES)
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _verify_reference_provenance(
    path: Path,
    *,
    task: str,
    direction: str | None,
    adapter_manifest_sha256: str,
    manifest_path: Path,
    predictions_path: Path,
    prediction_rows: int,
) -> dict[str, Any]:
    provenance = _read_json(path, label="Reference prediction provenance")
    specification = provenance.get("specification")
    prediction = provenance.get("predictions")
    decoding = provenance.get("decoding")
    runtime = provenance.get("runtime")
    if (
        provenance.get("schema_version") != 1
        or not isinstance(specification, dict)
        or provenance.get("specification_sha256")
        != canonical_sha256(specification)
        or not isinstance(prediction, dict)
        or not isinstance(decoding, dict)
        or not isinstance(runtime, dict)
    ):
        raise ValueError("Reference prediction provenance is inconsistent")
    if specification.get("task") != task:
        raise ValueError("Reference prediction provenance task mismatch")
    expected_direction = direction if task == "mt" else None
    if specification.get("mt_direction") != expected_direction:
        raise ValueError("Reference prediction provenance direction mismatch")
    adapter = specification.get("adapter")
    if (
        not isinstance(adapter, dict)
        or adapter.get("manifest_sha256") != adapter_manifest_sha256
    ):
        raise ValueError("Reference prediction provenance adapter mismatch")
    manifest = specification.get("manifest")
    if (
        not isinstance(manifest, dict)
        or manifest.get("bytes") != manifest_path.stat().st_size
        or manifest.get("sha256") != sha256(manifest_path)
    ):
        raise ValueError("Reference prediction provenance manifest mismatch")
    if (
        prediction.get("bytes") != predictions_path.stat().st_size
        or prediction.get("sha256") != sha256(predictions_path)
        or prediction.get("rows") != prediction_rows
    ):
        raise ValueError("Reference prediction provenance prediction mismatch")
    num_beams = specification.get("num_beams")
    if (
        isinstance(num_beams, bool)
        or not isinstance(num_beams, int)
        or num_beams < 1
        or decoding.get("num_beams") != num_beams
    ):
        raise ValueError("Reference prediction provenance decoding mismatch")
    return {"num_beams": num_beams, "do_sample": False}


def _verify_quantized_provenance(
    path: Path,
    *,
    task: str,
    direction: str | None,
    candidate_id: str,
    adapter_manifest_sha256: str,
    decoding: dict[str, Any],
    artifact_path: Path,
    manifest_path: Path,
    predictions_path: Path,
    prediction_rows: int,
) -> None:
    provenance = _read_json(path, label="Quantized prediction provenance")
    expected = {
        "task": task,
        "direction": direction,
        "candidate_id": candidate_id,
        "adapter_manifest_sha256": adapter_manifest_sha256,
        "decoding": decoding,
        "artifact": {
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
        },
        "manifest": {
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256(manifest_path),
        },
        "predictions": {
            "bytes": predictions_path.stat().st_size,
            "sha256": sha256(predictions_path),
            "rows": prediction_rows,
        },
    }
    failures = compiled_prediction_provenance_failures(provenance, expected=expected)
    if failures:
        raise ValueError(
            "Quantized prediction provenance failed: " + ", ".join(failures)
        )


def _row_key(row: dict[str, Any], *, task: str, label: str) -> tuple[str, ...]:
    identifier = str(row.get("id") or "").strip()
    if not identifier:
        raise ValueError(f"{label} row is missing id")
    if task == "mt":
        direction = str(row.get("direction") or "").strip()
        return direction, identifier
    return (identifier,)


def _indexed_rows(
    rows: list[dict[str, Any]],
    *,
    task: str,
    label: str,
) -> dict[tuple[str, ...], dict[str, Any]]:
    indexed: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = _row_key(row, task=task, label=label)
        if key in indexed:
            raise ValueError(f"{label} contains duplicate prediction key: {key}")
        hypothesis = row.get("hypothesis")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError(f"{label} has an empty hypothesis: {key}")
        indexed[key] = row
    return indexed


def _without_hypothesis(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "hypothesis"}


def _validate_prediction_pair(
    reference_rows: list[dict[str, Any]],
    quantized_rows: list[dict[str, Any]],
    *,
    task: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = _indexed_rows(reference_rows, task=task, label="Reference predictions")
    quantized = _indexed_rows(quantized_rows, task=task, label="Quantized predictions")
    if set(reference) != set(quantized):
        raise ValueError("Reference and quantized prediction key sets differ")
    ordered_keys = sorted(reference)
    for key in ordered_keys:
        if _without_hypothesis(reference[key]) != _without_hypothesis(quantized[key]):
            raise ValueError(f"Prediction metadata differs for key: {key}")
    return (
        [reference[key] for key in ordered_keys],
        [quantized[key] for key in ordered_keys],
    )


def _validate_manifest_binding(
    manifest_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    task: str,
    direction: str | None,
) -> None:
    manifest_by_id: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        identifier = str(row.get("id") or "").strip()
        if not identifier or identifier in manifest_by_id:
            raise ValueError("Parity manifest has missing or duplicate ids")
        manifest_by_id[identifier] = row
    prediction_ids = {str(row["id"]).strip() for row in predictions}
    if prediction_ids != set(manifest_by_id) or len(predictions) != len(manifest_rows):
        raise ValueError("Predictions do not cover the exact parity manifest")

    for prediction in predictions:
        manifest = manifest_by_id[str(prediction["id"]).strip()]
        if task == "mt":
            if prediction.get("direction") != direction:
                raise ValueError("MT prediction direction differs from requested parity slot")
            source_field, target_field = (
                ("source_text", "target_text")
                if direction == "en_to_vi"
                else ("target_text", "source_text")
            )
            bindings = {
                "source": manifest.get(source_field),
                "reference": manifest.get(target_field),
                "categories": manifest.get("categories", []),
                "terminology": manifest.get("terminology", {}),
            }
        else:
            bindings = {
                "reference": manifest.get("text"),
                "categories": manifest.get("categories", []),
                "safety_expectations": manifest.get("safety_expectations", {}),
                "speaker": manifest.get("speaker"),
                "group": manifest.get("group"),
            }
        for field, value in bindings.items():
            if prediction.get(field) != value:
                raise ValueError(
                    f"Prediction {prediction['id']} {field} differs from parity manifest"
                )


def _relative_degradation(
    reference: float,
    quantized: float,
    *,
    greater_is_better: bool,
) -> float:
    if reference == 0.0:
        if (greater_is_better and quantized >= 0.0) or quantized == 0.0:
            return 0.0
        raise ValueError("Relative degradation is undefined from a zero reference metric")
    delta = reference - quantized if greater_is_better else quantized - reference
    return float(delta / abs(reference))


def _mt_metric_statistics(
    rows: list[dict[str, Any]],
    metric: BLEU | CHRF,
) -> np.ndarray:
    hypotheses = [row["hypothesis"] for row in rows]
    references = [[row["reference"] for row in rows]]
    return np.asarray(
        metric._extract_corpus_statistics(hypotheses, references),
        dtype=np.int64,
    )


def _paired_mt_bootstrap(
    reference_rows: list[dict[str, Any]],
    quantized_rows: list[dict[str, Any]],
) -> dict[str, list[float]]:
    metrics: dict[str, BLEU | CHRF] = {
        "sacrebleu": BLEU(tokenize="intl"),
        "chrf2": CHRF(),
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample_indices = rng.integers(
        0,
        len(reference_rows),
        size=(QUANTIZATION_PARITY_BOOTSTRAP_REPEATS, len(reference_rows)),
    )
    result: dict[str, list[float]] = {}
    for name, metric in metrics.items():
        reference_stats = _mt_metric_statistics(reference_rows, metric)
        quantized_stats = _mt_metric_statistics(quantized_rows, metric)
        degradations = np.empty(
            QUANTIZATION_PARITY_BOOTSTRAP_REPEATS,
            dtype=np.float64,
        )
        for index, selected in enumerate(sample_indices):
            reference_value = metric._compute_score_from_stats(
                reference_stats[selected].sum(axis=0).tolist()
            ).score
            quantized_value = metric._compute_score_from_stats(
                quantized_stats[selected].sum(axis=0).tolist()
            ).score
            degradations[index] = _relative_degradation(
                float(reference_value),
                float(quantized_value),
                greater_is_better=True,
            )
        result[name] = [
            float(np.quantile(degradations, 0.025)),
            float(np.quantile(degradations, 0.975)),
        ]
    return result


def _error_counts(
    reference: str,
    hypothesis: str,
    *,
    processor: Callable[[str, str], Any],
) -> tuple[float, float]:
    scored = processor(reference, hypothesis)
    return (
        float(scored.substitutions + scored.deletions + scored.insertions),
        float(scored.hits + scored.substitutions + scored.deletions),
    )


def _asr_group_key(rows: list[dict[str, Any]]) -> tuple[str, list[str]]:
    for field in ("group", "speaker"):
        values = [str(row.get(field) or "").strip() for row in rows]
        if all(values) and len(set(values)) >= 32:
            return field, values
    raise ValueError("ASR parity requires at least 32 independent groups or speakers")


def _paired_asr_metric_bootstrap(
    reference_rows: list[dict[str, Any]],
    quantized_rows: list[dict[str, Any]],
    *,
    groups: list[str],
    processor: Callable[[str, str], Any],
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> list[float]:
    counts: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for reference, quantized, group in zip(reference_rows, quantized_rows, groups):
        if predicate is not None and not predicate(reference):
            continue
        ref_errors, ref_words = _error_counts(
            str(reference["reference"]),
            str(reference["hypothesis"]),
            processor=processor,
        )
        quant_errors, quant_words = _error_counts(
            str(quantized["reference"]),
            str(quantized["hypothesis"]),
            processor=processor,
        )
        aggregate = counts[group]
        aggregate[0] += ref_errors
        aggregate[1] += ref_words
        aggregate[2] += quant_errors
        aggregate[3] += quant_words
    if not counts:
        raise ValueError("ASR parity code-switch slice is empty")
    array = np.asarray(list(counts.values()), dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    degradations = np.empty(
        QUANTIZATION_PARITY_BOOTSTRAP_REPEATS,
        dtype=np.float64,
    )
    for index in range(QUANTIZATION_PARITY_BOOTSTRAP_REPEATS):
        sampled = array[rng.integers(0, len(array), len(array))].sum(axis=0)
        reference_value = sampled[0] / max(1.0, sampled[1])
        quantized_value = sampled[2] / max(1.0, sampled[3])
        degradations[index] = _relative_degradation(
            float(reference_value),
            float(quantized_value),
            greater_is_better=False,
        )
    return [
        float(np.quantile(degradations, 0.025)),
        float(np.quantile(degradations, 0.975)),
    ]


def _paired_asr_bootstrap(
    reference_rows: list[dict[str, Any]],
    quantized_rows: list[dict[str, Any]],
) -> tuple[str, int, dict[str, list[float]]]:
    unit, groups = _asr_group_key(reference_rows)
    intervals = {
        "wer": _paired_asr_metric_bootstrap(
            reference_rows,
            quantized_rows,
            groups=groups,
            processor=process_words,
        ),
        "cer": _paired_asr_metric_bootstrap(
            reference_rows,
            quantized_rows,
            groups=groups,
            processor=process_characters,
        ),
        "code_switch_wer": _paired_asr_metric_bootstrap(
            reference_rows,
            quantized_rows,
            groups=groups,
            processor=process_words,
            predicate=lambda row: row.get("code_switch") is True,
        ),
    }
    return unit, len(set(groups)), intervals


def _asr_code_switch_wer(report: dict[str, Any]) -> float:
    values = (report.get("slices") or {}).get("code_switch") or {}
    for key, record in values.items():
        if str(key).casefold() == "true" and isinstance(record, dict):
            return float(record["wer"])
    raise ValueError("ASR parity report lacks a non-empty code-switch slice")


def _metric_values(report: dict[str, Any], *, task: str) -> dict[str, float]:
    if task == "mt":
        return {
            "sacrebleu": float(report["sacrebleu"]),
            "chrf2": float(report["chrf2"]),
        }
    return {
        "wer": float(report["wer"]),
        "cer": float(report["cer"]),
        "code_switch_wer": _asr_code_switch_wer(report),
    }


def _safety_slices(
    reference_report: dict[str, Any],
    quantized_report: dict[str, Any],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name in QUANTIZATION_PARITY_SLICES:
        reference = (reference_report.get("categories") or {}).get(name)
        quantized = (quantized_report.get("categories") or {}).get(name)
        if not isinstance(reference, dict) or not isinstance(quantized, dict):
            raise ValueError(f"Parity predictions lack required safety slice: {name}")
        reference_samples = int(reference.get("samples", 0))
        quantized_samples = int(quantized.get("samples", 0))
        if reference_samples < 1 or reference_samples != quantized_samples:
            raise ValueError(f"Parity safety slice sample mismatch: {name}")
        if "failures" in reference:
            reference_failures = int(reference.get("failures", 0))
            quantized_failures = int(quantized.get("failures", 0))
        else:
            reference_failures = round(
                float(reference.get("safety_failure_rate", 1.0)) * reference_samples
            )
            quantized_failures = round(
                float(quantized.get("safety_failure_rate", 1.0)) * quantized_samples
            )
        result[name] = {
            "samples": reference_samples,
            "reference_failures": reference_failures,
            "quantized_failures": quantized_failures,
        }
    return result


def seal_quantization_parity(
    *,
    task: str,
    direction: str | None,
    candidate_id: str,
    adapter_manifest_sha256: str,
    artifact_path: Path,
    manifest_path: Path,
    reference_predictions_path: Path,
    reference_provenance_path: Path,
    quantized_predictions_path: Path,
    quantized_provenance_path: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Recompute and bind one locked quantization-parity result."""
    if task not in {"mt", "asr"}:
        raise ValueError("task must be mt or asr")
    if task == "mt" and direction not in {"en_to_vi", "vi_to_en"}:
        raise ValueError("MT parity requires en_to_vi or vi_to_en direction")
    if task == "asr" and direction is not None:
        raise ValueError("ASR parity direction must be omitted")
    if not CANDIDATE_RE.fullmatch(str(candidate_id or "")):
        raise ValueError("candidate_id is invalid")
    if not SHA256_RE.fullmatch(str(adapter_manifest_sha256 or "")):
        raise ValueError("adapter_manifest_sha256 is invalid")

    _regular_file(
        artifact_path,
        label="Compiled deployment artifact",
        maximum_bytes=sys.maxsize,
    )
    manifest_rows = _read_jsonl(manifest_path, label="Parity manifest")
    reference_rows = _read_jsonl(
        reference_predictions_path,
        label="Reference predictions",
    )
    quantized_rows = _read_jsonl(
        quantized_predictions_path,
        label="Quantized predictions",
    )
    reference_rows, quantized_rows = _validate_prediction_pair(
        reference_rows,
        quantized_rows,
        task=task,
    )
    if len(reference_rows) < QUANTIZATION_PARITY_MIN_SAMPLES:
        raise ValueError(
            f"Quantization parity requires at least {QUANTIZATION_PARITY_MIN_SAMPLES} samples"
        )
    _validate_manifest_binding(
        manifest_rows,
        reference_rows,
        task=task,
        direction=direction,
    )
    decoding = _verify_reference_provenance(
        reference_provenance_path,
        task=task,
        direction=direction,
        adapter_manifest_sha256=adapter_manifest_sha256,
        manifest_path=manifest_path,
        predictions_path=reference_predictions_path,
        prediction_rows=len(reference_rows),
    )
    _verify_quantized_provenance(
        quantized_provenance_path,
        task=task,
        direction=direction,
        candidate_id=candidate_id,
        adapter_manifest_sha256=adapter_manifest_sha256,
        decoding=decoding,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        predictions_path=quantized_predictions_path,
        prediction_rows=len(quantized_rows),
    )

    reference_report = score_mt(reference_rows) if task == "mt" else score_asr(reference_rows)
    quantized_report = score_mt(quantized_rows) if task == "mt" else score_asr(quantized_rows)
    reference_metrics = _metric_values(reference_report, task=task)
    quantized_metrics = _metric_values(quantized_report, task=task)
    if task == "mt":
        bootstrap_unit = "row"
        bootstrap_clusters = len(reference_rows)
        intervals = _paired_mt_bootstrap(reference_rows, quantized_rows)
    else:
        bootstrap_unit, bootstrap_clusters, intervals = _paired_asr_bootstrap(
            reference_rows,
            quantized_rows,
        )

    metrics: dict[str, dict[str, Any]] = {}
    for name in reference_metrics:
        reference_value = reference_metrics[name]
        quantized_value = quantized_metrics[name]
        metrics[name] = {
            "reference": reference_value,
            "quantized": quantized_value,
            "relative_degradation": _relative_degradation(
                reference_value,
                quantized_value,
                greater_is_better=task == "mt",
            ),
            "relative_degradation_bootstrap_95ci": intervals[name],
            "maximum_relative_degradation": (
                QUANTIZATION_PARITY_MAX_RELATIVE_DEGRADATION
            ),
        }

    evaluated_at = now or datetime.now(timezone.utc)
    if evaluated_at.tzinfo is None:
        raise ValueError("now must include a timezone")
    evidence = {
        "version": 2,
        "status": "pass",
        "evidence_source": "onevoice_quantization_parity",
        "evaluated_at": evaluated_at.isoformat(),
        "task": task,
        "direction": direction,
        "candidate_id": candidate_id,
        "adapter_manifest_sha256": adapter_manifest_sha256,
        "artifact_sha256": sha256(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "manifest_sha256": sha256(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "reference_predictions_sha256": sha256(reference_predictions_path),
        "reference_predictions_bytes": reference_predictions_path.stat().st_size,
        "reference_provenance_sha256": sha256(reference_provenance_path),
        "reference_provenance_bytes": reference_provenance_path.stat().st_size,
        "quantized_predictions_sha256": sha256(quantized_predictions_path),
        "quantized_predictions_bytes": quantized_predictions_path.stat().st_size,
        "quantized_provenance_sha256": sha256(quantized_provenance_path),
        "quantized_provenance_bytes": quantized_provenance_path.stat().st_size,
        "decoding": decoding,
        "samples": len(reference_rows),
        "bootstrap": {
            "unit": bootstrap_unit,
            "clusters": bootstrap_clusters,
            "repeats": QUANTIZATION_PARITY_BOOTSTRAP_REPEATS,
            "seed": BOOTSTRAP_SEED,
        },
        "metrics": metrics,
        "safety_slices": _safety_slices(reference_report, quantized_report),
    }
    failures = quantization_parity_evidence_failures(evidence)
    if failures:
        raise ValueError("Quantization parity failed: " + ", ".join(failures))
    return evidence


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite parity evidence: {path}")
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if len(serialized.encode("utf-8")) > MAX_QUANTIZATION_PARITY_EVIDENCE_BYTES:
        raise ValueError("Quantization parity evidence is too large")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["mt", "asr"], required=True)
    parser.add_argument("--direction", choices=["en_to_vi", "vi_to_en"])
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--adapter-manifest-sha256", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--reference-provenance", type=Path, required=True)
    parser.add_argument("--quantized-predictions", type=Path, required=True)
    parser.add_argument("--quantized-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = seal_quantization_parity(
        task=args.task,
        direction=args.direction,
        candidate_id=args.candidate_id,
        adapter_manifest_sha256=args.adapter_manifest_sha256,
        artifact_path=args.artifact,
        manifest_path=args.manifest,
        reference_predictions_path=args.reference_predictions,
        reference_provenance_path=args.reference_provenance,
        quantized_predictions_path=args.quantized_predictions,
        quantized_provenance_path=args.quantized_provenance,
    )
    write_exclusive(args.output, evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

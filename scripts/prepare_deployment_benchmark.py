"""Prepare and finalize physical-QCS6490 evidence for selected winners.

The template action binds a measurement draft to the exact selected adapters.
After board measurements, quantization-parity evidence and compiled artifacts
have been copied back, the finalize action computes immutable artifact
identities plus latency/power/thermal metrics, validates the complete report,
and writes it atomically.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_model_bakeoff import (  # noqa: E402
    MAX_DEPLOYMENT_MEASUREMENT_BYTES,
    MAX_QUANTIZATION_PARITY_EVIDENCE_BYTES,
    QUANTIZATION_PARITY_BOOTSTRAP_REPEATS,
    QUANTIZATION_PARITY_MAX_RELATIVE_DEGRADATION,
    QUANTIZATION_PARITY_MIN_SAMPLES,
    QUANTIZATION_PARITY_SLICES,
    deployment_required_metrics,
    deployment_expectations,
    load_config,
    percentile_linear,
    quantization_parity_evidence_failures,
    sha256,
    validate_deployment_report,
)


DEFAULT_CONFIG = ROOT / "configs/model_bakeoff.yaml"
DEFAULT_DRAFT = ROOT / "data/reports/model_bakeoff/deployment_selected_winners.draft.json"
DEFAULT_OUTPUT = ROOT / "data/reports/model_bakeoff/deployment_selected_winners.json"


def atomic_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Create immutable evidence without overwriting an existing draft/report."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite deployment evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def selected_winner_specs(comparison: dict[str, Any]) -> list[tuple[str, str | None, dict[str, Any]]]:
    if comparison.get("status") not in {"blind_complete", "complete"}:
        raise ValueError("Deployment evidence requires completed blind evaluation")
    results = comparison.get("results")
    if not isinstance(results, dict):
        raise ValueError("Selection comparison has no results")
    mt_winners = (results.get("mt") or {}).get("winners")
    asr_winners = (results.get("asr") or {}).get("winners")
    if not isinstance(mt_winners, dict) or not isinstance(asr_winners, dict):
        raise ValueError("Selection comparison has no winner bindings")
    required_mt = {"en_to_vi", "vi_to_en"}
    if set(mt_winners) != required_mt or set(asr_winners) != {"vi"}:
        raise ValueError("Selection comparison must contain both MT directions and ASR vi")
    specs = [
        ("mt", direction, mt_winners[direction])
        for direction in ("en_to_vi", "vi_to_en")
    ] + [("asr", None, asr_winners["vi"])]
    for task, direction, winner in specs:
        if not isinstance(winner, dict):
            raise ValueError(f"Invalid winner record: {task}/{direction or 'vi'}")
        if not str(winner.get("candidate_id") or "").strip():
            raise ValueError(f"Winner candidate is missing: {task}/{direction or 'vi'}")
        if not str(winner.get("adapter") or "").strip():
            raise ValueError(f"Winner adapter is missing: {task}/{direction or 'vi'}")
    return specs


def build_template(
    selection_path: Path,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    expected = deployment_expectations(selected_winner_specs(comparison))
    return {
        "version": 1,
        "status": "pending_physical_measurement",
        "target": "QCS6490",
        "measurement_source": "physical_board",
        "measured_at": None,
        "selection_comparison": {
            "path": str(selection_path.resolve()),
            "sha256": sha256(selection_path),
        },
        "device": {
            "chipset": "QCS6490",
            "board": "",
            "os": "",
            "identity_evidence_path": "",
            "identity_evidence_sha256": "",
        },
        "winners": [
            {
                **winner,
                "artifact_path": "",
                "parity_evidence_path": "",
                "measurement_evidence_path": "",
                "latency_samples_ms": [],
                "power_sensor": "",
                "power_samples_mw": [],
                "temperature_sensor": "",
                "temperature_samples_c": [],
                "peak_ram_bytes": None,
                "peak_vram_bytes": None,
            }
            for winner in expected
        ],
    }


def _winner_key(record: dict[str, Any]) -> tuple[str, str | None]:
    direction = record.get("direction")
    return str(record.get("task") or ""), str(direction) if direction is not None else None


def _resolve_artifact(path_value: Any, project_root: Path) -> Path:
    value = str(path_value or "").strip()
    if not value:
        raise ValueError("Deployment artifact path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _load_measurement_evidence(
    record: dict[str, Any],
    expected_record: dict[str, Any],
    *,
    identity_sha256: str,
    artifact: Path,
    project_root: Path,
) -> tuple[Path, dict[str, Any]]:
    value = str(record.get("measurement_evidence_path") or "").strip()
    if not value:
        raise ValueError("Physical measurement evidence path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    evidence_root = (
        project_root
        / "data/reports/model_bakeoff/board-evidence/measurements"
    ).resolve()
    if evidence_root not in path.parents:
        raise ValueError("Physical measurement evidence must be under board-evidence/measurements")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Physical measurement evidence is missing: {path}")
    if path.stat().st_size > MAX_DEPLOYMENT_MEASUREMENT_BYTES:
        raise ValueError("Physical measurement evidence is too large")
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Physical measurement evidence is not valid JSON") from exc
    if not isinstance(evidence, dict):
        raise ValueError("Physical measurement evidence payload is invalid")
    if evidence.get("version") != 1:
        raise ValueError("Physical measurement evidence version is invalid")
    if evidence.get("capture_source") != "physical_qcs6490":
        raise ValueError("Physical measurement evidence source is invalid")
    for field in ("task", "direction", "candidate_id", "adapter_manifest_sha256"):
        if evidence.get(field) != expected_record.get(field):
            raise ValueError(f"Physical measurement evidence {field} binding changed")
    if evidence.get("identity_evidence_sha256") != identity_sha256:
        raise ValueError("Physical measurement evidence identity binding changed")
    artifact_sha = sha256(artifact)
    if evidence.get("artifact_sha256") != artifact_sha:
        raise ValueError("Physical measurement evidence artifact checksum changed")
    if evidence.get("artifact_bytes") != artifact.stat().st_size:
        raise ValueError("Physical measurement evidence artifact size changed")
    captured_at = str(evidence.get("captured_at") or "")
    try:
        captured_time = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        if captured_time.tzinfo is None:
            raise ValueError("timezone required")
    except ValueError as exc:
        raise ValueError("Physical measurement evidence timestamp is invalid") from exc
    return path, evidence


def _load_parity_evidence(
    record: dict[str, Any],
    expected_record: dict[str, Any],
    *,
    artifact: Path,
    project_root: Path,
) -> tuple[Path, dict[str, Any]]:
    value = str(record.get("parity_evidence_path") or "").strip()
    if not value:
        raise ValueError("Quantization parity evidence path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    evidence_root = (
        project_root / "data/reports/model_bakeoff/board-evidence/parity"
    ).resolve()
    if evidence_root not in path.parents:
        raise ValueError("Quantization parity evidence must be under board-evidence/parity")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Quantization parity evidence is missing: {path}")
    if path.stat().st_size > MAX_QUANTIZATION_PARITY_EVIDENCE_BYTES:
        raise ValueError("Quantization parity evidence is too large")
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Quantization parity evidence is not valid JSON") from exc
    failures = quantization_parity_evidence_failures(
        evidence,
        expected=expected_record,
        artifact_sha256=sha256(artifact),
        artifact_bytes=artifact.stat().st_size,
    )
    if failures:
        raise ValueError("Quantization parity evidence failed: " + ", ".join(failures))
    return path, evidence


def _parse_samples(
    record: dict[str, Any],
    key: str,
    *,
    minimum: float | None = None,
) -> list[float]:
    values = record.get(key)
    if not isinstance(values, list):
        raise ValueError(f"Deployment {key} are missing")
    parsed: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid deployment {key}") from exc
        if not math.isfinite(number) or (
            minimum is not None and number <= minimum
        ):
            raise ValueError(f"Invalid deployment {key}")
        parsed.append(number)
    if not parsed:
        raise ValueError(f"Deployment {key} are empty")
    return parsed


def finalize_report(
    selection_path: Path,
    comparison: dict[str, Any],
    draft: dict[str, Any],
    config: dict[str, Any],
    project_root: Path = ROOT,
) -> dict[str, Any]:
    expected = deployment_expectations(selected_winner_specs(comparison))
    records = draft.get("winners")
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise ValueError("Deployment draft winners are missing or invalid")
    record_by_key: dict[tuple[str, str | None], dict[str, Any]] = {}
    for record in records:
        key = _winner_key(record)
        if key in record_by_key:
            raise ValueError(f"Duplicate deployment winner: {key}")
        record_by_key[key] = record
    expected_by_key = {_winner_key(item): item for item in expected}
    if set(record_by_key) != set(expected_by_key):
        raise ValueError("Deployment draft winner set does not match selection")

    final = deepcopy(draft)
    final["version"] = 1
    final["status"] = "pass"
    final["target"] = "QCS6490"
    final["measurement_source"] = "physical_board"
    final["selection_comparison"] = {
        "path": str(selection_path.resolve()),
        "sha256": sha256(selection_path),
    }
    device = final.get("device")
    if not isinstance(device, dict):
        raise ValueError("Deployment device metadata is missing")
    identity_value = str(device.get("identity_evidence_path") or "").strip()
    if not identity_value:
        raise ValueError("QCS6490 identity evidence path is missing")
    identity_path = _resolve_artifact(identity_value, project_root)
    identity_root = (
        project_root / "data" / "reports" / "model_bakeoff" / "board-evidence"
    ).resolve()
    if identity_root not in identity_path.parents:
        raise ValueError("QCS6490 identity evidence must be under board-evidence")
    if not identity_path.is_file():
        raise FileNotFoundError(f"QCS6490 identity evidence is missing: {identity_path}")
    try:
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("QCS6490 identity evidence is not valid JSON") from exc
    if not isinstance(identity, dict):
        raise ValueError("QCS6490 identity evidence payload is invalid")
    device["identity_evidence_sha256"] = sha256(identity_path)
    device["board"] = str(identity.get("board_model") or "").strip()
    device["architecture"] = str(identity.get("architecture") or "").strip()
    final_records = {_winner_key(item): item for item in final["winners"]}
    for key, expected_record in expected_by_key.items():
        record = final_records[key]
        if record.get("candidate_id") != expected_record["candidate_id"]:
            raise ValueError(f"Deployment candidate binding changed: {key}")
        if record.get("adapter_manifest_sha256") != expected_record["adapter_manifest_sha256"]:
            raise ValueError(f"Deployment adapter binding changed: {key}")
        artifact = _resolve_artifact(record.get("artifact_path"), project_root)
        if not artifact.is_file():
            raise FileNotFoundError(f"Deployment artifact is missing: {artifact}")
        parity_path, parity = _load_parity_evidence(
            record,
            expected_record,
            artifact=artifact,
            project_root=project_root,
        )
        measurement_path, measurement = _load_measurement_evidence(
            record,
            expected_record,
            identity_sha256=device["identity_evidence_sha256"],
            artifact=artifact,
            project_root=project_root,
        )
        for field in (
            "latency_samples_ms",
            "power_sensor",
            "power_samples_mw",
            "temperature_sensor",
            "temperature_samples_c",
            "peak_ram_bytes",
            "peak_vram_bytes",
        ):
            record[field] = deepcopy(measurement.get(field))
        parsed_samples = _parse_samples(
            record,
            "latency_samples_ms",
            minimum=0.0,
        )
        power_samples = _parse_samples(
            record,
            "power_samples_mw",
            minimum=0.0,
        )
        temperature_samples = _parse_samples(
            record,
            "temperature_samples_c",
            minimum=-273.15,
        )
        record.update(
            {
                "parity_evidence_path": str(parity_path),
                "parity_evidence_sha256": sha256(parity_path),
                "parity_evaluated_at": parity["evaluated_at"],
                "parity_manifest_sha256": parity["manifest_sha256"],
                "parity_reference_predictions_sha256": parity[
                    "reference_predictions_sha256"
                ],
                "parity_quantized_predictions_sha256": parity[
                    "quantized_predictions_sha256"
                ],
                "parity_samples": parity["samples"],
                "parity_bootstrap": deepcopy(parity["bootstrap"]),
                "parity_metrics": deepcopy(parity["metrics"]),
                "parity_safety_slices": deepcopy(parity["safety_slices"]),
                "measurement_evidence_path": str(measurement_path),
                "measurement_evidence_sha256": sha256(measurement_path),
                "measurement_captured_at": measurement["captured_at"],
                "measurement_runs": len(parsed_samples),
                "latency_samples_ms": parsed_samples,
                "latency_p50_ms": percentile_linear(parsed_samples, 0.50),
                "latency_p95_ms": percentile_linear(parsed_samples, 0.95),
                "power_samples_mw": power_samples,
                "power_avg_mw": sum(power_samples) / len(power_samples),
                "power_p95_mw": percentile_linear(power_samples, 0.95),
                "temperature_samples_c": temperature_samples,
                "temperature_peak_c": max(temperature_samples),
                "artifact_sha256": sha256(artifact),
                "model_bytes": artifact.stat().st_size,
            }
        )

    gate = config["promotion_gate"]
    required_metrics = deployment_required_metrics(
        list(gate["deployment_metrics"])
    )
    final["deployment_gate"] = {
        "required_metrics": required_metrics,
        "minimum_measurement_runs": int(gate["deployment_min_runs"]),
        "quantization_parity": {
            "maximum_relative_degradation": (
                QUANTIZATION_PARITY_MAX_RELATIVE_DEGRADATION
            ),
            "minimum_samples": QUANTIZATION_PARITY_MIN_SAMPLES,
            "minimum_bootstrap_repeats": QUANTIZATION_PARITY_BOOTSTRAP_REPEATS,
            "asr_minimum_independent_groups": 32,
            "required_zero_failure_slices": list(QUANTIZATION_PARITY_SLICES),
        },
    }
    passed, failures = validate_deployment_report(
        final,
        expected,
        required_metrics,
        int(gate["deployment_min_runs"]),
        project_root,
    )
    if not passed:
        raise ValueError("Deployment report failed validation: " + ", ".join(failures))
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["template", "finalize"], required=True)
    parser.add_argument("--selection-comparison", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.selection_comparison.is_file():
        raise FileNotFoundError(args.selection_comparison)
    comparison = json.loads(args.selection_comparison.read_text(encoding="utf-8"))
    config = load_config(args.config)
    if args.action == "template":
        template = build_template(args.selection_comparison, comparison)
        atomic_json_exclusive(args.draft, template)
        print(json.dumps(template, ensure_ascii=False, indent=2))
        return 0

    if not args.draft.is_file():
        raise FileNotFoundError(args.draft)
    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    report = finalize_report(args.selection_comparison, comparison, draft, config)
    atomic_json_exclusive(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

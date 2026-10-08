"""Seal raw deployment telemetry on a verified physical QCS6490 board.

The input is a bounded JSON capture produced by the board benchmark harness.
This command validates the live device identity, sample contract and artifact
binding, then creates the immutable measurement evidence consumed by
``prepare_deployment_benchmark.py --action finalize``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.capture_qcs6490_identity import capture_identity
from scripts.run_model_bakeoff import (
    MAX_DEPLOYMENT_ARTIFACT_BYTES,
    MAX_DEPLOYMENT_MEASUREMENT_BYTES,
    MAX_IDENTITY_EVIDENCE_BYTES,
    SHA256_RE,
    qcs6490_identity_failures,
    read_json_mapping,
)
from src.pipeline.durable_json import write_durable_json_exclusive  # noqa: E402
from src.pipeline.evidence_paths import (  # noqa: E402
    is_link_or_junction,
    resolve_regular_file,
)
from src.utils.bounded_file import sha256_stable_regular_file  # noqa: E402


MIN_MEASUREMENT_RUNS = 30
RAW_FIELDS = {
    "version",
    "capture_source",
    "captured_at",
    "latency_samples_ms",
    "power_sensor",
    "power_samples_mw",
    "temperature_sensor",
    "temperature_samples_c",
    "peak_ram_bytes",
    "peak_vram_bytes",
}
STABLE_IDENTITY_FIELDS = (
    "architecture",
    "board_model",
    "device_tree_compatible",
    "soc_family",
    "soc_id",
)


def _resolve_regular_input(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.parts[0])
    for part in absolute.parts[1:]:
        current /= part
        if is_link_or_junction(current):
            raise ValueError(f"{label} cannot traverse a symlink or junction")
    return resolve_regular_file(
        absolute,
        label=label,
        maximum_bytes=maximum_bytes,
    )


def _load_json(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> tuple[Path, dict[str, Any], str, int]:
    resolved = _resolve_regular_input(
        path,
        maximum_bytes=maximum_bytes,
        label=label,
    )
    digest, size = sha256_stable_regular_file(
        resolved,
        maximum_bytes=maximum_bytes,
        label=label,
    )
    payload = read_json_mapping(
        resolved,
        maximum_bytes=maximum_bytes,
        label=label,
        expected_sha256=digest,
    )
    return resolved, payload, digest, size


def _parse_timestamp(value: Any, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _parse_samples(
    payload: dict[str, Any],
    key: str,
    *,
    minimum_exclusive: float,
) -> list[float]:
    values = payload.get(key)
    if not isinstance(values, list):
        raise ValueError(f"{key} must be an array")
    if len(values) < MIN_MEASUREMENT_RUNS:
        raise ValueError(
            f"{key} requires at least {MIN_MEASUREMENT_RUNS} samples"
        )
    parsed: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{key} contains an invalid sample")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} contains an invalid sample") from exc
        if not math.isfinite(number) or number <= minimum_exclusive:
            raise ValueError(f"{key} contains an invalid sample")
        parsed.append(number)
    return parsed


def _positive_integer(value: Any, *, label: str, allow_zero: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    invalid = value < 0 if allow_zero else value <= 0
    if invalid:
        raise ValueError(f"{label} is outside the valid range")
    return value


def seal_measurement(
    *,
    identity_path: Path,
    artifact_path: Path,
    raw_measurement_path: Path,
    task: str,
    direction: str | None,
    candidate_id: str,
    adapter_manifest_sha256: str,
    system_root: Path = Path("/"),
    architecture: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate and bind one raw board measurement to one selected winner."""
    if task not in {"mt", "asr"}:
        raise ValueError("task must be mt or asr")
    if task == "mt" and direction not in {"en_to_vi", "vi_to_en"}:
        raise ValueError("MT measurement requires en_to_vi or vi_to_en direction")
    if task == "asr" and direction is not None:
        raise ValueError("ASR measurement direction must be omitted")
    candidate_id = str(candidate_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", candidate_id):
        raise ValueError("candidate_id is invalid")
    adapter_manifest_sha256 = str(adapter_manifest_sha256 or "")
    if not SHA256_RE.fullmatch(adapter_manifest_sha256):
        raise ValueError("adapter_manifest_sha256 is invalid")

    identity_path, identity, identity_sha256, _ = _load_json(
        identity_path,
        maximum_bytes=MAX_IDENTITY_EVIDENCE_BYTES,
        label="QCS6490 identity evidence",
    )
    identity_failures = qcs6490_identity_failures(identity)
    if identity_failures:
        raise ValueError(
            "Identity evidence is not verified QCS6490: "
            + ", ".join(identity_failures)
        )
    live_identity = capture_identity(
        system_root,
        architecture=architecture,
    )
    for field in STABLE_IDENTITY_FIELDS:
        if live_identity.get(field) != identity.get(field):
            raise ValueError(f"Live board identity differs from evidence: {field}")

    artifact_path = _resolve_regular_input(
        artifact_path,
        maximum_bytes=MAX_DEPLOYMENT_ARTIFACT_BYTES,
        label="Compiled deployment artifact",
    )
    artifact_sha256, artifact_bytes = sha256_stable_regular_file(
        artifact_path,
        maximum_bytes=MAX_DEPLOYMENT_ARTIFACT_BYTES,
        label="Compiled deployment artifact",
    )
    raw_measurement_path, raw, raw_sha256, _ = _load_json(
        raw_measurement_path,
        maximum_bytes=MAX_DEPLOYMENT_MEASUREMENT_BYTES,
        label="Raw QCS6490 measurement",
    )
    unknown = set(raw) - RAW_FIELDS
    missing = RAW_FIELDS - set(raw)
    if unknown:
        raise ValueError(f"Raw measurement has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"Raw measurement is missing fields: {sorted(missing)}")
    if raw["version"] != 1:
        raise ValueError("Raw measurement version must be 1")
    if raw["capture_source"] != "qcs6490_runtime_sampler":
        raise ValueError("Raw measurement capture_source is invalid")

    captured_time = _parse_timestamp(
        raw["captured_at"],
        label="Raw measurement captured_at",
    )
    identity_time = _parse_timestamp(
        identity.get("captured_at"),
        label="Identity captured_at",
    )
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ValueError("now must include a timezone")
    if abs((captured_time - identity_time).total_seconds()) > 86_400:
        raise ValueError("Raw measurement and identity are not from the same session")
    if abs((current_time - captured_time).total_seconds()) > 86_400:
        raise ValueError("Raw measurement is stale or from the future")

    latency_samples = _parse_samples(
        raw,
        "latency_samples_ms",
        minimum_exclusive=0.0,
    )
    power_samples = _parse_samples(
        raw,
        "power_samples_mw",
        minimum_exclusive=0.0,
    )
    temperature_samples = _parse_samples(
        raw,
        "temperature_samples_c",
        minimum_exclusive=-273.15,
    )
    power_sensor = raw["power_sensor"]
    temperature_sensor = raw["temperature_sensor"]
    for value, label in (
        (power_sensor, "power_sensor"),
        (temperature_sensor, "temperature_sensor"),
    ):
        if not isinstance(value, str) or not value.strip() or len(value) > 512:
            raise ValueError(f"{label} must be a non-empty bounded string")

    return {
        "version": 1,
        "capture_source": "physical_qcs6490",
        "captured_at": captured_time.isoformat(),
        "task": task,
        "direction": direction,
        "candidate_id": candidate_id,
        "adapter_manifest_sha256": adapter_manifest_sha256,
        "identity_evidence_sha256": identity_sha256,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": artifact_bytes,
        "raw_measurement_sha256": raw_sha256,
        "latency_samples_ms": latency_samples,
        "power_sensor": power_sensor.strip(),
        "power_samples_mw": power_samples,
        "temperature_sensor": temperature_sensor.strip(),
        "temperature_samples_c": temperature_samples,
        "peak_ram_bytes": _positive_integer(
            raw["peak_ram_bytes"],
            label="peak_ram_bytes",
            allow_zero=False,
        ),
        "peak_vram_bytes": _positive_integer(
            raw["peak_vram_bytes"],
            label="peak_vram_bytes",
            allow_zero=True,
        ),
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    try:
        write_durable_json_exclusive(
            path,
            payload,
            maximum_bytes=MAX_DEPLOYMENT_MEASUREMENT_BYTES,
            label="QCS6490 measurement evidence",
        )
    except FileExistsError:
        raise FileExistsError(
            f"Refusing to overwrite measurement evidence: {path}"
        ) from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-evidence", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--raw-measurement", type=Path, required=True)
    parser.add_argument("--task", choices=["mt", "asr"], required=True)
    parser.add_argument("--direction", choices=["en_to_vi", "vi_to_en"])
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--adapter-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = seal_measurement(
        identity_path=args.identity_evidence,
        artifact_path=args.artifact,
        raw_measurement_path=args.raw_measurement,
        task=args.task,
        direction=args.direction,
        candidate_id=args.candidate_id,
        adapter_manifest_sha256=args.adapter_manifest_sha256,
    )
    write_exclusive(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Capture reproducible Qualcomm AI Hub environment and profile evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DEVICE = "Dragonwing RB3 Gen 2 Vision Kit"
DEFAULT_DEVICE_OS = "1.6"
DEFAULT_QAIRT_VERSION = "2.50"
SUPPORTED_INPUT_DTYPES = {
    "float32",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
}


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projected_autoregressive_latency(
    encoder_ms: float,
    decoder_ms_per_token: float,
    token_counts: Iterable[int],
) -> dict[str, float]:
    projections: dict[str, float] = {}
    for token_count in token_counts:
        if token_count < 1:
            raise ValueError("token counts must be at least 1")
        projections[str(token_count)] = round(
            encoder_ms + decoder_ms_per_token * token_count,
            3,
        )
    return projections


def capture_environment(device_name: str, device_os: str) -> dict[str, Any]:
    import qai_hub as hub

    devices = hub.get_devices(name=device_name, os=device_os)
    if not devices:
        raise RuntimeError(f"AI Hub device unavailable: {device_name} / {device_os}")
    frameworks = hub.get_frameworks()
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "packages": {
            "qai-hub": version("qai-hub"),
            "qai-hub-models-cli": version("qai-hub-models-cli"),
        },
        "device": vars(devices[0]),
        "frameworks": [vars(item) for item in frameworks],
    }


def capture_reference(
    artifact_dir: str | Path,
    encoder_ms: float,
    decoder_ms: float,
    token_counts: Iterable[int],
) -> dict[str, Any]:
    root = Path(artifact_dir)
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    files = []
    for path in sorted(root.iterdir()):
        if path.is_file():
            files.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Qualcomm AI Hub Models catalog v0.63.0",
        "model_id": metadata["model_id"],
        "runtime": metadata["runtime"],
        "precision": metadata["precision"],
        "chipset": metadata["chipset_attributes"]["name"],
        "device": metadata["chipset_attributes"]["reference_device"],
        "tool_versions": metadata["tool_versions"],
        "performance_ms": {
            "encoder": encoder_ms,
            "decoder_per_token": decoder_ms,
            "projected_by_output_tokens": projected_autoregressive_latency(
                encoder_ms,
                decoder_ms,
                token_counts,
            ),
        },
        "numerics": {
            "available_for_device_filter": False,
            "note": "The catalog returned no QCS6490 numerics for this filter.",
        },
        "files": files,
    }


def summarize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Keep decision-grade metrics without persisting thousands of layer rows."""
    execution_summary = dict(profile.get("execution_summary", {}))
    inference_times = execution_summary.pop("all_inference_times", [])
    load_times = {
        "first": execution_summary.pop("all_first_load_times", []),
        "warm": execution_summary.pop("all_warm_load_times", []),
    }
    def distribution_us(values: list[int]) -> dict[str, float | int]:
        ordered = sorted(values)
        if not ordered:
            return {"samples": 0}
        percentile_95 = ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        return {
            "samples": len(ordered),
            "min_ms": round(ordered[0] / 1000, 3),
            "p50_ms": round(median / 1000, 3),
            "p95_ms": round(percentile_95 / 1000, 3),
            "max_ms": round(ordered[-1] / 1000, 3),
        }

    compute_units: dict[str, int] = {}
    for operation in profile.get("execution_detail", []):
        unit = operation.get("compute_unit", "UNKNOWN")
        compute_units[unit] = compute_units.get(unit, 0) + 1
    return {
        "execution_summary": execution_summary,
        "latency_distribution": {
            "inference": distribution_us(inference_times),
            "first_load": distribution_us(load_times["first"]),
            "warm_load": distribution_us(load_times["warm"]),
        },
        "operation_count_by_compute_unit": compute_units,
    }


def describe_model(model: Any) -> dict[str, Any]:
    return {
        "model_id": model.model_id,
        "name": model.name,
        "model_type": str(model.model_type),
    }


def model_input_specs(model: Any) -> dict[str, tuple[tuple[int, ...], str]]:
    specs: dict[str, tuple[tuple[int, ...], str]] = {}
    for graph_specs in model.input_spec.values():
        for spec in graph_specs:
            specs[spec.name] = (tuple(spec.shape), spec.dtype)
    return specs


def load_input_specs(path: str | Path) -> dict[str, tuple[tuple[int, ...], str]]:
    """Load and validate AI Hub input specs from a small JSON mapping."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("input specs must be a non-empty JSON object")
    specs: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError("input spec names must be non-empty strings")
        if not isinstance(value, dict):
            raise ValueError(f"input spec for {name} must be an object")
        shape = value.get("shape")
        dtype = value.get("dtype")
        if (
            not isinstance(shape, list)
            or not shape
            or any(not isinstance(dimension, int) or dimension < 1 for dimension in shape)
        ):
            raise ValueError(f"input spec for {name} has invalid shape: {shape}")
        if dtype not in SUPPORTED_INPUT_DTYPES:
            raise ValueError(f"input spec for {name} has unsupported dtype: {dtype}")
        specs[name] = (tuple(shape), dtype)
    return specs


def capture_job(
    job_id: str,
    wait: bool,
    timeout: int,
    include_details: bool = False,
) -> dict[str, Any]:
    import qai_hub as hub

    job = hub.get_job(job_id)
    if wait:
        job.wait(timeout=timeout)
    status = job.get_status()
    payload: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "job_type": type(job).__name__,
        "url": job.url,
        "status": {"code": status.code, "message": status.message},
        "name": job.name,
        "options": job.options,
        "device": vars(job.device) if getattr(job, "device", None) else None,
    }
    source_model = getattr(job, "model", None)
    source_models = getattr(job, "models", None)
    if source_model is not None:
        payload["model"] = describe_model(source_model)
    elif source_models is not None:
        payload["models"] = [describe_model(model) for model in source_models]
    if isinstance(job, hub.ProfileJob) and status.code == "SUCCESS":
        profile = job.download_profile()
        payload["profile"] = profile if include_details else summarize_profile(profile)
    if status.code == "SUCCESS" and hasattr(job, "get_target_model"):
        target_model = job.get_target_model()
        if target_model is not None:
            payload["target_model"] = describe_model(target_model)
    return payload


def build_profile_options(qairt_version: str, compute_unit: str) -> str:
    """Build the opaque QAIRT profile options accepted by AI Hub."""
    if not qairt_version.strip():
        raise ValueError("QAIRT version must not be empty")
    if compute_unit not in {"cpu", "gpu", "npu"}:
        raise ValueError(f"unsupported compute unit: {compute_unit}")
    return f" --qairt_version {qairt_version} --compute_unit {compute_unit}"


def submit_profile(
    model_path: str | Path | None,
    model_id: str | None,
    device_name: str,
    device_os: str,
    name: str,
    qairt_version: str,
    compute_unit: str,
    use_runtime_defaults: bool = False,
) -> dict[str, Any]:
    """Upload a target artifact and submit a reproducible profile job."""
    import qai_hub as hub

    if (model_path is None) == (model_id is None):
        raise ValueError("provide exactly one of model_path or model_id")
    if model_path is not None:
        source = Path(model_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        model: Any = source
        model_evidence = {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        }
    else:
        model = hub.get_model(str(model_id))
        model_evidence = {
            "model_id": model.model_id,
            "name": model.name,
            "model_type": str(model.model_type),
        }
    options = "" if use_runtime_defaults else build_profile_options(qairt_version, compute_unit)
    job = hub.submit_profile_job(
        model=model,
        device=hub.Device(name=device_name, os=device_os),
        name=name,
        options=options,
    )
    if isinstance(job, list):
        if len(job) != 1:
            raise RuntimeError(f"expected one profile job, received {len(job)}")
        job = job[0]
    return {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job.job_id,
        "url": job.url,
        "name": name,
        "model": model_evidence,
        "target": {"device": device_name, "device_os": device_os},
        "profile": {
            "options": options,
            "qairt_version": None if use_runtime_defaults else qairt_version,
            "compute_unit": None if use_runtime_defaults else compute_unit,
        },
    }


def submit_compile(
    model_path: str | Path | None,
    model_id: str | None,
    device_name: str,
    device_os: str,
    name: str,
    target_runtime: str,
    extra_options: str,
    input_specs_path: str | Path | None,
) -> dict[str, Any]:
    import qai_hub as hub

    if (model_path is None) == (model_id is None):
        raise ValueError("provide exactly one of model_path or model_id")
    if model_path is not None:
        source = Path(model_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        model: Any = source
        model_evidence = {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        }
    else:
        model = hub.get_model(str(model_id))
        model_evidence = describe_model(model)
    input_specs = load_input_specs(input_specs_path) if input_specs_path else None
    options = f"--target_runtime {target_runtime} {extra_options}".strip()
    job = hub.submit_compile_job(
        model=model,
        device=hub.Device(name=device_name, os=device_os),
        name=name,
        input_specs=input_specs,
        options=options,
    )
    return {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job.job_id,
        "url": job.url,
        "name": name,
        "model": model_evidence,
        "target": {"device": device_name, "device_os": device_os},
        "compile": {
            "target_runtime": target_runtime,
            "options": options,
            "input_specs": {
                key: {"shape": list(shape), "dtype": dtype}
                for key, (shape, dtype) in (input_specs or {}).items()
            },
        },
    }


def submit_link(
    model_ids: list[str],
    device_name: str,
    device_os: str,
    name: str,
    extra_options: str,
) -> dict[str, Any]:
    """Link one or more QNN DLC models into a device context binary."""
    import qai_hub as hub

    if not model_ids:
        raise ValueError("at least one QNN DLC model ID is required")
    models = [hub.get_model(model_id) for model_id in model_ids]
    job = hub.submit_link_job(
        models=models,
        device=hub.Device(name=device_name, os=device_os),
        name=name,
        options=extra_options.strip(),
    )
    if isinstance(job, list):
        if len(job) != 1:
            raise RuntimeError(f"expected one link job, received {len(job)}")
        job = job[0]
    return {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job.job_id,
        "url": job.url,
        "name": name,
        "models": [describe_model(model) for model in models],
        "target": {"device": device_name, "device_os": device_os},
        "link": {"options": extra_options.strip()},
    }


def submit_random_quantize(
    model_id: str,
    input_name: str | None,
    input_shape: tuple[int, ...] | None,
    input_specs_job_id: str | None,
    sample_count: int,
    seed: int,
    name: str,
    activations_dtype: str,
) -> dict[str, Any]:
    """Submit a latency-only cloud quantization smoke with synthetic inputs."""
    import qai_hub as hub

    if sample_count < 1 or sample_count > 10:
        raise ValueError("random smoke sample count must be between 1 and 10")
    if input_specs_job_id:
        if input_name is not None or input_shape is not None:
            raise ValueError("input_specs_job_id cannot be combined with manual input specs")
        specs_job = hub.get_job(input_specs_job_id)
        input_specs = specs_job.shapes
        if not input_specs and hasattr(specs_job, "get_target_model"):
            target_model = specs_job.get_target_model()
            if target_model is not None:
                input_specs = model_input_specs(target_model)
        if not input_specs:
            raise ValueError(f"no input specs found for job {input_specs_job_id}")
        specs_source = {"job_id": input_specs_job_id}
    else:
        if input_name is None or input_shape is None:
            raise ValueError("manual calibration requires input_name and input_shape")
        input_specs = {input_name: (input_shape, "float32")}
        specs_source = {"manual": True}
    activation_type = {
        "int8": hub.QuantizeDtype.INT8,
        "int16": hub.QuantizeDtype.INT16,
    }[activations_dtype]
    calibration_data, total_bytes = build_random_calibration(
        input_specs,
        sample_count,
        seed,
    )
    model = hub.get_model(model_id)
    job = hub.submit_quantize_job(
        model=model,
        calibration_data=calibration_data,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=activation_type,
        name=name,
    )
    return {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job.job_id,
        "url": job.url,
        "name": name,
        "model": describe_model(model),
        "calibration": {
            "kind": "synthetic_random_latency_only",
            "accuracy_valid": False,
            "specs_source": specs_source,
            "input_count": len(input_specs),
            "sample_count": sample_count,
            "seed": seed,
            "uncompressed_bytes": total_bytes,
        },
        "quantization": {"weights": "int8", "activations": activations_dtype},
    }


def build_random_calibration(
    input_specs: dict[str, tuple[tuple[int, ...], str]],
    sample_count: int,
    seed: int,
    max_bytes: int = 512 * 1024 * 1024,
) -> tuple[dict[str, list[Any]], int]:
    """Create bounded synthetic calibration data for deployment smoke only."""
    import numpy as np

    rng = np.random.default_rng(seed)
    calibration_data: dict[str, list[Any]] = {}
    total_bytes = 0
    sequence_length = next(
        (
            int(shape[-1])
            for name, (shape, _) in input_specs.items()
            if name in {"input", "input_ids"} and len(shape) >= 2
        ),
        None,
    )
    for name, (shape, dtype_name) in input_specs.items():
        normalized_shape = tuple(int(dimension) for dimension in shape)
        if not normalized_shape or any(dimension < 1 for dimension in normalized_shape):
            raise ValueError(f"invalid input shape for {name}: {shape}")
        dtype = np.dtype(dtype_name)
        tensor_bytes = math.prod(normalized_shape) * dtype.itemsize * sample_count
        total_bytes += tensor_bytes
        if total_bytes > max_bytes:
            raise ValueError(f"synthetic calibration exceeds {max_bytes} bytes")
        if name == "scales" and np.issubdtype(dtype, np.floating):
            samples = [np.ones(normalized_shape, dtype=dtype) for _ in range(sample_count)]
        elif np.issubdtype(dtype, np.floating):
            samples = [
                rng.standard_normal(normalized_shape).astype(dtype)
                for _ in range(sample_count)
            ]
        elif name in {"input", "input_ids"}:
            samples = [
                rng.integers(1, 256, size=normalized_shape, dtype=dtype)
                for _ in range(sample_count)
            ]
        elif name in {"input_length", "input_lengths"} and sequence_length is not None:
            samples = [
                np.full(normalized_shape, sequence_length, dtype=dtype)
                for _ in range(sample_count)
            ]
        else:
            samples = [np.zeros(normalized_shape, dtype=dtype) for _ in range(sample_count)]
        calibration_data[name] = samples
    return calibration_data, total_bytes


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    environment = subparsers.add_parser("environment")
    environment.add_argument("--device", default=DEFAULT_DEVICE)
    environment.add_argument("--device-os", default=DEFAULT_DEVICE_OS)
    environment.add_argument("--output", required=True)

    reference = subparsers.add_parser("reference")
    reference.add_argument("--artifact-dir", required=True)
    reference.add_argument("--encoder-ms", required=True, type=float)
    reference.add_argument("--decoder-ms", required=True, type=float)
    reference.add_argument("--token-counts", nargs="+", type=int, default=[10, 20, 40, 80, 200])
    reference.add_argument("--output", required=True)

    job_parser = subparsers.add_parser("job")
    job_parser.add_argument("--job-id", required=True)
    job_parser.add_argument("--wait", action="store_true")
    job_parser.add_argument("--timeout", type=int, default=900)
    job_parser.add_argument("--include-details", action="store_true")
    job_parser.add_argument("--output", required=True)

    submit_parser = subparsers.add_parser("submit-profile")
    model_group = submit_parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model")
    model_group.add_argument("--model-id")
    submit_parser.add_argument("--device", default=DEFAULT_DEVICE)
    submit_parser.add_argument("--device-os", default=DEFAULT_DEVICE_OS)
    submit_parser.add_argument("--name", required=True)
    submit_parser.add_argument("--qairt-version", default=DEFAULT_QAIRT_VERSION)
    submit_parser.add_argument(
        "--compute-unit",
        choices=["cpu", "gpu", "npu"],
        default="npu",
    )
    submit_parser.add_argument(
        "--use-runtime-defaults",
        action="store_true",
        help="A/B retry without forcing QAIRT version or compute unit.",
    )
    submit_parser.add_argument("--output", required=True)

    compile_parser = subparsers.add_parser("submit-compile")
    compile_model_group = compile_parser.add_mutually_exclusive_group(required=True)
    compile_model_group.add_argument("--model")
    compile_model_group.add_argument("--model-id")
    compile_parser.add_argument("--device", default=DEFAULT_DEVICE)
    compile_parser.add_argument("--device-os", default=DEFAULT_DEVICE_OS)
    compile_parser.add_argument("--name", required=True)
    compile_parser.add_argument(
        "--target-runtime",
        choices=["onnx", "qnn_dlc", "qnn_context_binary"],
        required=True,
    )
    compile_parser.add_argument("--extra-options", default="")
    compile_parser.add_argument("--input-specs-json")
    compile_parser.add_argument("--output", required=True)

    link_parser = subparsers.add_parser("submit-link")
    link_parser.add_argument("--model-id", action="append", required=True)
    link_parser.add_argument("--device", default=DEFAULT_DEVICE)
    link_parser.add_argument("--device-os", default=DEFAULT_DEVICE_OS)
    link_parser.add_argument("--name", required=True)
    link_parser.add_argument("--extra-options", default="")
    link_parser.add_argument("--output", required=True)

    quantize_parser = subparsers.add_parser("submit-random-quantize")
    quantize_parser.add_argument("--model-id", required=True)
    quantize_parser.add_argument("--input-name")
    quantize_parser.add_argument("--input-shape", nargs="+", type=int)
    quantize_parser.add_argument("--input-specs-job-id")
    quantize_parser.add_argument("--samples", type=int, default=1)
    quantize_parser.add_argument("--seed", type=int, default=20260926)
    quantize_parser.add_argument("--name", required=True)
    quantize_parser.add_argument(
        "--activations-dtype",
        choices=["int8", "int16"],
        default="int16",
    )
    quantize_parser.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "environment":
        payload = capture_environment(args.device, args.device_os)
    elif args.command == "reference":
        payload = capture_reference(
            args.artifact_dir,
            args.encoder_ms,
            args.decoder_ms,
            args.token_counts,
        )
    elif args.command == "job":
        payload = capture_job(args.job_id, args.wait, args.timeout, args.include_details)
    elif args.command == "submit-profile":
        payload = submit_profile(
            args.model,
            args.model_id,
            args.device,
            args.device_os,
            args.name,
            args.qairt_version,
            args.compute_unit,
            args.use_runtime_defaults,
        )
    elif args.command == "submit-compile":
        payload = submit_compile(
            args.model,
            args.model_id,
            args.device,
            args.device_os,
            args.name,
            args.target_runtime,
            args.extra_options,
            args.input_specs_json,
        )
    elif args.command == "submit-link":
        payload = submit_link(
            args.model_id,
            args.device,
            args.device_os,
            args.name,
            args.extra_options,
        )
    else:
        payload = submit_random_quantize(
            args.model_id,
            args.input_name,
            tuple(args.input_shape) if args.input_shape else None,
            args.input_specs_job_id,
            args.samples,
            args.seed,
            args.name,
            args.activations_dtype,
        )
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

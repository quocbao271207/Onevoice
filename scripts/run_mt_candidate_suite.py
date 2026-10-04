"""Evaluate one MT LoRA candidate on every locked gate and archive the evidence.

Fine-tuning is not treated as a safety claim.  Promotion is fail-closed unless
the full MT test manifest and every required clinical preservation slice pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_gpu_rounds import monitor_process  # noqa: E402


BASE_MODEL = "facebook/nllb-200-distilled-600M"
BASE_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_locked_suite(config: dict[str, Any], suite_path: Path, lock_path: Path) -> str:
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    expected = lock["evaluation"]["medical_safety_mt_sha256"]
    actual = sha256(suite_path)
    configured = (ROOT / config["evaluation"]["locked_mt_safety_suite"]).resolve()
    if suite_path.resolve() != configured:
        raise ValueError(f"Safety suite is not the configured locked artifact: {suite_path}")
    if actual != expected:
        raise ValueError(f"Locked safety suite checksum mismatch: {actual} != {expected}")
    return actual


def candidate_checks(
    config: dict[str, Any], aggregate_report: dict[str, Any], clinical_report: dict[str, Any]
) -> list[dict[str, Any]]:
    aggregate = config["release_gates"]["aggregate"]
    checks = [
        {
            "name": "mt_sacrebleu",
            "actual": float(aggregate_report["sacrebleu"]),
            "operator": ">=",
            "limit": float(aggregate["mt_sacrebleu_min"]),
            "pass": float(aggregate_report["sacrebleu"]) >= float(aggregate["mt_sacrebleu_min"]),
        },
        {
            "name": "mt_chrf2",
            "actual": float(aggregate_report["chrf2"]),
            "operator": ">=",
            "limit": float(aggregate["mt_chrf2_min"]),
            "pass": float(aggregate_report["chrf2"]) >= float(aggregate["mt_chrf2_min"]),
        },
    ]
    clinical_limits = config["release_gates"]["clinical_safety"]
    categories = clinical_report.get("categories", {})
    for category in config["evaluation"]["required_slices"]:
        category_report = categories.get(category)
        limit_key = f"{category}_failure_rate_max"
        limit = float(clinical_limits.get(limit_key, clinical_limits["terminology_failure_rate_max"]))
        if category_report is None:
            checks.append(
                {"name": f"clinical_{category}", "pass": False, "reason": "missing required slice"}
            )
            continue
        actual = float(category_report["safety_failure_rate"])
        checks.append(
            {
                "name": f"clinical_{category}_failure_rate",
                "actual": actual,
                "operator": "<=",
                "limit": limit,
                "pass": actual <= limit,
            }
        )
    return checks


def run_benchmark(
    *,
    adapter: Path,
    manifest: Path,
    name: str,
    output_dir: Path,
    device: str,
    precision: str,
    batch_size: int,
    num_beams: int,
    gpu_memory_fraction: float,
    utilization_limits: dict[str, Any],
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_baseline_benchmarks.py"),
        "--task",
        "mt",
        "--model",
        BASE_MODEL,
        "--model-revision",
        BASE_REVISION,
        "--adapter",
        str(adapter),
        "--manifest",
        str(manifest),
        "--samples",
        "0",
        "--batch-size",
        str(batch_size),
        "--num-beams",
        str(num_beams),
        "--device",
        device,
        "--precision",
        precision,
        "--gpu-memory-fraction",
        str(gpu_memory_fraction),
        "--name",
        name,
        "--output-dir",
        str(output_dir),
    ]
    log_path = output_dir / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        monitor_summary = monitor_process(
            process,
            output_dir / f"{name}_resource_monitor.jsonl",
            utilization_limits,
        )
    if monitor_summary["return_code"] != 0:
        raise subprocess.CalledProcessError(monitor_summary["return_code"], command)
    return monitor_summary


def archive_evidence(output_dir: Path) -> tuple[Path, str]:
    archive_path = output_dir.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(output_dir.parent), recursive=False)
    digest = sha256(archive_path)
    archive_path.with_suffix(archive_path.suffix + ".sha256").write_text(
        f"{digest}  {archive_path.name}\n", encoding="utf-8"
    )
    return archive_path, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=ROOT / "data" / "processed" / "manifests" / "mt--test.jsonl",
    )
    parser.add_argument(
        "--safety-manifest", type=Path, default=ROOT / "data" / "eval" / "medical_safety_mt.jsonl"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "accuracy_program.yaml")
    parser.add_argument("--artifact-lock", type=Path, default=ROOT / "configs" / "artifact_lock.yaml")
    parser.add_argument("--resource-config", type=Path, default=ROOT / "configs" / "gpu_rounds.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.35)
    args = parser.parse_args()

    if not (args.adapter / "adapter_config.json").is_file():
        parser.error("--adapter must be a completed PEFT adapter containing adapter_config.json")
    if min(args.batch_size, args.num_beams) < 1:
        parser.error("batch size and beam count must be positive")
    if not 0.0 < args.gpu_memory_fraction <= 0.40:
        parser.error("--gpu-memory-fraction must be in (0, 0.40]")
    for path in (
        args.test_manifest,
        args.safety_manifest,
        args.config,
        args.artifact_lock,
        args.resource_config,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = (args.output_dir or ROOT / "data" / "reports" / "candidates" / f"mt-{stamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    utilization_limits = yaml.safe_load(args.resource_config.read_text(encoding="utf-8"))["limits"]
    if float(utilization_limits["gpu_memory_fraction"]) != args.gpu_memory_fraction:
        raise ValueError(
            "Candidate memory fraction must match the centrally configured GPU resource limit"
        )
    suite_sha256 = validate_locked_suite(config, args.safety_manifest, args.artifact_lock)

    status = "error"
    error: str | None = None
    checks: list[dict[str, Any]] = []
    resources: dict[str, Any] = {}
    try:
        resources["aggregate"] = run_benchmark(
            adapter=args.adapter,
            manifest=args.test_manifest,
            name="mt_candidate",
            output_dir=output_dir,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            gpu_memory_fraction=args.gpu_memory_fraction,
            utilization_limits=utilization_limits,
        )
        resources["clinical"] = run_benchmark(
            adapter=args.adapter,
            manifest=args.safety_manifest,
            name="mt_clinical_candidate",
            output_dir=output_dir,
            device=args.device,
            precision=args.precision,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            gpu_memory_fraction=args.gpu_memory_fraction,
            utilization_limits=utilization_limits,
        )
        checks = candidate_checks(
            config,
            read_json(output_dir / "mt_candidate.json"),
            read_json(output_dir / "mt_clinical_candidate.json"),
        )
        status = "pass" if all(check["pass"] for check in checks) else "fail"
    except Exception as exc:  # Preserve logs and a machine-readable failure artifact.
        error = f"{type(exc).__name__}: {exc}"

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "promotion_allowed": status == "pass",
        "adapter": str(args.adapter.resolve()),
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "test_manifest": str(args.test_manifest.resolve()),
        "locked_safety_manifest": str(args.safety_manifest.resolve()),
        "locked_safety_sha256": suite_sha256,
        "checks": checks,
        "resource_limits": utilization_limits,
        "resource_runs": resources,
        "error": error,
        "policy": (
            "Fine-tuning does not establish safety. Full aggregate MT quality and every locked "
            "drug-name, dose, number, unit, negation, terminology, and code-switch slice must pass."
        ),
    }
    (output_dir / "candidate_gate.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive_path, archive_sha256 = archive_evidence(output_dir)
    print(
        json.dumps(
            {**summary, "archive": str(archive_path), "archive_sha256": archive_sha256},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

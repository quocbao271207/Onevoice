"""Evaluate one Vietnamese ASR LoRA candidate on every locked release gate.

Fine-tuning is not a safety claim. Promotion is fail-closed unless the full
Vietnamese test set, code-switch slice, and every explicit clinical
transcription-preservation slice pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.candidate_evidence import (  # noqa: E402
    archive_evidence,
    evidence_sidecars,
    read_json,
    sha256,
)
from scripts.run_gpu_rounds import monitor_process  # noqa: E402


BASE_MODEL = "vinai/PhoWhisper-small"
BASE_REVISION = "a86b604c346caf7148c37512eafe783a16420adb"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_locked_inputs(
    config: dict[str, Any], test_path: Path, safety_path: Path, lock_path: Path
) -> dict[str, str]:
    """Verify hashes and prove the clinical suite is an exact test-set subset."""
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    configured_safety = (ROOT / config["evaluation"]["locked_asr_safety_suite"]).resolve()
    configured_test = (ROOT / "data/processed/manifests/asr--test-local.jsonl").resolve()
    if safety_path.resolve() != configured_safety:
        raise ValueError(f"Safety suite is not the configured locked artifact: {safety_path}")
    if test_path.resolve() != configured_test:
        raise ValueError(f"Test manifest is not the configured locked artifact: {test_path}")

    hashes = {
        "test": sha256(test_path),
        "safety": sha256(safety_path),
    }
    expected = {
        "test": lock["manifests"]["asr_test_local_sha256"],
        "safety": lock["evaluation"]["medical_safety_asr_vi_sha256"],
    }
    for name in hashes:
        if hashes[name] != expected[name]:
            raise ValueError(f"Locked {name} checksum mismatch: {hashes[name]} != {expected[name]}")

    test_rows = {row["id"]: row for row in read_jsonl(test_path)}
    safety_rows = read_jsonl(safety_path)
    if len({row["id"] for row in safety_rows}) != len(safety_rows):
        raise ValueError("Locked ASR safety suite contains duplicate IDs")
    observed_categories = set()
    for row in safety_rows:
        source = test_rows.get(row["id"])
        if source is None:
            raise ValueError(f"Safety row is absent from locked test: {row['id']}")
        for field in ("text", "audio_path", "language"):
            if row.get(field) != source.get(field):
                raise ValueError(f"Safety row {row['id']} changed locked field: {field}")
        if not str(row.get("language", "")).startswith("vi"):
            raise ValueError(f"Non-Vietnamese row in ASR-VI safety suite: {row['id']}")
        categories = set(row.get("categories", []))
        expectations = row.get("safety_expectations", {})
        if not categories or categories != set(expectations):
            raise ValueError(f"Safety row has incomplete expectations: {row['id']}")
        observed_categories.update(categories)
    required = set(config["evaluation"]["required_slices"])
    if not required <= observed_categories:
        missing = sorted(required - observed_categories)
        raise ValueError(f"Locked ASR safety suite is missing slices: {missing}")
    return hashes


def candidate_checks(
    config: dict[str, Any], aggregate_report: dict[str, Any], safety_report: dict[str, Any]
) -> list[dict[str, Any]]:
    aggregate_limit = float(config["release_gates"]["aggregate"]["asr_vi_wer_max"])
    checks = [
        {
            "name": "asr_vi_wer",
            "actual": float(aggregate_report["wer"]),
            "operator": "<=",
            "limit": aggregate_limit,
            "pass": float(aggregate_report["wer"]) <= aggregate_limit,
        }
    ]
    code_switch = aggregate_report.get("slices", {}).get("code_switch", {}).get("True")
    code_switch_limit = float(config["release_gates"]["slices"]["asr_code_switch_wer_max"])
    if code_switch is None:
        checks.append(
            {"name": "asr_code_switch_wer", "pass": False, "reason": "missing required slice"}
        )
    else:
        actual = float(code_switch["wer"])
        checks.append(
            {
                "name": "asr_code_switch_wer",
                "actual": actual,
                "operator": "<=",
                "limit": code_switch_limit,
                "pass": actual <= code_switch_limit,
            }
        )

    category_reports = safety_report.get("categories", {})
    clinical_limits = config["release_gates"]["clinical_safety"]
    for category in config["evaluation"]["required_slices"]:
        report = category_reports.get(category)
        if report is None or int(report.get("samples", 0)) < 1:
            checks.append(
                {
                    "name": f"clinical_{category}",
                    "pass": False,
                    "reason": "missing or empty required slice",
                }
            )
            continue
        limit = float(
            clinical_limits.get(
                f"{category}_failure_rate_max",
                clinical_limits["terminology_failure_rate_max"],
            )
        )
        actual = float(report["safety_failure_rate"])
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
        str(ROOT / "scripts/run_baseline_benchmarks.py"),
        "--task",
        "asr",
        "--language",
        "vi",
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
        summary = monitor_process(
            process,
            output_dir / f"{name}_resource_monitor.jsonl",
            utilization_limits,
        )
    if summary["return_code"] != 0:
        raise subprocess.CalledProcessError(summary["return_code"], command)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=ROOT / "data/processed/manifests/asr--test-local.jsonl",
    )
    parser.add_argument(
        "--safety-manifest",
        type=Path,
        default=ROOT / "data/eval/medical_safety_asr_vi.jsonl",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/accuracy_program.yaml")
    parser.add_argument("--artifact-lock", type=Path, default=ROOT / "configs/artifact_lock.yaml")
    parser.add_argument("--resource-config", type=Path, default=ROOT / "configs/gpu_rounds.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=1)
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
    output_dir = (
        args.output_dir or ROOT / "data/reports/candidates" / f"asr-vi-{stamp}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    utilization_limits = yaml.safe_load(args.resource_config.read_text(encoding="utf-8"))["limits"]
    if float(utilization_limits["gpu_memory_fraction"]) != args.gpu_memory_fraction:
        raise ValueError(
            "Candidate memory fraction must match the centrally configured GPU resource limit"
        )
    locked_hashes = validate_locked_inputs(
        config, args.test_manifest, args.safety_manifest, args.artifact_lock
    )

    status = "error"
    error: str | None = None
    checks: list[dict[str, Any]] = []
    resources: dict[str, Any] = {}
    try:
        resources["aggregate"] = run_benchmark(
            adapter=args.adapter,
            manifest=args.test_manifest,
            name="asr_vi_candidate",
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
            name="asr_vi_clinical_candidate",
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
            read_json(output_dir / "asr_vi_candidate.json"),
            read_json(output_dir / "asr_vi_clinical_candidate.json"),
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
        "locked_hashes": locked_hashes,
        "checks": checks,
        "resource_limits": utilization_limits,
        "resource_runs": resources,
        "error": error,
        "policy": (
            "Fine-tuning does not establish safety. Full Vietnamese WER, code-switch WER, and "
            "every locked drug-name, dose, number, unit, negation, terminology, and "
            "code-switch transcription-preservation slice must pass."
        ),
    }
    (output_dir / "candidate_gate.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive_path, archive_sha256 = archive_evidence(output_dir)
    archive_checksum, archive_manifest = evidence_sidecars(archive_path)
    print(
        json.dumps(
            {
                **summary,
                "archive": str(archive_path),
                "archive_bytes": archive_path.stat().st_size,
                "archive_sha256": archive_sha256,
                "archive_checksum": str(archive_checksum),
                "archive_manifest": str(archive_manifest),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

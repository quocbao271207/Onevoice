"""Report OneVoice readiness and fail if a requested gate is not satisfied."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.durable_json import write_durable_json  # noqa: E402
from src.pipeline.evidence_paths import resolve_regular_file_under  # noqa: E402
from src.pipeline.stable_json import read_stable_json_mapping  # noqa: E402
from src.pipeline.stable_jsonl import read_stable_jsonl_mappings  # noqa: E402
from src.pipeline.stable_yaml import read_stable_yaml_mapping  # noqa: E402
from src.utils.bounded_file import sha256_stable_regular_file  # noqa: E402


MAX_PROJECT_CONFIG_BYTES = 1_000_000
MAX_PREFLIGHT_JSON_BYTES = 100_000_000
MAX_PREFLIGHT_JSONL_BYTES = 250_000_000
MAX_PREFLIGHT_JSONL_LINE_BYTES = 2_000_000
MAX_PREFLIGHT_JSONL_ROWS = 100_000
MAX_PREFLIGHT_HASHED_FILE_BYTES = 16_000_000_000
MAX_PREFLIGHT_AUDIO_BYTES = 128 * 1024 * 1024
MAX_PREFLIGHT_REPORT_BYTES = 10_000_000


def sha256(
    path: Path,
    *,
    maximum_bytes: int = MAX_PREFLIGHT_HASHED_FILE_BYTES,
    label: str = "Preflight input",
) -> str:
    digest, _ = sha256_stable_regular_file(
        path,
        maximum_bytes=maximum_bytes,
        label=label,
    )
    return digest


def check(condition: bool, name: str, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(condition), "detail": detail}


def load_json(path: Path) -> dict[str, Any]:
    return read_stable_json_mapping(
        path,
        maximum_bytes=MAX_PREFLIGHT_JSON_BYTES,
        label="Preflight JSON input",
    ).mapping


def local_audio_manifest(path: Path) -> tuple[int, int, int]:
    document = read_stable_jsonl_mappings(
        path,
        maximum_bytes=MAX_PREFLIGHT_JSONL_BYTES,
        maximum_line_bytes=MAX_PREFLIGHT_JSONL_LINE_BYTES,
        maximum_rows=MAX_PREFLIGHT_JSONL_ROWS,
        label="Preflight local-audio manifest",
    )
    missing = 0
    ids: set[str] = set()
    audio_root = ROOT / "data" / "processed" / "audio_16k"
    for row in document.rows:
        record_id = row.get("id")
        if isinstance(record_id, str) and record_id:
            ids.add(record_id)
        relative = row.get("audio_path")
        if not isinstance(relative, str) or not relative:
            missing += 1
            continue
        audio_path = Path(relative)
        if audio_path.is_absolute():
            missing += 1
            continue
        try:
            resolve_regular_file_under(
                audio_path,
                project_root=ROOT,
                allowed_root=audio_root,
                label="Preflight manifest audio",
                maximum_bytes=MAX_PREFLIGHT_AUDIO_BYTES,
            )
        except (FileNotFoundError, ValueError):
            missing += 1
    return len(document.rows), len(ids), missing


def verify_audio_inventory(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    try:
        document = read_stable_jsonl_mappings(
            path,
            maximum_bytes=MAX_PREFLIGHT_JSONL_BYTES,
            maximum_line_bytes=MAX_PREFLIGHT_JSONL_LINE_BYTES,
            maximum_rows=MAX_PREFLIGHT_JSONL_ROWS,
            label="Preflight audio inventory",
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return False, f"invalid_inventory={exc}"

    entries: list[tuple[Path, str, int, str]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    total_bytes = 0
    audio_root = ROOT / "data" / "processed" / "audio_16k"
    for line_number, row in enumerate(document.rows, 1):
        relative = row.get("audio_path")
        record_id = row.get("id")
        expected_size = row.get("bytes")
        expected_hash = row.get("sha256")
        if not isinstance(relative, str) or not relative:
            return False, f"line={line_number} invalid_audio_path"
        audio_path = Path(relative)
        if audio_path.is_absolute():
            return False, f"line={line_number} absolute_path"
        if not isinstance(record_id, str) or not record_id:
            return False, f"line={line_number} invalid_id"
        if record_id in seen_ids:
            return False, f"line={line_number} duplicate_id={record_id}"
        seen_ids.add(record_id)
        portable_path = audio_path.as_posix()
        if portable_path in seen_paths:
            return False, f"line={line_number} duplicate_audio_path={portable_path}"
        seen_paths.add(portable_path)
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or not 0 < expected_size <= MAX_PREFLIGHT_AUDIO_BYTES
        ):
            return False, f"line={line_number} invalid_bytes"
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            return False, f"line={line_number} invalid_sha256"
        if expected_hash in seen_hashes:
            return False, f"line={line_number} duplicate_audio_hash={expected_hash}"
        seen_hashes.add(expected_hash)
        try:
            resolved = resolve_regular_file_under(
                audio_path,
                project_root=ROOT,
                allowed_root=audio_root,
                label="Preflight inventory audio",
                maximum_bytes=MAX_PREFLIGHT_AUDIO_BYTES,
            )
        except (FileNotFoundError, ValueError) as exc:
            return False, f"line={line_number} invalid_audio={exc}"
        entries.append((resolved, expected_hash, expected_size, portable_path))
        total_bytes += expected_size

    def hash_entry(
        entry: tuple[Path, str, int, str],
    ) -> tuple[str | None, int | None, str, str | None]:
        resolved, expected, expected_size, relative = entry
        try:
            actual, actual_size = sha256_stable_regular_file(
                resolved,
                maximum_bytes=MAX_PREFLIGHT_AUDIO_BYTES,
                label="Preflight inventory audio",
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            return None, None, relative, str(exc)
        mismatch = None
        if actual_size != expected_size:
            mismatch = "size_mismatch"
        elif actual != expected:
            mismatch = "hash_mismatch"
        return actual, actual_size, relative, mismatch

    with ThreadPoolExecutor(max_workers=8) as pool:
        for _actual, _size, relative, error in pool.map(hash_entry, entries):
            if error:
                return False, f"{error}={relative}"
    return bool(entries), f"files={len(entries)} bytes={total_bytes}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=["data", "gpu"], default="data")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "reports" / "preflight.json")
    args = parser.parse_args()

    dataset_config_path = ROOT / "configs" / "datasets.yaml"
    config = read_stable_yaml_mapping(
        dataset_config_path,
        maximum_bytes=MAX_PROJECT_CONFIG_BYTES,
        label="Preflight dataset config",
    ).mapping
    checks = [check(True, "dataset_config", str(dataset_config_path))]
    eda_dir = ROOT / "data" / "reports" / "eda"

    for name, spec in config["datasets"].items():
        if not spec.get("enabled"):
            continue
        audit_path = eda_dir / f"{name}.json"
        if not audit_path.exists():
            checks.append(check(False, f"audit:{name}", "missing"))
            continue
        audit = load_json(audit_path)
        passed = audit.get("full_audit") and audit.get("status") in {"pass", "reviewed"}
        checks.append(check(passed, f"audit:{name}", f"status={audit.get('status')} full={audit.get('full_audit')}"))

        if spec.get("task") == "asr":
            review = audit.get("human_listening_review") or {}
            expected_samples = int(config.get("quality_gate", {}).get("human_review_samples_per_source", 0))
            verdicts = review.get("verdicts") or {}
            sample_count = int(review.get("sample_count") or 0)
            review_passed = (
                sample_count >= expected_samples
                and int(verdicts.get("keep") or 0) == sample_count
                and int((review.get("audio_ok") or {}).get("yes") or 0) == sample_count
                and int((review.get("transcript_exact") or {}).get("yes") or 0) == sample_count
            )
            checks.append(
                check(
                    review_passed,
                    f"human_review:{name}",
                    f"samples={sample_count} expected>={expected_samples} keep={verdicts.get('keep', 0)}",
                )
            )

    review_decisions_path = ROOT / "configs" / "data_review_decisions.yaml"
    if review_decisions_path.is_file():
        decisions = read_stable_yaml_mapping(
            review_decisions_path,
            maximum_bytes=MAX_PROJECT_CONFIG_BYTES,
            label="Preflight data review decisions",
        ).mapping
        pii_decision = decisions.get("pii_scan", {})
        pii_report_path = ROOT / str(pii_decision.get("report") or "")
        pii_report = load_json(pii_report_path) if pii_report_path.is_file() else {}
        exclusions_path = ROOT / "configs" / "data_exclusions.yaml"
        exclusion_config = read_stable_yaml_mapping(
            exclusions_path,
            maximum_bytes=MAX_PROJECT_CONFIG_BYTES,
            label="Preflight data exclusions",
        ).mapping
        excluded_ids = {str(item.get("id")) for item in exclusion_config.get("exclusions", [])}
        true_positive_ids = {str(value) for value in pii_decision.get("true_positive_exclusions", [])}
        resolved_count = len(true_positive_ids) + int(pii_decision.get("retained_reviewed_false_positives") or 0)
        pii_ok = (
            pii_decision.get("status") == "reviewed"
            and pii_report_path.is_file()
            and sha256(pii_report_path) == pii_decision.get("report_sha256")
            and int(pii_report.get("candidate_records") or 0) == int(pii_decision.get("candidate_records") or -1)
            and resolved_count == int(pii_report.get("candidate_records") or -1)
            and true_positive_ids <= excluded_ids
        )
        checks.append(
            check(
                pii_ok,
                "pii_review_resolution",
                f"candidates={pii_report.get('candidate_records')} resolved={resolved_count} exclusions={len(true_positive_ids)}",
            )
        )

        alignment_decision = decisions.get("medev_alignment", {})
        alignment_path = ROOT / str(alignment_decision.get("report") or "")
        alignment_report = load_json(alignment_path) if alignment_path.is_file() else {}
        reviewed_fatal = set(alignment_decision.get("fatal_flags_removed") or [])
        reviewed_slices = set(alignment_decision.get("retained_review_slices") or [])
        alignment_ok = (
            alignment_decision.get("status") == "reviewed"
            and alignment_path.is_file()
            and sha256(alignment_path) == alignment_decision.get("report_sha256")
            and reviewed_fatal == set(alignment_report.get("fatal_candidates") or [])
            and reviewed_slices == set(alignment_report.get("review_slices") or [])
        )
        checks.append(
            check(
                alignment_ok,
                "medev_alignment_resolution",
                f"fatal={len(reviewed_fatal)} review_slices={len(reviewed_slices)}",
            )
        )
    else:
        checks.append(check(False, "data_review_decisions", "missing"))

    if args.gate == "gpu":
        from scripts.build_terminology_challenge_set import verify_challenge_artifact
        from src.training.finetune_mt_medical import MTTrainingConfig
        from src.training.finetune_whisper_vi import ASRTrainingConfig

        manifest_dir = ROOT / "data" / "processed" / "manifests"
        lock_path = ROOT / "configs" / "artifact_lock.yaml"
        artifact_lock = (
            read_stable_yaml_mapping(
                lock_path,
                maximum_bytes=MAX_PROJECT_CONFIG_BYTES,
                label="Preflight artifact lock",
            ).mapping
            if lock_path.is_file()
            else {}
        )
        locked_hashes = artifact_lock.get("manifests", {})
        locked_models = artifact_lock.get("base_models", {})
        locked_evaluation = artifact_lock.get("evaluation", {})
        for label, model_name, revision in (
            ("asr", ASRTrainingConfig.base_model, ASRTrainingConfig.base_model_revision),
            ("mt", MTTrainingConfig.base_model, MTTrainingConfig.base_model_revision),
        ):
            expected_revision = locked_models.get(model_name)
            checks.append(
                check(
                    bool(expected_revision) and revision == expected_revision,
                    f"training_revision:{label}",
                    f"model={model_name} revision={revision} expected={expected_revision}",
                )
            )
        required = {
            "asr--train-local.jsonl": "asr_train_local_sha256",
            "asr--validation-local.jsonl": "asr_validation_local_sha256",
            "asr--test-local.jsonl": "asr_test_local_sha256",
            "mt--train.jsonl": "mt_train_sha256",
            "mt--validation.jsonl": "mt_validation_sha256",
            "mt--test.jsonl": "mt_test_sha256",
        }
        for filename, lock_key in required.items():
            path = manifest_dir / filename
            actual = sha256(path) if path.is_file() else None
            expected = locked_hashes.get(lock_key)
            detail = f"sha256={actual} expected={expected}" if actual else "missing"
            checks.append(
                check(
                    path.is_file() and path.stat().st_size > 0 and actual == expected,
                    f"manifest:{filename}",
                    detail,
                )
            )
            if filename.startswith("asr--") and filename.endswith("-local.jsonl") and path.is_file():
                rows, unique_ids, missing = local_audio_manifest(path)
                checks.append(
                    check(
                        rows > 0 and unique_ids == rows and missing == 0,
                        f"local_audio:{filename}",
                        f"rows={rows} unique_ids={unique_ids} missing_audio={missing}",
                    )
                )
        for filename, lock_key in {
            "medical_safety_asr_vi.jsonl": "medical_safety_asr_vi_sha256",
            "medical_safety_mt.jsonl": "medical_safety_mt_sha256",
        }.items():
            path = ROOT / "data" / "eval" / filename
            actual = sha256(path) if path.is_file() else None
            expected = locked_evaluation.get(lock_key)
            checks.append(
                check(
                    actual is not None and actual == expected,
                    f"evaluation:{filename}",
                    f"sha256={actual} expected={expected}",
                )
            )
        challenge_paths = {
            "sources": ROOT / str(locked_evaluation.get("terminology_challenge_sources") or ""),
            "artifact": ROOT / str(locked_evaluation.get("terminology_challenge_set") or ""),
            "report": ROOT / str(locked_evaluation.get("terminology_challenge_report") or ""),
        }
        challenge_hash_keys = {
            "sources": "terminology_challenge_sources_sha256",
            "artifact": "terminology_challenge_set_sha256",
            "report": "terminology_challenge_report_sha256",
        }
        challenge_hashes_match = all(
            path.is_file()
            and sha256(path) == locked_evaluation.get(challenge_hash_keys[name])
            for name, path in challenge_paths.items()
        )
        challenge_ok, challenge_detail = (
            verify_challenge_artifact(
                challenge_paths["artifact"],
                challenge_paths["report"],
                challenge_paths["sources"],
            )
            if challenge_hashes_match
            else (False, "missing artifact or SHA-256 mismatch")
        )
        checks.append(
            check(
                challenge_hashes_match
                and challenge_ok
                and locked_evaluation.get("terminology_challenge_status")
                == "awaiting_human_review",
                "terminology_challenge_set",
                challenge_detail,
            )
        )
        for name, path in {
            "manifest_validation": ROOT / "data" / "reports" / "manifests" / "validation.json",
            "audio_qc": ROOT / "data" / "reports" / "audio_qc.json",
        }.items():
            status = load_json(path).get("status") if path.is_file() else "missing"
            checks.append(check(status == "pass", name, f"status={status}"))
        for task in ("asr_vi", "asr_en", "mt"):
            path = ROOT / "data" / "reports" / "baselines" / f"{task}_base.json"
            samples = int(load_json(path).get("samples", 0)) if path.is_file() else 0
            checks.append(check(samples >= 20, f"baseline:{task}", f"samples={samples}"))
        e2e_path = ROOT / "data" / "reports" / "smoke" / "base_e2e" / "report.json"
        e2e = load_json(e2e_path) if e2e_path.is_file() else {}
        e2e_results = e2e.get("results") or []
        e2e_ok = (
            e2e.get("status") == "pass"
            and int(e2e.get("samples") or 0) >= 2
            and {row.get("direction") for row in e2e_results} >= {"vi_to_en", "en_to_vi"}
            and all(row.get("output_audio_exists") for row in e2e_results)
        )
        checks.append(
            check(
                e2e_ok,
                "base_e2e_cascade_smoke",
                f"status={e2e.get('status')} samples={e2e.get('samples')} directions={sorted(row.get('direction') for row in e2e_results)}",
            )
        )
        token_report_path = ROOT / "data" / "reports" / "eda" / "medev_token_lengths.json"
        token_report = load_json(token_report_path) if token_report_path.is_file() else {}
        token_length_ok = (
            token_report.get("status") == "pass"
            and int(token_report.get("max_length") or 0) == MTTrainingConfig.max_source_length
            and MTTrainingConfig.max_source_length == MTTrainingConfig.max_target_length
            and sha256(token_report_path) == locked_hashes.get("mt_token_length_report_sha256")
        )
        checks.append(
            check(
                token_length_ok,
                "mt_token_length_eda",
                f"status={token_report.get('status')} max_length={token_report.get('max_length')} "
                f"sha256={sha256(token_report_path) if token_report_path.is_file() else None}",
            )
        )
        checks.append(check(lock_path.is_file(), "artifact_lock", str(lock_path)))
        inventory_path = ROOT / "data" / "reports" / "audio_inventory.jsonl"
        inventory_actual = sha256(inventory_path) if inventory_path.is_file() else None
        inventory_expected = locked_hashes.get("audio_inventory_sha256")
        checks.append(
            check(
                inventory_actual is not None and inventory_actual == inventory_expected,
                "audio_inventory_lock",
                f"sha256={inventory_actual} expected={inventory_expected}",
            )
        )
        inventory_ok, inventory_detail = verify_audio_inventory(inventory_path)
        checks.append(check(inventory_ok, "audio_inventory_files", inventory_detail))
        checks.append(check((ROOT / "requirements-gpu.txt").is_file(), "gpu_requirements", "requirements-gpu.txt"))

    report = {
        "gate": args.gate,
        "ready": all(item["passed"] for item in checks),
        "checks": checks,
    }
    write_durable_json(
        args.output,
        report,
        maximum_bytes=MAX_PREFLIGHT_REPORT_BYTES,
        label="Project preflight report",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

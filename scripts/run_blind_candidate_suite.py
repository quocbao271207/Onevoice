"""Lock, verify and open the blind v2 suite exactly once per final winner."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.quality import fingerprint_text, normalize_text  # noqa: E402
from src.pipeline.evidence_paths import (  # noqa: E402
    is_link_or_junction,
    resolve_regular_file,
    resolve_regular_file_under,
)
from src.pipeline.durable_json import write_durable_json  # noqa: E402
from src.pipeline.license_policy import license_decisions  # noqa: E402
from src.pipeline.selection_policy import (  # noqa: E402
    configured_selection_hashes,
    selection_identity,
    selection_policy_record,
)
from src.utils.bounded_file import (  # noqa: E402
    read_stable_regular_file,
    sha256_stable_regular_file,
)


MAX_BLIND_REPORT_BYTES = 50_000_000
MAX_BLIND_PROVENANCE_BYTES = 5_000_000
MAX_BLIND_LOCK_BYTES = 5_000_000
MAX_BLIND_MANIFEST_BYTES = 250_000_000
MAX_BLIND_MANIFEST_LINE_BYTES = 2_000_000
MAX_ACCURACY_CONFIG_BYTES = 1_000_000
MAX_BLIND_ADAPTER_FILES = 10_000
MAX_BLIND_ADAPTER_DIRECTORIES = 10_000
MAX_BLIND_ADAPTER_ENTRIES = 20_000
MAX_BLIND_ADAPTER_FILE_BYTES = 8_000_000_000
MAX_BLIND_ADAPTER_TREE_BYTES = 16_000_000_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BLIND_LOCK_ALLOWED_FIELDS = frozenset(
    {
        "version",
        "status",
        "policy",
        "required_slices",
        "minimum_coverage",
        "manifests",
        "selection",
        "opened",
        "locked_at",
    }
)
BLIND_LOCK_REQUIRED_FIELDS = BLIND_LOCK_ALLOWED_FIELDS - {"policy", "locked_at"}
BLIND_PROVENANCE_FIELDS = frozenset(
    {
        "version",
        "created_at",
        "specification_sha256",
        "blind_manifest_sha256",
        "selection_sha256",
        "report",
        "report_bytes",
        "report_sha256",
        "resource_run",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_lock_envelope(lock: dict[str, Any]) -> None:
    if not BLIND_LOCK_REQUIRED_FIELDS.issubset(lock) or not set(lock).issubset(
        BLIND_LOCK_ALLOWED_FIELDS
    ):
        raise ValueError("Blind v2 lock schema is invalid")
    for field in ("required_slices", "minimum_coverage", "manifests"):
        value = lock.get(field)
        if not isinstance(value, dict) or set(value) != {"mt", "asr"}:
            raise ValueError(f"Blind v2 lock {field} structure is invalid")
    if "policy" in lock and not isinstance(lock["policy"], str):
        raise ValueError("Blind v2 lock policy is invalid")
    if "locked_at" in lock:
        _aware_timestamp(lock["locked_at"], label="Blind lock")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON document contains a duplicate key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> object:
    raise ValueError("JSON document contains a non-finite number")


def _parse_json_mapping(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError(f"{label} is not valid strict JSON") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _filesystem_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        getattr(metadata, "st_file_attributes", 0),
    )


def _enumerate_adapter_files(adapter: Path) -> tuple[Path, list[Path]]:
    lexical = Path(os.path.abspath(adapter))
    if is_link_or_junction(lexical):
        raise ValueError(f"Blind adapter cannot be a symlink or junction: {lexical}")
    try:
        root_metadata = lexical.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"Blind adapter is missing: {lexical}")
    except OSError:
        raise ValueError("Failed to inspect blind adapter") from None
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError(f"Blind adapter must be a directory: {lexical}")
    root = lexical.resolve(strict=True)
    try:
        lexical_after_resolve = lexical.lstat()
        resolved_metadata = root.lstat()
    except OSError:
        raise RuntimeError("Blind adapter root changed while resolving") from None
    if (
        is_link_or_junction(lexical)
        or _filesystem_identity(root_metadata)
        != _filesystem_identity(lexical_after_resolve)
        or _filesystem_identity(root_metadata)
        != _filesystem_identity(resolved_metadata)
    ):
        raise RuntimeError("Blind adapter root changed while resolving")
    pending = [root]
    files: list[Path] = []
    directory_count = 0
    entry_count = 0
    while pending:
        directory = pending.pop()
        if is_link_or_junction(directory):
            raise ValueError(
                f"Blind adapter cannot contain symlinks or junctions: {directory}"
            )
        directory_count += 1
        if directory_count > MAX_BLIND_ADAPTER_DIRECTORIES:
            raise ValueError(
                "Blind adapter exceeds "
                f"{MAX_BLIND_ADAPTER_DIRECTORIES} directories"
            )
        try:
            before = directory.lstat()
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError(f"Blind adapter entry must be a directory: {directory}")
            children = []
            for child in directory.iterdir():
                entry_count += 1
                if entry_count > MAX_BLIND_ADAPTER_ENTRIES:
                    raise ValueError(
                        f"Blind adapter exceeds {MAX_BLIND_ADAPTER_ENTRIES} entries"
                    )
                children.append(child)
            children.sort(key=lambda child: child.name)
            after = directory.lstat()
        except ValueError:
            raise
        except OSError:
            raise ValueError(f"Failed to enumerate blind adapter: {directory}") from None
        if _filesystem_identity(before) != _filesystem_identity(after):
            raise RuntimeError(f"Blind adapter directory changed while reading: {directory}")
        for child in children:
            if is_link_or_junction(child):
                raise ValueError(
                    f"Blind adapter cannot contain symlinks or junctions: {child}"
                )
            try:
                metadata = child.lstat()
            except OSError:
                raise RuntimeError(
                    f"Blind adapter entry changed while reading: {child}"
                ) from None
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(child)
            elif stat.S_ISREG(metadata.st_mode):
                files.append(child)
                if len(files) > MAX_BLIND_ADAPTER_FILES:
                    raise ValueError(
                        f"Blind adapter exceeds {MAX_BLIND_ADAPTER_FILES} files"
                    )
            else:
                raise ValueError(f"Blind adapter contains a special file: {child}")
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    return root, files


def adapter_tree_manifest(adapter: Path) -> dict[str, Any]:
    adapter, paths = _enumerate_adapter_files(adapter)
    relative_paths = [path.relative_to(adapter).as_posix() for path in paths]
    if "adapter_config.json" not in relative_paths:
        raise FileNotFoundError(f"Blind adapter is incomplete: {adapter}")
    files = []
    total_bytes = 0
    for path, relative in zip(paths, relative_paths, strict=True):
        digest, size = sha256_stable_regular_file(
            path,
            maximum_bytes=MAX_BLIND_ADAPTER_FILE_BYTES,
            label=f"Blind adapter file {relative}",
        )
        total_bytes += size
        if total_bytes > MAX_BLIND_ADAPTER_TREE_BYTES:
            raise ValueError(
                f"Blind adapter exceeds {MAX_BLIND_ADAPTER_TREE_BYTES} total bytes"
            )
        files.append({"path": relative, "bytes": size, "sha256": digest})
    _, verified_paths = _enumerate_adapter_files(adapter)
    verified_relative = [
        path.relative_to(adapter).as_posix() for path in verified_paths
    ]
    if verified_relative != relative_paths:
        raise RuntimeError("Blind adapter tree changed while hashing")
    for path, record in zip(verified_paths, files, strict=True):
        digest, size = sha256_stable_regular_file(
            path,
            maximum_bytes=MAX_BLIND_ADAPTER_FILE_BYTES,
            label=f"Blind adapter file {record['path']}",
        )
        if digest != record["sha256"] or size != record["bytes"]:
            raise RuntimeError(
                f"Blind adapter file changed while hashing: {record['path']}"
            )
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "root": str(adapter),
        "file_count": len(files),
        "bytes": total_bytes,
        "manifest_sha256": digest,
        "files": files,
    }


def _read_jsonl_document(path: Path) -> tuple[Path, bytes, list[dict[str, Any]]]:
    resolved = resolve_regular_file(
        path,
        label="Blind JSONL manifest",
        maximum_bytes=MAX_BLIND_MANIFEST_BYTES,
    )
    payload = read_stable_regular_file(
        resolved,
        maximum_bytes=MAX_BLIND_MANIFEST_BYTES,
        label="Blind JSONL manifest",
    )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        if len(line) > MAX_BLIND_MANIFEST_LINE_BYTES:
            raise ValueError(
                f"Blind JSONL line {line_number} exceeds "
                f"{MAX_BLIND_MANIFEST_LINE_BYTES} bytes"
            )
        try:
            row = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise ValueError(
                f"Blind JSONL line {line_number} is not valid strict JSON"
            ) from None
        if not isinstance(row, dict):
            raise ValueError(f"Blind JSONL line {line_number} must be an object")
        rows.append(row)
    return resolved, payload, rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    _, _, rows = _read_jsonl_document(path)
    return rows


def atomic_json(
    path: Path,
    payload: dict[str, Any],
    *,
    maximum_bytes: int = MAX_BLIND_REPORT_BYTES,
) -> None:
    write_durable_json(
        path,
        payload,
        maximum_bytes=maximum_bytes,
        label="Blind JSON",
    )


def _mutex_path(lock_path: Path, scope: str) -> Path:
    normalized_lock = os.path.normcase(str(lock_path.resolve()))
    identity = hashlib.sha256(
        f"{normalized_lock}\x00{scope}".encode("utf-8")
    ).hexdigest()
    return Path(tempfile.gettempdir()) / "onevoice-blind-locks" / f"{identity}.lock"


@contextmanager
def exclusive_mutex(path: Path) -> Iterator[None]:
    """Hold one crash-safe, non-blocking host mutex for a blind transaction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                "Blind evaluation is already active for this scope"
            ) from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def leakage_values(task: str, rows: list[dict[str, Any]]) -> set[str]:
    """Derive comparison keys from payloads so absent metadata cannot hide overlap."""
    result: set[str] = set()
    for index, row in enumerate(rows, start=1):
        identifier = str(row.get("id") or "").strip()
        if identifier:
            result.add(f"id:{identifier}")
        if task == "mt":
            source = normalize_text(row.get("source_text"))
            target = normalize_text(row.get("target_text"))
            if not source or not target:
                raise ValueError(
                    f"MT leakage comparison row {index} lacks source_text or target_text"
                )
            result.add(f"pair_fingerprint:{fingerprint_text(source + chr(31) + target)}")
            continue

        text_value = normalize_text(row.get("text"))
        if not text_value:
            raise ValueError(f"ASR leakage comparison row {index} lacks text")
        result.add(f"text_fingerprint:{fingerprint_text(text_value)}")
        audio_digest = str(row.get("audio_sha256") or "").strip()
        if not audio_digest:
            audio_value = str(row.get("audio_path") or "").strip()
            if not audio_value:
                raise ValueError(
                    f"ASR leakage comparison row {index} lacks audio_sha256 and audio_path"
                )
            audio_path = Path(audio_value)
            if not audio_path.is_absolute():
                audio_path = ROOT / audio_path
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"ASR leakage comparison audio is missing on row {index}: {audio_path}"
                )
            audio_digest = sha256(audio_path)
        result.add(f"audio_sha256:{audio_digest}")
        for key in ("speaker", "group"):
            value = str(row.get(key) or "").strip()
            if value:
                result.add(f"{key}:{value}")
    return result


def validate_identifiers(task: str, rows: list[dict[str, Any]]) -> None:
    """Fail closed when blind rows cannot be audited for leakage or duplicates."""
    required = (
        ("id", "pair_fingerprint")
        if task == "mt"
        else ("id", "text_fingerprint", "audio_sha256", "speaker", "group")
    )
    unique = ("id", "pair_fingerprint") if task == "mt" else ("id", "audio_sha256")
    for key in required:
        missing = [
            index + 1
            for index, row in enumerate(rows)
            if not str(row.get(key) or "").strip()
        ]
        if missing:
            preview = ", ".join(str(index) for index in missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise ValueError(
                f"Blind {task} manifest is missing {key} on rows {preview}{suffix}"
            )
    for key in unique:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for row in rows:
            value = str(row[key]).strip()
            if value in seen:
                duplicates.add(value)
            seen.add(value)
        if duplicates:
            raise ValueError(
                f"Blind {task} manifest contains duplicate {key}: {len(duplicates)} values"
            )


def validate_content_integrity(task: str, rows: list[dict[str, Any]]) -> None:
    """Recompute blind fingerprints from payloads instead of trusting metadata."""
    for index, row in enumerate(rows, start=1):
        if task == "mt":
            if row.get("source_language") != "en" or row.get("target_language") != "vi":
                raise ValueError(
                    f"Blind mt row {index} must use canonical source_language=en and "
                    "target_language=vi"
                )
            source = normalize_text(row.get("source_text"))
            target = normalize_text(row.get("target_text"))
            if not source or not target:
                raise ValueError(
                    f"Blind mt manifest is missing source_text or target_text on row {index}"
                )
            expected = fingerprint_text(source + "\x1f" + target)
            if row["pair_fingerprint"] != expected:
                raise ValueError(
                    f"Blind mt pair_fingerprint does not match row content on row {index}"
                )
            continue

        if not str(row.get("language") or "").startswith("vi"):
            raise ValueError(f"Blind asr row {index} must declare a Vietnamese language")
        text_value = normalize_text(row.get("text"))
        if not text_value:
            raise ValueError(f"Blind asr manifest is missing text on row {index}")
        if row["text_fingerprint"] != fingerprint_text(text_value):
            raise ValueError(
                f"Blind asr text_fingerprint does not match row content on row {index}"
            )

        audio_value = str(row.get("audio_path") or "").strip()
        if not audio_value:
            raise ValueError(f"Blind asr manifest is missing audio_path on row {index}")
        audio_path = Path(audio_value)
        if not audio_path.is_absolute():
            audio_path = ROOT / audio_path
        if not audio_path.is_file():
            raise FileNotFoundError(f"Blind asr audio is missing on row {index}: {audio_path}")
        if row["audio_sha256"] != sha256(audio_path):
            raise ValueError(
                f"Blind asr audio_sha256 does not match audio content on row {index}"
            )


def validate_required_slices(task: str, rows: list[dict[str, Any]], required: list[str]) -> None:
    observed = set(coverage_counts(task, rows)["slice_samples"])
    missing = sorted(set(required) - observed)
    if missing:
        raise ValueError(f"Blind {task} manifest is missing required slices: {missing}")


def coverage_counts(task: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count each declared blind slice at most once per independent row."""
    counts: Counter[str] = Counter()
    for row in rows:
        observed = {
            str(value).strip().casefold()
            for value in row.get("categories", [])
            if str(value).strip()
        }
        if task == "mt":
            if row.get("source_text") and row.get("target_text"):
                observed.update({"en_to_vi", "vi_to_en"})
            else:
                direction = str(row.get("direction") or "").strip().casefold()
                if direction:
                    observed.add(direction)
        else:
            dimensions = row.get("blind_dimensions") or row.get("selection_dimensions") or {}
            accent = str(dimensions.get("accent_region") or "").strip().casefold()
            role = str(dimensions.get("role") or "").strip().casefold()
            if accent:
                observed.add(accent)
            if role:
                observed.add(role)
            if dimensions.get("noise") is True:
                observed.add("noise")
        counts.update(observed)
    coverage: dict[str, Any] = {
        "rows": len(rows),
        "slice_samples": dict(sorted(counts.items())),
    }
    if task == "asr":
        coverage.update(
            {
                "unique_speakers": len(
                    {
                        str(row.get("speaker") or "").strip()
                        for row in rows
                        if str(row.get("speaker") or "").strip()
                    }
                ),
                "unique_groups": len(
                    {
                        str(row.get("group") or "").strip()
                        for row in rows
                        if str(row.get("group") or "").strip()
                    }
                ),
            }
        )
    return coverage


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Blind coverage policy {label} must be a positive integer")
    return value


def validate_minimum_coverage(
    task: str,
    rows: list[dict[str, Any]],
    policy: dict[str, Any],
    required_slices: list[str],
) -> dict[str, Any]:
    """Fail closed when a blind suite is too small for meaningful slice evidence."""
    if not isinstance(policy, dict):
        raise ValueError(f"Blind {task} minimum coverage policy is missing")
    minimum_rows = _positive_int(policy.get("rows"), f"{task}.rows")
    minimum_slices = policy.get("slice_samples")
    if not isinstance(minimum_slices, dict):
        raise ValueError(f"Blind coverage policy {task}.slice_samples is missing")
    missing_policy = sorted(set(required_slices) - set(minimum_slices))
    if missing_policy:
        raise ValueError(
            f"Blind coverage policy {task} lacks required slice quotas: {missing_policy}"
        )

    normalized_minimums = {
        str(name).strip().casefold(): _positive_int(
            minimum, f"{task}.slice_samples.{name}"
        )
        for name, minimum in minimum_slices.items()
        if str(name).strip()
    }
    coverage = coverage_counts(task, rows)
    deficits: list[str] = []
    if coverage["rows"] < minimum_rows:
        deficits.append(f"rows={coverage['rows']}/{minimum_rows}")
    if task == "asr":
        for name in ("unique_speakers", "unique_groups"):
            if name not in policy:
                continue
            minimum = _positive_int(policy[name], f"{task}.{name}")
            if coverage[name] < minimum:
                deficits.append(f"{name}={coverage[name]}/{minimum}")
    observed = coverage["slice_samples"]
    deficits.extend(
        f"{name}={observed.get(name, 0)}/{minimum}"
        for name, minimum in sorted(normalized_minimums.items())
        if observed.get(name, 0) < minimum
    )
    if deficits:
        raise ValueError(
            f"Blind {task} manifest is below minimum coverage: {', '.join(deficits)}"
        )
    return coverage


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def confidence_interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    lower = finite_float(value[0])
    upper = finite_float(value[1])
    if lower is None or upper is None or lower > upper:
        return None
    return lower, upper


def clinical_failures(report: dict[str, Any], required: list[str]) -> list[str]:
    categories = report.get("categories") or {}
    failures = []
    for name in required:
        record = categories.get(name)
        try:
            samples = int(record.get("samples", 0)) if record else 0
        except (TypeError, ValueError):
            samples = 0
        rate = finite_float(record.get("safety_failure_rate")) if record else None
        if samples < 1 or rate is None or rate != 0.0:
            failures.append(name)
    return failures


def _reported_samples(record: Any) -> int | None:
    try:
        samples = int(record.get("samples")) if isinstance(record, dict) else None
    except (TypeError, ValueError):
        return None
    return samples if samples is not None and samples >= 0 else None


def _casefold_record(records: Any, name: str) -> dict[str, Any] | None:
    if not isinstance(records, dict):
        return None
    target = name.casefold()
    for key, value in records.items():
        if str(key).casefold() == target and isinstance(value, dict):
            return value
    return None


def blind_report_coverage_failures(
    report: dict[str, Any],
    task: str,
    direction: str | None,
    expected_rows: int,
    expected_slices: dict[str, int],
    expected_unique: dict[str, int] | None = None,
) -> list[str]:
    """Bind report sample counts to the immutable blind manifest coverage."""
    metric_view = report
    if task == "mt":
        metric_view = _casefold_record(report.get("directions"), str(direction)) or {}
    failures: list[str] = []
    actual_rows = _reported_samples(metric_view)
    if actual_rows != expected_rows:
        failures.append(f"rows:{actual_rows}/{expected_rows}")
    if task == "asr" and expected_unique:
        for name, expected in sorted(expected_unique.items()):
            try:
                actual = int(report.get(name))
            except (TypeError, ValueError):
                actual = None
            if actual != expected:
                failures.append(f"{name}:{actual}/{expected}")
        if report.get("wer_bootstrap_unit") != "group":
            failures.append("wer_bootstrap_unit:not_group")
        try:
            clusters = int(report.get("wer_bootstrap_clusters"))
        except (TypeError, ValueError):
            clusters = None
        expected_groups = expected_unique.get("unique_groups")
        if clusters != expected_groups:
            failures.append(f"wer_bootstrap_clusters:{clusters}/{expected_groups}")

    categories = report.get("categories") or {}
    slices = report.get("slices") or {}
    for name, expected in sorted(expected_slices.items()):
        if task == "mt" and name in {"en_to_vi", "vi_to_en"}:
            if name != direction:
                continue
            record = _casefold_record(report.get("directions"), name)
        elif task == "asr" and name in {"north", "central", "south"}:
            record = _casefold_record(slices.get("accent"), name)
        elif task == "asr" and name in {"doctor", "patient"}:
            record = _casefold_record(slices.get("role"), name)
        elif task == "asr" and name == "noise":
            record = _casefold_record(slices.get("noise"), "true")
        else:
            record = _casefold_record(categories, name)
        actual = _reported_samples(record)
        if actual != expected:
            failures.append(f"slice:{name}:{actual}/{expected}")
    return failures


def blind_quality_failures(
    report: dict[str, Any],
    task: str,
    direction: str | None,
    accuracy_config: dict[str, Any],
    bakeoff_config: dict[str, Any],
) -> list[str]:
    """Require point estimates and worst-case 95% CI to clear release thresholds."""
    aggregate = accuracy_config["release_gates"]["aggregate"]
    failures: list[str] = []
    metric_view = report
    if task == "mt":
        metric_view = report.get("directions", {}).get(str(direction)) or {}
    try:
        samples = int(metric_view.get("samples", 0))
    except (TypeError, ValueError):
        samples = 0
    if samples < 1:
        failures.append("samples:missing")

    if task == "mt":
        policies = (
            ("sacrebleu", "sacrebleu_bootstrap_95ci", float(aggregate["mt_sacrebleu_min"])),
            ("chrf2", "chrf2_bootstrap_95ci", float(aggregate["mt_chrf2_min"])),
        )
        for metric_name, interval_name, minimum in policies:
            metric = finite_float(metric_view.get(metric_name))
            interval = confidence_interval(metric_view.get(interval_name))
            if metric is None:
                failures.append(f"{metric_name}:missing_or_invalid")
            elif metric < minimum:
                failures.append(f"{metric_name}:below_policy")
            if interval is None:
                failures.append(f"{interval_name}:missing_or_invalid")
            elif interval[0] < minimum:
                failures.append(f"{interval_name}:lower_bound_below_policy")
        return failures

    wer = finite_float(report.get("wer"))
    wer_interval = confidence_interval(report.get("wer_bootstrap_95ci"))
    wer_max = float(aggregate["asr_vi_wer_max"])
    if wer is None:
        failures.append("wer:missing_or_invalid")
    elif wer > wer_max:
        failures.append("wer:above_policy")
    if wer_interval is None:
        failures.append("wer_bootstrap_95ci:missing_or_invalid")
    elif wer_interval[1] > wer_max:
        failures.append("wer_bootstrap_95ci:upper_bound_above_policy")
    if finite_float(report.get("cer")) is None:
        failures.append("cer:missing_or_invalid")
    code_switch = report.get("slices", {}).get("code_switch", {}).get("True")
    code_switch_wer = finite_float(code_switch.get("wer")) if code_switch else None
    if code_switch_wer is None:
        failures.append("code_switch_wer:missing_or_invalid")
    elif code_switch_wer > float(
        bakeoff_config["promotion_gate"]["asr_code_switch_wer_max"]
    ):
        failures.append("code_switch_wer:above_policy")
    return failures


def load_locked_accuracy_config(
    config: dict[str, Any],
    project_root: Path = ROOT,
) -> dict[str, Any]:
    record = config["data"]["accuracy_program"]
    configured_path = Path(str(record["path"]))
    if configured_path.is_absolute():
        path = resolve_regular_file(
            configured_path,
            label="Accuracy policy",
            maximum_bytes=MAX_ACCURACY_CONFIG_BYTES,
        )
    else:
        path = resolve_regular_file_under(
            configured_path,
            project_root=project_root,
            allowed_root=project_root,
            label="Accuracy policy",
            maximum_bytes=MAX_ACCURACY_CONFIG_BYTES,
        )
    payload = read_stable_regular_file(
        path,
        maximum_bytes=MAX_ACCURACY_CONFIG_BYTES,
        label="Accuracy policy",
    )
    actual = hashlib.sha256(payload).hexdigest()
    if actual != record["sha256"]:
        raise ValueError(f"Accuracy policy checksum mismatch: {actual}")
    try:
        loaded = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError):
        raise ValueError("Accuracy policy is not valid UTF-8 YAML") from None
    if not isinstance(loaded, dict):
        raise ValueError("Accuracy policy root must be a mapping")
    return loaded


def ensure_unseen(task: str, blind_rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    blind_values = leakage_values(task, blind_rows)
    comparison_paths = [
        ROOT / config["data"]["train"][task],
        ROOT / config["data"]["selection_dev"][task]["path"],
    ]
    comparison_paths.extend(
        ROOT / path
        for path in config["data"]["forbidden_selection_inputs"]
        if (task == "mt" and "mt" in path) or (task == "asr" and "asr" in path)
    )
    for path in comparison_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        overlap = blind_values & leakage_values(task, read_jsonl(path))
        if overlap:
            raise ValueError(f"Blind {task} leakage against {path}: {len(overlap)} keys")


def lock_suite(lock_path: Path, mt_path: Path, asr_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    lock_path = resolve_regular_file(
        lock_path,
        label="Blind v2 lock",
        maximum_bytes=MAX_BLIND_LOCK_BYTES,
    )
    existing = _load_bounded_json(
        lock_path,
        label="Blind v2 lock",
        maximum_bytes=MAX_BLIND_LOCK_BYTES,
    )
    _validate_lock_envelope(existing)
    if existing.get("version") != 2:
        raise ValueError("Blind v2 lock requires schema version 2")
    if existing.get("status") != "awaiting_unseen_data":
        raise ValueError("Blind v2 is already locked and cannot be replaced")
    manifests = {}
    for task, path in (("mt", mt_path), ("asr", asr_path)):
        resolved, manifest_payload, rows = _read_jsonl_document(path)
        if not rows:
            raise ValueError(f"Blind {task} manifest is empty")
        validate_identifiers(task, rows)
        validate_content_integrity(task, rows)
        ensure_unseen(task, rows, config)
        required_slices = existing["required_slices"][task]
        validate_required_slices(task, rows, required_slices)
        coverage = validate_minimum_coverage(
            task,
            rows,
            existing["minimum_coverage"][task],
            required_slices,
        )
        manifests[task] = {
            "path": str(resolved),
            "rows": len(rows),
            "sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "coverage": coverage,
        }
    locked = {
        **existing,
        "status": "locked_unopened",
        "locked_at": utc_now(),
        "manifests": manifests,
        "selection": None,
        "opened": {"mt": {"en_to_vi": None, "vi_to_en": None}, "asr": None},
    }
    atomic_json(lock_path, locked, maximum_bytes=MAX_BLIND_LOCK_BYTES)
    return locked


def _validate_opened_slots(lock: dict[str, Any]) -> None:
    opened = lock.get("opened")
    if (
        not isinstance(opened, dict)
        or set(opened) != {"mt", "asr"}
        or not isinstance(opened.get("mt"), dict)
        or set(opened["mt"]) != {"en_to_vi", "vi_to_en"}
    ):
        raise ValueError("Blind opened-slot structure is invalid")
    records = [
        ("mt", "en_to_vi", opened["mt"]["en_to_vi"]),
        ("mt", "vi_to_en", opened["mt"]["vi_to_en"]),
        ("asr", None, opened["asr"]),
    ]
    opened_count = sum(record is not None for _, _, record in records)
    expected_status = (
        "locked_unopened"
        if opened_count == 0
        else "opened" if opened_count == len(records) else "partially_opened"
    )
    if lock.get("status") != expected_status:
        raise ValueError("Blind status does not match opened slots")
    selection = lock.get("selection")
    if not opened_count and selection is not None:
        raise ValueError("Blind unopened suite cannot bind a selection")
    if opened_count and not isinstance(selection, dict):
        raise ValueError("Blind opened slots lack a bound selection")
    specification_fields = {
        "candidate",
        "model",
        "revision",
        "adapter",
        "adapter_manifest_sha256",
        "selection_sha256",
        "blind_manifest_sha256",
        "direction",
        "scope",
    }
    for task, direction, record in records:
        if record is None:
            continue
        allowed_fields = specification_fields | {
            "candidate_sha256",
            "opened_at",
            "result",
        }
        if not isinstance(record, dict) or not set(record).issubset(
            allowed_fields
        ):
            raise ValueError(f"Blind opened slot {task}/{direction or 'vi'} is invalid")
        if not specification_fields.issubset(record):
            raise ValueError(f"Blind opened slot {task}/{direction or 'vi'} is incomplete")
        specification = {field: record[field] for field in specification_fields}
        expected_digest = hashlib.sha256(
            json.dumps(
                specification,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if record.get("candidate_sha256") != expected_digest:
            raise ValueError(
                f"Blind opened slot {task}/{direction or 'vi'} candidate digest mismatch"
            )
        if record.get("direction") != direction:
            raise ValueError(
                f"Blind opened slot {task}/{direction or 'vi'} direction mismatch"
            )
        if record.get("scope") not in {"research", "production"}:
            raise ValueError(
                f"Blind opened slot {task}/{direction or 'vi'} scope is invalid"
            )
        if record.get("selection_sha256") != selection.get("sha256"):
            raise ValueError(
                f"Blind opened slot {task}/{direction or 'vi'} selection mismatch"
            )
        if record.get("blind_manifest_sha256") != lock["manifests"][task]["sha256"]:
            raise ValueError(
                f"Blind opened slot {task}/{direction or 'vi'} manifest mismatch"
            )
        for field in ("candidate_sha256", "adapter_manifest_sha256"):
            value = str(record.get(field) or "")
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    f"Blind opened slot {task}/{direction or 'vi'} {field} is invalid"
                )
        try:
            opened_at = datetime.fromisoformat(
                str(record.get("opened_at") or "").replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError(
                f"Blind opened slot {task}/{direction or 'vi'} timestamp is invalid"
            ) from exc
        if opened_at.tzinfo is None:
            raise ValueError(
                f"Blind opened slot {task}/{direction or 'vi'} timestamp is invalid"
            )
        if "result" in record and not isinstance(record["result"], dict):
            raise ValueError(
                f"Blind opened slot {task}/{direction or 'vi'} result is invalid"
            )


def verify_lock(lock_path: Path) -> dict[str, Any]:
    lock_path = resolve_regular_file(
        lock_path,
        label="Blind v2 lock",
        maximum_bytes=MAX_BLIND_LOCK_BYTES,
    )
    lock = _load_bounded_json(
        lock_path,
        label="Blind v2 lock",
        maximum_bytes=MAX_BLIND_LOCK_BYTES,
    )
    _validate_lock_envelope(lock)
    if lock.get("version") != 2:
        raise ValueError("Blind v2 lock requires schema version 2")
    if lock.get("status") not in {"locked_unopened", "partially_opened", "opened"}:
        raise ValueError(f"Blind v2 is not locked: {lock.get('status')!r}")
    for task in ("mt", "asr"):
        record = lock["manifests"].get(task)
        if not isinstance(record, dict) or set(record) != {
            "path",
            "rows",
            "sha256",
            "coverage",
        }:
            raise ValueError(f"Blind lock lacks {task} manifest")
        path = resolve_regular_file(
            record["path"],
            label=f"Blind {task} manifest",
            maximum_bytes=MAX_BLIND_MANIFEST_BYTES,
        )
        digest = str(record.get("sha256") or "")
        _, manifest_payload, rows = _read_jsonl_document(path)
        if (
            not SHA256_RE.fullmatch(digest)
            or hashlib.sha256(manifest_payload).hexdigest() != digest
        ):
            raise ValueError(f"Blind {task} checksum mismatch")
        coverage = validate_minimum_coverage(
            task,
            rows,
            lock["minimum_coverage"][task],
            lock["required_slices"][task],
        )
        if record.get("rows") != len(rows) or record.get("coverage") != coverage:
            raise ValueError(f"Blind {task} coverage record mismatch")
    selection = lock.get("selection")
    if selection:
        if not isinstance(selection, dict) or set(selection) != {"path", "sha256"}:
            raise ValueError("Blind selection comparison record is invalid")
        path = resolve_regular_file(
            selection["path"],
            label="Blind selection comparison",
            maximum_bytes=MAX_BLIND_REPORT_BYTES,
        )
        _, selection_payload = _load_bounded_json_document(
            path,
            label="Blind selection comparison",
            maximum_bytes=MAX_BLIND_REPORT_BYTES,
        )
        digest = str(selection.get("sha256") or "")
        if (
            not SHA256_RE.fullmatch(digest)
            or hashlib.sha256(selection_payload).hexdigest() != digest
        ):
            raise ValueError("Blind selection comparison checksum mismatch")
    _validate_opened_slots(lock)
    return lock


def candidate_from_config(config: dict[str, Any], task: str, candidate_id: str) -> dict[str, Any]:
    for candidate in config["candidates"][task]:
        if candidate["id"] == candidate_id:
            return candidate
    raise ValueError(f"Unknown {task} candidate: {candidate_id}")


def verify_selection_winner(
    comparison_path: Path,
    task: str,
    direction: str | None,
    candidate_id: str,
    adapter: Path,
    config: dict[str, Any],
    scope: str,
) -> dict[str, Any]:
    comparison_path = resolve_regular_file(
        comparison_path,
        label="Blind selection comparison",
        maximum_bytes=MAX_BLIND_REPORT_BYTES,
    )
    comparison, comparison_payload = _load_bounded_json_document(
        comparison_path,
        label="Blind selection comparison",
        maximum_bytes=MAX_BLIND_REPORT_BYTES,
    )
    if comparison.get("status") != "selection_complete":
        raise ValueError("Blind evaluation requires a finalized selection comparison")
    if comparison.get("selection_policy") != selection_policy_record(config):
        raise ValueError("Blind evaluation selection policy is missing, stale, or mismatched")
    if comparison.get("selection_sha256") != configured_selection_hashes(config):
        raise ValueError("Blind evaluation selection inputs are missing, stale, or mismatched")
    if comparison.get("scope") != scope:
        raise ValueError("Blind evaluation scope differs from the selected bake-off")
    key = str(direction or "vi")
    try:
        winner = comparison["results"][task]["winners"][key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Selection comparison lacks winner for {task}/{key}") from exc
    if winner.get("candidate_id") != candidate_id:
        raise ValueError(f"Blind candidate is not the selected winner for {task}/{key}")
    approvals = comparison.get("research_license_approvals")
    if (
        not isinstance(approvals, list)
        or any(not isinstance(value, str) or not value for value in approvals)
        or approvals != sorted(set(approvals))
    ):
        raise ValueError("Blind evaluation research license approvals are invalid")
    expected_decisions = license_decisions(config, set(approvals))
    if comparison.get("license_decisions") != expected_decisions:
        raise ValueError("Blind evaluation license decisions are missing or mismatched")
    decision = expected_decisions.get(candidate_id)
    if not isinstance(decision, dict) or decision.get("gpu_allowed") is not True:
        raise ValueError("Blind candidate lacks the license approval used by selection")
    if decision.get("task") != task:
        raise ValueError("Blind candidate license decision belongs to another task")
    if scope == "production" and decision.get("production_eligible") is not True:
        raise ValueError("Candidate license is not approved for production promotion")
    adapter = adapter.resolve()
    if Path(str(winner.get("adapter") or "")).resolve() != adapter:
        raise ValueError(f"Blind adapter path does not match selected winner for {task}/{key}")
    manifest = adapter_tree_manifest(adapter)
    expected_manifest = str(winner.get("adapter_manifest_sha256") or "")
    if manifest["manifest_sha256"] != expected_manifest:
        raise ValueError(f"Blind adapter checksum does not match selected winner for {task}/{key}")
    return {
        "path": str(comparison_path),
        "sha256": hashlib.sha256(comparison_payload).hexdigest(),
        "winner": winner,
        "adapter_manifest": manifest,
    }


def verify_report_provenance(
    report_path: Path,
    provenance_path: Path,
    specification_sha256: str,
    manifest_sha256: str,
    selection_sha256: str,
) -> dict[str, Any]:
    report_path = resolve_regular_file(
        report_path,
        label="Blind report",
        maximum_bytes=MAX_BLIND_REPORT_BYTES,
    )
    provenance_path = resolve_regular_file(
        provenance_path,
        label="Blind report provenance",
        maximum_bytes=MAX_BLIND_PROVENANCE_BYTES,
    )
    provenance, _ = _load_bounded_json_document(
        provenance_path,
        label="Blind report provenance",
        maximum_bytes=MAX_BLIND_PROVENANCE_BYTES,
    )
    report_payload = read_stable_regular_file(
        report_path,
        maximum_bytes=MAX_BLIND_REPORT_BYTES,
        label="Blind report",
    )
    expected = {
        "specification_sha256": specification_sha256,
        "blind_manifest_sha256": manifest_sha256,
        "selection_sha256": selection_sha256,
        "report_sha256": hashlib.sha256(report_payload).hexdigest(),
        "report_bytes": len(report_payload),
    }
    resource_run = provenance.get("resource_run")
    recorded_report = Path(str(provenance.get("report") or ""))
    if (
        set(provenance) != BLIND_PROVENANCE_FIELDS
        or isinstance(provenance.get("version"), bool)
        or provenance.get("version") != 1
        or Path(os.path.abspath(recorded_report)) != report_path
        or not isinstance(resource_run, dict)
        or isinstance(resource_run.get("return_code"), bool)
        or resource_run.get("return_code") != 0
        or any(provenance.get(key) != value for key, value in expected.items())
    ):
        raise ValueError(f"Blind report provenance mismatch: {report_path}")
    _aware_timestamp(provenance.get("created_at"), label="Blind provenance")
    return provenance


def _load_bounded_json_document(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    payload = read_stable_regular_file(
        path,
        maximum_bytes=maximum_bytes,
        label=label,
    )
    return _parse_json_mapping(payload, label=label), payload


def _load_bounded_json(path: Path, *, label: str, maximum_bytes: int) -> dict[str, Any]:
    parsed, _ = _load_bounded_json_document(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
    )
    return parsed


def _aware_timestamp(value: Any, *, label: str) -> None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} timestamp is invalid")


def _resolved_record_path(value: Any, project_root: Path, *, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{label} path is missing")
    path = Path(raw)
    if not path.is_absolute():
        return resolve_regular_file_under(
            path,
            project_root=project_root,
            allowed_root=project_root,
            label=label,
        )
    project_boundary = Path(os.path.abspath(project_root))
    lexical = Path(os.path.abspath(path))
    if lexical.is_relative_to(project_boundary):
        return resolve_regular_file_under(
            lexical,
            project_root=project_boundary,
            allowed_root=project_boundary,
            label=label,
        )
    return resolve_regular_file(lexical, label=label)


def verify_completed_blind_selection(
    selection_path: Path,
    comparison: dict[str, Any],
    config: dict[str, Any],
    winner_specs: list[tuple[str, str | None, dict[str, Any]]],
    project_root: Path = ROOT,
) -> dict[str, Any]:
    """Verify all immutable blind evidence before deployment work may begin."""
    selection_path = _resolved_record_path(
        selection_path,
        project_root,
        label="Deployment selection comparison",
    )
    loaded_comparison = _load_bounded_json(
        selection_path,
        label="Deployment selection comparison",
        maximum_bytes=MAX_BLIND_REPORT_BYTES,
    )
    if loaded_comparison != comparison:
        raise ValueError("Deployment selection comparison changed while validating")
    if comparison.get("status") not in {"blind_complete", "complete"}:
        raise ValueError("Deployment evidence requires completed blind evaluation")
    scope = comparison.get("scope")
    if scope not in {"research", "production"}:
        raise ValueError("Blind completion scope is invalid")
    if comparison.get("selection_policy") != selection_policy_record(config):
        raise ValueError("Blind completion selection policy is stale or mismatched")
    if comparison.get("selection_sha256") != configured_selection_hashes(config):
        raise ValueError("Blind completion selection inputs are stale or mismatched")
    approvals = comparison.get("research_license_approvals")
    if (
        not isinstance(approvals, list)
        or approvals != sorted(set(approvals))
        or any(not isinstance(value, str) or not value for value in approvals)
    ):
        raise ValueError("Blind completion research license approvals are invalid")
    decisions = license_decisions(config, set(approvals))
    if comparison.get("license_decisions") != decisions:
        raise ValueError("Blind completion license decisions are missing or mismatched")

    snapshot_record = comparison.get("selection_snapshot")
    if not isinstance(snapshot_record, dict) or set(snapshot_record) != {"path", "sha256"}:
        raise ValueError("Blind completion selection snapshot is missing")
    snapshot_path = _resolved_record_path(
        snapshot_record.get("path"),
        project_root,
        label="Selection snapshot",
    )
    snapshot, snapshot_payload = _load_bounded_json_document(
        snapshot_path,
        label="Selection snapshot",
        maximum_bytes=MAX_BLIND_REPORT_BYTES,
    )
    snapshot_sha256 = str(snapshot_record.get("sha256") or "")
    if (
        not SHA256_RE.fullmatch(snapshot_sha256)
        or hashlib.sha256(snapshot_payload).hexdigest() != snapshot_sha256
    ):
        raise ValueError("Selection snapshot checksum mismatch")
    if snapshot.get("status") != "selection_complete":
        raise ValueError("Selection snapshot status is invalid")
    if selection_identity(snapshot) != selection_identity(comparison):
        raise ValueError("Selection snapshot identity differs from deployment comparison")

    lock_value = config.get("data", {}).get("blind_test_v2_lock")
    lock_path = _resolved_record_path(
        lock_value,
        project_root,
        label="Blind v2 lock",
    )
    lock = verify_lock(lock_path)
    if lock.get("status") != "opened":
        raise ValueError("Blind v2 lock is not fully opened")
    lock_selection = lock.get("selection")
    lock_selection_path = _resolved_record_path(
        lock_selection.get("path") if isinstance(lock_selection, dict) else None,
        project_root,
        label="Blind lock selection",
    )
    if (
        not isinstance(lock_selection, dict)
        or lock_selection_path != snapshot_path
        or lock_selection.get("sha256") != snapshot_sha256
    ):
        raise ValueError("Blind v2 lock selection binding mismatch")

    expected_by_binding: dict[
        tuple[str, str | None], tuple[str, str | None, dict[str, Any]]
    ] = {}
    for task, direction, winner in winner_specs:
        candidate_id = str(winner.get("candidate_id") or "")
        binding = (candidate_id, direction)
        if not candidate_id or binding in expected_by_binding:
            raise ValueError("Deployment winners contain duplicate or missing bindings")
        expected_by_binding[binding] = (task, direction, winner)
    results = comparison.get("blind_test_v2")
    if not isinstance(results, list) or len(results) != len(expected_by_binding):
        raise ValueError("Blind completion must contain one result for every winner")

    result_fields = {
        "evaluated_at",
        "candidate",
        "manifest_sha256",
        "selection_sha256",
        "report",
        "report_bytes",
        "report_sha256",
        "report_provenance",
        "critical_gate",
        "failed_slices",
        "coverage_gate",
        "coverage_failures",
        "quality_gate",
        "quality_failures",
        "promotion_allowed",
    }
    candidate_fields = {
        "candidate",
        "model",
        "revision",
        "adapter",
        "adapter_manifest_sha256",
        "selection_sha256",
        "blind_manifest_sha256",
        "direction",
        "scope",
    }
    accuracy_config = load_locked_accuracy_config(config, project_root)
    required_slices = (
        config["promotion_gate"]["critical_slices"]
        + config["promotion_gate"]["policy_slices"]
    )
    observed: set[tuple[str, str | None]] = set()
    for result in results:
        if not isinstance(result, dict) or set(result) != result_fields:
            raise ValueError("Blind completion result schema is invalid")
        candidate_specification = result.get("candidate")
        if (
            not isinstance(candidate_specification, dict)
            or set(candidate_specification) != candidate_fields
        ):
            raise ValueError("Blind completion candidate specification is invalid")
        candidate_id = str(candidate_specification.get("candidate") or "")
        candidate_direction = candidate_specification.get("direction")
        binding = (candidate_id, candidate_direction)
        expected = expected_by_binding.get(binding)
        if expected is None or binding in observed:
            raise ValueError("Blind completion candidate set is invalid")
        observed.add(binding)
        task, direction, winner = expected
        decision = decisions.get(candidate_id)
        if (
            not isinstance(decision, dict)
            or decision.get("task") != task
            or decision.get("gpu_allowed") is not True
            or (scope == "production" and decision.get("production_eligible") is not True)
        ):
            raise ValueError(f"Blind completion license gate failed: {candidate_id}")
        configured = candidate_from_config(config, task, candidate_id)
        if (
            candidate_specification.get("model") != configured.get("model")
            or candidate_specification.get("revision") != configured.get("revision")
            or candidate_specification.get("direction") != direction
            or candidate_specification.get("scope") != scope
            or Path(str(candidate_specification.get("adapter") or "")).resolve()
            != Path(str(winner.get("adapter") or "")).resolve()
            or candidate_specification.get("adapter_manifest_sha256")
            != winner.get("adapter_manifest_sha256")
            or candidate_specification.get("selection_sha256") != snapshot_sha256
        ):
            raise ValueError(f"Blind completion winner binding mismatch: {candidate_id}")
        manifest_sha256 = str(result.get("manifest_sha256") or "")
        if (
            not SHA256_RE.fullmatch(manifest_sha256)
            or candidate_specification.get("blind_manifest_sha256") != manifest_sha256
            or result.get("selection_sha256") != snapshot_sha256
        ):
            raise ValueError(f"Blind completion manifest binding mismatch: {candidate_id}")
        if (
            result.get("critical_gate") != "pass"
            or result.get("coverage_gate") != "pass"
            or result.get("quality_gate") != "pass"
            or result.get("failed_slices") != []
            or result.get("coverage_failures") != []
            or result.get("quality_failures") != []
            or result.get("promotion_allowed") is not True
        ):
            raise ValueError(f"Blind completion gate did not pass: {candidate_id}")
        _aware_timestamp(result.get("evaluated_at"), label="Blind result")

        report_path = _resolved_record_path(
            result.get("report"),
            project_root,
            label="Blind report",
        )
        report_payload, report_document = _load_bounded_json_document(
            report_path,
            label="Blind report",
            maximum_bytes=MAX_BLIND_REPORT_BYTES,
        )
        if (
            result.get("report_bytes") != len(report_document)
            or result.get("report_sha256")
            != hashlib.sha256(report_document).hexdigest()
        ):
            raise ValueError(f"Blind report checksum mismatch: {candidate_id}")
        manifest_record = lock["manifests"][task]
        expected_slices = {
            name: int(manifest_record["coverage"]["slice_samples"][name])
            for name in lock["required_slices"][task]
        }
        recomputed_clinical = clinical_failures(report_payload, required_slices)
        recomputed_coverage = blind_report_coverage_failures(
            report_payload,
            task,
            direction,
            int(manifest_record["rows"]),
            expected_slices,
            {
                name: int(manifest_record["coverage"][name])
                for name in ("unique_speakers", "unique_groups")
                if name in manifest_record["coverage"]
            }
            if task == "asr"
            else None,
        )
        recomputed_quality = blind_quality_failures(
            report_payload,
            task,
            direction,
            accuracy_config,
            config,
        )
        if (
            recomputed_clinical != result.get("failed_slices")
            or recomputed_coverage != result.get("coverage_failures")
            or recomputed_quality != result.get("quality_failures")
        ):
            raise ValueError(f"Blind report gate recomputation mismatch: {candidate_id}")
        provenance_path = _resolved_record_path(
            result.get("report_provenance"),
            project_root,
            label="Blind report provenance",
        )
        provenance = _load_bounded_json(
            provenance_path,
            label="Blind report provenance",
            maximum_bytes=MAX_BLIND_PROVENANCE_BYTES,
        )
        resource_run = provenance.get("resource_run")
        if (
            set(provenance) != BLIND_PROVENANCE_FIELDS
            or isinstance(provenance.get("version"), bool)
            or provenance.get("version") != 1
            or provenance.get("specification_sha256")
            != hashlib.sha256(
                json.dumps(
                    candidate_specification,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            or provenance.get("blind_manifest_sha256") != manifest_sha256
            or provenance.get("selection_sha256") != snapshot_sha256
            or Path(str(provenance.get("report") or "")).resolve() != report_path
            or provenance.get("report_bytes") != result.get("report_bytes")
            or provenance.get("report_sha256") != result.get("report_sha256")
            or not isinstance(resource_run, dict)
            or isinstance(resource_run.get("return_code"), bool)
            or resource_run.get("return_code") != 0
        ):
            raise ValueError(f"Blind report provenance mismatch: {candidate_id}")
        _aware_timestamp(provenance.get("created_at"), label="Blind provenance")

        lock_slot, lock_key = opening_slot(lock, task, direction)
        lock_record = lock_slot.get(lock_key)
        if not isinstance(lock_record, dict) or lock_record.get("result") != result:
            raise ValueError(f"Blind lock result mismatch: {candidate_id}")
    if observed != set(expected_by_binding):
        raise ValueError("Blind completion winner set is incomplete")
    persisted_lock, persisted_lock_payload = _load_bounded_json_document(
        lock_path,
        label="Blind v2 lock",
        maximum_bytes=MAX_BLIND_LOCK_BYTES,
    )
    if persisted_lock != lock:
        raise RuntimeError("Blind v2 lock changed while validating completion")
    return {
        "selection_snapshot_sha256": snapshot_sha256,
        "blind_lock_sha256": hashlib.sha256(persisted_lock_payload).hexdigest(),
        "winner_count": len(observed),
    }


def opening_slot(lock: dict[str, Any], task: str, direction: str | None) -> tuple[dict[str, Any], str]:
    if task == "mt":
        if direction not in {"en_to_vi", "vi_to_en"}:
            raise ValueError("MT blind evaluation requires one explicit direction")
        return lock["opened"]["mt"], str(direction)
    return lock["opened"], "asr"


def _evaluate_slot(
    args: argparse.Namespace, lock_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    lock = verify_lock(lock_path)
    _, key = opening_slot(lock, args.task, args.direction)
    candidate = candidate_from_config(config, args.task, args.candidate)
    if args.scope == "production" and not candidate["license"].get("production_eligible"):
        raise ValueError("Candidate license is not approved for production promotion")
    selection = verify_selection_winner(
        args.selection_comparison,
        args.task,
        args.direction,
        args.candidate,
        args.adapter,
        config,
        args.scope,
    )
    selection_record = {
        "path": selection["path"],
        "sha256": selection["sha256"],
    }
    specification = {
        "candidate": args.candidate,
        "model": candidate["model"],
        "revision": candidate["revision"],
        "adapter": str(args.adapter.resolve()),
        "adapter_manifest_sha256": selection["adapter_manifest"]["manifest_sha256"],
        "selection_sha256": selection["sha256"],
        "blind_manifest_sha256": lock["manifests"][args.task]["sha256"],
        "direction": args.direction,
        "scope": args.scope,
    }
    digest = hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"blind_v2_{args.task}_{key}_{args.candidate}"
    report_path = output_dir / f"{stem}.json"
    provenance_path = output_dir / f"{stem}_provenance.json"
    if report_path.is_file():
        verify_report_provenance(
            report_path,
            provenance_path,
            digest,
            specification["blind_manifest_sha256"],
            selection["sha256"],
        )

    with exclusive_mutex(_mutex_path(lock_path, "state")):
        lock = verify_lock(lock_path)
        existing_selection = lock.get("selection")
        if existing_selection is not None and existing_selection != selection_record:
            raise ValueError("Blind suite was already bound to another selection comparison")
        slot, key = opening_slot(lock, args.task, args.direction)
        previous = slot.get(key)
        if previous and previous.get("candidate_sha256") != digest:
            raise ValueError(
                f"Blind slot {args.task}/{key} was already opened for another candidate"
            )
        if previous is None:
            lock["selection"] = selection_record
            slot[key] = {
                **specification,
                "candidate_sha256": digest,
                "opened_at": utc_now(),
            }
            mt_done = all(lock["opened"]["mt"].values())
            lock["status"] = (
                "opened"
                if mt_done and lock["opened"]["asr"]
                else "partially_opened"
            )
            atomic_json(lock_path, lock, maximum_bytes=MAX_BLIND_LOCK_BYTES)

    if not report_path.is_file():
        command = [
            args.python,
            str(ROOT / "scripts/run_baseline_benchmarks.py"),
            "--task",
            args.task,
            "--model",
            candidate["model"],
            "--model-revision",
            candidate["revision"],
            "--adapter",
            str(args.adapter.resolve()),
            "--manifest",
            lock["manifests"][args.task]["path"],
            "--samples",
            "0",
            "--device",
            "cuda",
            "--precision",
            "bf16",
            "--gpu-memory-fraction",
            "0.35",
            "--batch-size",
            "4",
            "--num-beams",
            "1",
            "--name",
            stem,
            "--output-dir",
            str(output_dir),
            "--resume-scoring",
        ]
        if args.task == "mt":
            command += [
                "--mt-model-family",
                candidate["model_family"],
                "--mt-direction",
                str(args.direction),
            ]
        else:
            command += ["--language", "vi"]
        from scripts.run_gpu_rounds import monitor_process, wait_for_gpu_spawn_capacity

        log_path = output_dir / f"{stem}.log"
        spawn_capacity = wait_for_gpu_spawn_capacity(
            config["resources"],
            output_dir / f"{stem}_pre_spawn.jsonl",
        )
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            monitored = monitor_process(
                process,
                output_dir / f"{stem}_resources.jsonl",
                config["resources"],
            )
        monitored["spawn_capacity"] = spawn_capacity
        if monitored["return_code"]:
            raise RuntimeError(
                f"Blind benchmark failed with {monitored['return_code']}; see {log_path}"
            )
        if not report_path.is_file():
            raise FileNotFoundError(f"Blind benchmark did not create report: {report_path}")
        completed_report = read_stable_regular_file(
            report_path,
            maximum_bytes=MAX_BLIND_REPORT_BYTES,
            label="Blind report",
        )
        atomic_json(
            provenance_path,
            {
                "version": 1,
                "created_at": utc_now(),
                "specification_sha256": digest,
                "blind_manifest_sha256": specification["blind_manifest_sha256"],
                "selection_sha256": selection["sha256"],
                "report": str(report_path),
                "report_bytes": len(completed_report),
                "report_sha256": hashlib.sha256(completed_report).hexdigest(),
                "resource_run": monitored,
            },
            maximum_bytes=MAX_BLIND_PROVENANCE_BYTES,
        )
    provenance = verify_report_provenance(
        report_path,
        provenance_path,
        digest,
        specification["blind_manifest_sha256"],
        selection["sha256"],
    )
    if adapter_tree_manifest(args.adapter)["manifest_sha256"] != specification[
        "adapter_manifest_sha256"
    ]:
        raise ValueError("Blind adapter changed during evaluation")
    report, report_payload = _load_bounded_json_document(
        report_path,
        label="Blind report",
        maximum_bytes=MAX_BLIND_REPORT_BYTES,
    )
    if (
        len(report_payload) != provenance.get("report_bytes")
        or hashlib.sha256(report_payload).hexdigest()
        != provenance.get("report_sha256")
    ):
        raise RuntimeError("Blind report changed after provenance verification")
    required = config["promotion_gate"]["critical_slices"] + config["promotion_gate"]["policy_slices"]
    failed = clinical_failures(report, required)
    manifest_record = lock["manifests"][args.task]
    expected_slices = {
        name: int(manifest_record["coverage"]["slice_samples"][name])
        for name in lock["required_slices"][args.task]
    }
    coverage_failed = blind_report_coverage_failures(
        report,
        args.task,
        args.direction,
        int(manifest_record["rows"]),
        expected_slices,
        {
            name: int(manifest_record["coverage"][name])
            for name in ("unique_speakers", "unique_groups")
            if name in manifest_record["coverage"]
        }
        if args.task == "asr"
        else None,
    )
    accuracy_config = load_locked_accuracy_config(config)
    quality_failed = blind_quality_failures(
        report,
        args.task,
        args.direction,
        accuracy_config,
        config,
    )
    result = {
        "evaluated_at": utc_now(),
        "candidate": specification,
        "manifest_sha256": lock["manifests"][args.task]["sha256"],
        "selection_sha256": selection["sha256"],
        "report": str(report_path),
        "report_bytes": provenance["report_bytes"],
        "report_sha256": provenance["report_sha256"],
        "report_provenance": str(provenance_path),
        "critical_gate": "pass" if not failed else "fail",
        "failed_slices": failed,
        "coverage_gate": "pass" if not coverage_failed else "fail",
        "coverage_failures": coverage_failed,
        "quality_gate": "pass" if not quality_failed else "fail",
        "quality_failures": quality_failed,
        "promotion_allowed": not failed and not coverage_failed and not quality_failed,
    }
    atomic_json(
        output_dir / f"{stem}_gate.json",
        result,
        maximum_bytes=MAX_BLIND_REPORT_BYTES,
    )
    with exclusive_mutex(_mutex_path(lock_path, "state")):
        lock = verify_lock(lock_path)
        slot, key = opening_slot(lock, args.task, args.direction)
        current = slot.get(key)
        if not isinstance(current, dict) or current.get("candidate_sha256") != digest:
            raise ValueError(f"Blind slot {args.task}/{key} changed during evaluation")
        current["result"] = result
        mt_done = all(lock["opened"]["mt"].values())
        lock["status"] = (
            "opened" if mt_done and lock["opened"]["asr"] else "partially_opened"
        )
        atomic_json(lock_path, lock, maximum_bytes=MAX_BLIND_LOCK_BYTES)
    return result


def evaluate(
    args: argparse.Namespace, lock_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate one blind slot while preventing concurrent duplicate opens."""
    if args.task == "mt" and args.direction not in {"en_to_vi", "vi_to_en"}:
        raise ValueError("MT blind evaluation requires one explicit direction")
    key = str(args.direction) if args.task == "mt" else "asr"
    scope = f"slot-{args.task}-{key}"
    with exclusive_mutex(_mutex_path(lock_path, scope)):
        return _evaluate_slot(args, lock_path, config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["lock", "verify", "evaluate"], required=True)
    parser.add_argument("--lock", type=Path, default=ROOT / "data/eval/blind_test_v2.lock.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/model_bakeoff.yaml")
    parser.add_argument("--mt-manifest", type=Path)
    parser.add_argument("--asr-manifest", type=Path)
    parser.add_argument("--task", choices=["mt", "asr"])
    parser.add_argument("--direction", choices=["en_to_vi", "vi_to_en"])
    parser.add_argument("--candidate")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--selection-comparison", type=Path)
    parser.add_argument("--scope", choices=["research", "production"], default="research")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/reports/model_bakeoff/blind_v2")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    from scripts.run_model_bakeoff import load_config as load_bakeoff_config

    config = load_bakeoff_config(args.config)
    if args.action == "lock":
        if not args.mt_manifest or not args.asr_manifest:
            parser.error("--action lock requires --mt-manifest and --asr-manifest")
        result = lock_suite(args.lock, args.mt_manifest, args.asr_manifest, config)
    elif args.action == "verify":
        result = verify_lock(args.lock)
    else:
        if not args.task or not args.candidate or not args.adapter or not args.selection_comparison:
            parser.error(
                "--action evaluate requires --task, --candidate, --adapter and "
                "--selection-comparison"
            )
        if not args.adapter.is_dir():
            parser.error("--adapter must be a directory")
        result = evaluate(args, args.lock, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

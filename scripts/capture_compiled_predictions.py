"""Capture immutable provenance for predictions from one compiled artifact.

The runtime command is executed as an argv vector without a shell.  It must
contain the exact ``{artifact}``, ``{manifest}`` and ``{output_predictions}``
placeholder tokens so the resulting provenance can prove which files were
used.  Standard output and error are discarded to avoid leaking clinical
text; failures leave no reusable prediction checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.candidate_evidence import canonical_sha256  # noqa: E402
from scripts.run_model_bakeoff import SHA256_RE, sha256  # noqa: E402


SCHEMA_VERSION = 1
EVIDENCE_SOURCE = "onevoice_compiled_prediction_capture"
RUNTIME_KINDS = {"qnn_context_binary", "qnn_model_library", "onnxruntime_qnn"}
PLACEHOLDERS = ("{artifact}", "{manifest}", "{output_predictions}")
MAX_COMMAND_ARGUMENTS = 256
MAX_COMMAND_ARGUMENT_BYTES = 4_096
MAX_COMMAND_BYTES = 32_768
MAX_JSON_BYTES = 2_000_000
MAX_JSONL_BYTES = 250_000_000
MAX_JSONL_LINE_BYTES = 2_000_000
MAX_JSONL_ROWS = 100_000
MAX_TIMEOUT_SECONDS = 86_400.0
PROCESS_TERMINATION_GRACE_SECONDS = 5.0
CANDIDATE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _regular_file(path: Path, *, label: str, maximum_bytes: int) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is missing or not a regular file: {path}")
    size = path.stat().st_size
    if size < 1 or size > maximum_bytes:
        raise ValueError(f"{label} size is outside the valid range")


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _regular_file(path, label=label, maximum_bytes=MAX_JSONL_BYTES)
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
                if len(rows) > MAX_JSONL_ROWS:
                    raise ValueError(f"{label} has too many rows")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSONL") from exc
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def normalize_command_template(command: Sequence[str]) -> tuple[str, ...]:
    """Validate one bounded argv template and its required file bindings."""
    normalized = list(command)
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    if not normalized:
        raise ValueError("compiled runtime command is required after --")
    if len(normalized) > MAX_COMMAND_ARGUMENTS:
        raise ValueError("compiled runtime command has too many arguments")
    total_bytes = 0
    for argument in normalized:
        if not isinstance(argument, str) or not argument or "\x00" in argument:
            raise ValueError("compiled runtime command contains an invalid argument")
        argument_bytes = len(argument.encode("utf-8"))
        if argument_bytes > MAX_COMMAND_ARGUMENT_BYTES:
            raise ValueError("compiled runtime command argument is too long")
        total_bytes += argument_bytes
    if total_bytes > MAX_COMMAND_BYTES:
        raise ValueError("compiled runtime command is too long")
    for placeholder in PLACEHOLDERS:
        if normalized.count(placeholder) != 1:
            raise ValueError(
                f"compiled runtime command must contain {placeholder} exactly once"
            )
    return tuple(normalized)


def resolve_command_template(
    command: Sequence[str],
    *,
    artifact_path: Path,
    manifest_path: Path,
    output_predictions_path: Path,
) -> tuple[str, ...]:
    replacements = {
        "{artifact}": str(artifact_path.resolve()),
        "{manifest}": str(manifest_path.resolve()),
        "{output_predictions}": str(output_predictions_path.resolve()),
    }
    return tuple(replacements.get(argument, argument) for argument in command)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if platform.system() == "Windows":
        process.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    if platform.system() == "Windows":
        process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)


def execute_command(command: Sequence[str], timeout_seconds: float) -> int:
    """Execute a compiled inference command without shell interpretation."""
    process = subprocess.Popen(
        tuple(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=False,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        raise TimeoutError("compiled runtime command timed out") from exc


def _validate_prediction_coverage(
    manifest_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    task: str,
    direction: str | None,
) -> None:
    manifest_ids = [str(row.get("id") or "").strip() for row in manifest_rows]
    prediction_ids = [str(row.get("id") or "").strip() for row in prediction_rows]
    if (
        any(not identifier for identifier in manifest_ids)
        or len(manifest_ids) != len(set(manifest_ids))
    ):
        raise ValueError("manifest contains missing or duplicate ids")
    if (
        any(not identifier for identifier in prediction_ids)
        or len(prediction_ids) != len(set(prediction_ids))
    ):
        raise ValueError("compiled predictions contain missing or duplicate ids")
    if set(manifest_ids) != set(prediction_ids) or len(manifest_ids) != len(
        prediction_ids
    ):
        raise ValueError("compiled predictions do not cover the exact manifest")
    for row in prediction_rows:
        hypothesis = row.get("hypothesis")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError("compiled predictions contain an empty hypothesis")
        if task == "mt" and row.get("direction") != direction:
            raise ValueError("compiled MT prediction direction mismatch")


def _remove_failed_output(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite provenance: {path}")
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if len(serialized.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("compiled prediction provenance is too large")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_compiled_predictions(
    *,
    task: str,
    direction: str | None,
    candidate_id: str,
    adapter_manifest_sha256: str,
    runtime_kind: str,
    artifact_path: Path,
    manifest_path: Path,
    output_predictions_path: Path,
    output_provenance_path: Path,
    num_beams: int,
    timeout_seconds: float,
    command: Sequence[str],
    executor: Callable[[Sequence[str], float], int] = execute_command,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run one compiled inference command and seal its exact file identities."""
    if task not in {"mt", "asr"}:
        raise ValueError("task must be mt or asr")
    if task == "mt" and direction not in {"en_to_vi", "vi_to_en"}:
        raise ValueError("MT capture requires en_to_vi or vi_to_en direction")
    if task == "asr" and direction is not None:
        raise ValueError("ASR capture direction must be omitted")
    if not CANDIDATE_RE.fullmatch(str(candidate_id or "")):
        raise ValueError("candidate_id is invalid")
    if not SHA256_RE.fullmatch(str(adapter_manifest_sha256 or "")):
        raise ValueError("adapter_manifest_sha256 is invalid")
    if runtime_kind not in RUNTIME_KINDS:
        raise ValueError("runtime_kind is invalid")
    if isinstance(num_beams, bool) or not isinstance(num_beams, int) or num_beams < 1:
        raise ValueError("num_beams must be a positive integer")
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_seconds must be a finite number") from exc
    if not math.isfinite(timeout) or timeout <= 0 or timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError("timeout_seconds is outside the valid range")

    _regular_file(artifact_path, label="Compiled artifact", maximum_bytes=sys.maxsize)
    manifest_rows = _read_jsonl(manifest_path, label="Manifest")
    if output_predictions_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite predictions: {output_predictions_path}"
        )
    if output_provenance_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite provenance: {output_provenance_path}"
        )
    template = normalize_command_template(command)
    resolved = resolve_command_template(
        template,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        output_predictions_path=output_predictions_path,
    )
    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    output_provenance_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = now()
    if started_at.tzinfo is None:
        raise ValueError("capture timestamp must include a timezone")
    started = monotonic()
    try:
        return_code = executor(resolved, timeout)
        if isinstance(return_code, bool) or not isinstance(return_code, int):
            raise RuntimeError("compiled runtime returned an invalid status")
        if return_code != 0:
            raise RuntimeError(f"compiled runtime failed with return code {return_code}")
        prediction_rows = _read_jsonl(
            output_predictions_path,
            label="Compiled predictions",
        )
        _validate_prediction_coverage(
            manifest_rows,
            prediction_rows,
            task=task,
            direction=direction,
        )
    except BaseException:
        _remove_failed_output(output_predictions_path)
        raise
    completed_at = now()
    if completed_at.tzinfo is None:
        _remove_failed_output(output_predictions_path)
        raise ValueError("capture timestamp must include a timezone")
    duration_seconds = monotonic() - started
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        _remove_failed_output(output_predictions_path)
        raise ValueError("capture duration is invalid")

    artifact_resolved = artifact_path.resolve()
    manifest_resolved = manifest_path.resolve()
    predictions_resolved = output_predictions_path.resolve()
    command_payload = {
        "template": list(template),
        "template_sha256": canonical_sha256(list(template)),
        "resolved_argv": list(resolved),
        "resolved_sha256": canonical_sha256(list(resolved)),
        "executable": Path(resolved[0]).name,
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "evidence_source": EVIDENCE_SOURCE,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": duration_seconds,
        "runtime_kind": runtime_kind,
        "task": task,
        "direction": direction,
        "candidate_id": candidate_id,
        "adapter_manifest_sha256": adapter_manifest_sha256,
        "decoding": {"num_beams": num_beams, "do_sample": False},
        "artifact": {
            "path": str(artifact_resolved),
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256(artifact_path),
        },
        "manifest": {
            "path": str(manifest_resolved),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256(manifest_path),
        },
        "predictions": {
            "path": str(predictions_resolved),
            "bytes": output_predictions_path.stat().st_size,
            "sha256": sha256(output_predictions_path),
            "rows": len(prediction_rows),
        },
        "command": command_payload,
        "return_code": 0,
    }
    try:
        failures = compiled_prediction_provenance_failures(provenance)
        if failures:
            raise RuntimeError(
                "compiled prediction provenance is inconsistent: " + ", ".join(failures)
            )
        _write_exclusive_json(output_provenance_path, provenance)
    except BaseException:
        _remove_failed_output(output_predictions_path)
        raise
    return provenance


def _aware_timestamp(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def compiled_prediction_provenance_failures(
    provenance: Any,
    *,
    expected: dict[str, Any] | None = None,
) -> list[str]:
    """Validate a compiled prediction capture and optional external bindings."""
    if not isinstance(provenance, dict):
        return ["payload_invalid"]
    failures: list[str] = []
    fields = {
        "schema_version",
        "evidence_source",
        "started_at",
        "completed_at",
        "duration_seconds",
        "runtime_kind",
        "task",
        "direction",
        "candidate_id",
        "adapter_manifest_sha256",
        "decoding",
        "artifact",
        "manifest",
        "predictions",
        "command",
        "return_code",
    }
    if set(provenance) != fields:
        failures.append("fields_invalid")
    if provenance.get("schema_version") != SCHEMA_VERSION:
        failures.append("schema_version_invalid")
    if provenance.get("evidence_source") != EVIDENCE_SOURCE:
        failures.append("evidence_source_invalid")
    started_at = _aware_timestamp(provenance.get("started_at"))
    completed_at = _aware_timestamp(provenance.get("completed_at"))
    if started_at is None:
        failures.append("started_at_invalid")
    if completed_at is None:
        failures.append("completed_at_invalid")
    if (
        started_at is not None
        and completed_at is not None
        and completed_at < started_at
    ):
        failures.append("timestamp_order_invalid")
    duration = provenance.get("duration_seconds")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) < 0
    ):
        failures.append("duration_seconds_invalid")
    if provenance.get("runtime_kind") not in RUNTIME_KINDS:
        failures.append("runtime_kind_invalid")
    task = provenance.get("task")
    direction = provenance.get("direction")
    if task not in {"mt", "asr"}:
        failures.append("task_invalid")
    elif task == "mt" and direction not in {"en_to_vi", "vi_to_en"}:
        failures.append("direction_invalid")
    elif task == "asr" and direction is not None:
        failures.append("direction_invalid")
    if not CANDIDATE_RE.fullmatch(str(provenance.get("candidate_id") or "")):
        failures.append("candidate_id_invalid")
    if not SHA256_RE.fullmatch(
        str(provenance.get("adapter_manifest_sha256") or "")
    ):
        failures.append("adapter_manifest_sha256_invalid")
    if provenance.get("return_code") != 0:
        failures.append("return_code_invalid")

    decoding = provenance.get("decoding")
    if not isinstance(decoding, dict) or set(decoding) != {"num_beams", "do_sample"}:
        failures.append("decoding_invalid")
    elif (
        isinstance(decoding.get("num_beams"), bool)
        or not isinstance(decoding.get("num_beams"), int)
        or decoding["num_beams"] < 1
        or decoding.get("do_sample") is not False
    ):
        failures.append("decoding_invalid")

    for field in ("artifact", "manifest"):
        record = provenance.get(field)
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            failures.append(f"{field}_invalid")
            continue
        if not isinstance(record.get("path"), str) or not record["path"]:
            failures.append(f"{field}_path_invalid")
        if not SHA256_RE.fullmatch(str(record.get("sha256") or "")):
            failures.append(f"{field}_sha256_invalid")
        size = record.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            failures.append(f"{field}_bytes_invalid")
    predictions = provenance.get("predictions")
    if not isinstance(predictions, dict) or set(predictions) != {
        "path",
        "bytes",
        "sha256",
        "rows",
    }:
        failures.append("predictions_invalid")
    else:
        if not isinstance(predictions.get("path"), str) or not predictions["path"]:
            failures.append("predictions_path_invalid")
        if not SHA256_RE.fullmatch(str(predictions.get("sha256") or "")):
            failures.append("predictions_sha256_invalid")
        for field in ("bytes", "rows"):
            value = predictions.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                failures.append(f"predictions_{field}_invalid")

    command = provenance.get("command")
    if not isinstance(command, dict) or set(command) != {
        "template",
        "template_sha256",
        "resolved_argv",
        "resolved_sha256",
        "executable",
    }:
        failures.append("command_invalid")
    else:
        try:
            template = normalize_command_template(command.get("template", []))
        except (TypeError, ValueError):
            failures.append("command_template_invalid")
            template = ()
        resolved = command.get("resolved_argv")
        if not isinstance(resolved, list) or not all(
            isinstance(value, str) and value for value in resolved
        ):
            failures.append("command_resolved_argv_invalid")
            resolved = []
        if template and command.get("template_sha256") != canonical_sha256(list(template)):
            failures.append("command_template_sha256_invalid")
        if resolved and command.get("resolved_sha256") != canonical_sha256(resolved):
            failures.append("command_resolved_sha256_invalid")
        if resolved and command.get("executable") != Path(resolved[0]).name:
            failures.append("command_executable_invalid")
        records = [
            provenance.get("artifact"),
            provenance.get("manifest"),
            provenance.get("predictions"),
        ]
        if template and resolved and all(isinstance(record, dict) for record in records):
            rebuilt = resolve_command_template(
                template,
                artifact_path=Path(records[0].get("path", "")),
                manifest_path=Path(records[1].get("path", "")),
                output_predictions_path=Path(records[2].get("path", "")),
            )
            if list(rebuilt) != resolved:
                failures.append("command_resolved_argv_mismatch")

    if expected is not None:
        for field in (
            "task",
            "direction",
            "candidate_id",
            "adapter_manifest_sha256",
            "decoding",
        ):
            if provenance.get(field) != expected.get(field):
                failures.append(f"{field}_mismatch")
        for field in ("artifact", "manifest", "predictions"):
            record = provenance.get(field)
            expected_record = expected.get(field)
            if not isinstance(record, dict) or not isinstance(expected_record, dict):
                failures.append(f"{field}_mismatch")
                continue
            for identity in ("bytes", "sha256"):
                if record.get(identity) != expected_record.get(identity):
                    failures.append(f"{field}_{identity}_mismatch")
            if field == "predictions" and record.get("rows") != expected_record.get(
                "rows"
            ):
                failures.append("predictions_rows_mismatch")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["mt", "asr"], required=True)
    parser.add_argument("--direction", choices=["en_to_vi", "vi_to_en"])
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--adapter-manifest-sha256", required=True)
    parser.add_argument("--runtime-kind", choices=sorted(RUNTIME_KINDS), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--output-provenance", type=Path, required=True)
    parser.add_argument("--num-beams", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3_600.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    provenance = capture_compiled_predictions(
        task=args.task,
        direction=args.direction,
        candidate_id=args.candidate_id,
        adapter_manifest_sha256=args.adapter_manifest_sha256,
        runtime_kind=args.runtime_kind,
        artifact_path=args.artifact,
        manifest_path=args.manifest,
        output_predictions_path=args.output_predictions,
        output_provenance_path=args.output_provenance,
        num_beams=args.num_beams,
        timeout_seconds=args.timeout_seconds,
        command=args.command,
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

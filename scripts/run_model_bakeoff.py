"""Run the post-Candidate-A multi-model bake-off without touching the active job.

The runner is fail-closed and resumable.  It first verifies that the existing
GPU program is complete, freezes Candidate A by hashing its artifacts, then
runs one GPU child at a time through zero-shot, 400-step pilot, 2,000-step
semifinal and full-train stages.  The current locked tests are never used for
model selection; only the deterministic selection-dev manifests are opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_gpu_program import verified_training_result  # noqa: E402
from scripts.run_gpu_rounds import (  # noqa: E402
    validate_resource_limits,
    wait_for_gpu_spawn_capacity,
)
from scripts.verify_checkpoint_download import verify_checkpoint_download  # noqa: E402
from scripts.watch_checkpoints import (  # noqa: E402
    checkpoint_file_records,
    completed_checkpoint,
)
from scripts.candidate_evidence import (  # noqa: E402
    evidence_sidecars,
    verify_evidence_archive,
)
from scripts.create_verified_backup import (  # noqa: E402
    MANIFEST_NAME as BACKUP_MANIFEST_NAME,
    SCHEMA_VERSION as BACKUP_SCHEMA_VERSION,
    build_release_evidence,
    create_backup,
    verify as verify_backup,
)
from src.data.quality import fingerprint_text  # noqa: E402
from src.pipeline.evidence_paths import (  # noqa: E402
    resolve_regular_directory_under,
    resolve_regular_file_under,
    resolve_regular_file_without_links,
)
from src.pipeline.durable_json import write_durable_json  # noqa: E402
from src.pipeline.license_policy import license_decisions, license_gate  # noqa: E402
from src.pipeline.selection_policy import (  # noqa: E402
    selection_identity,
    selection_identity_sha256,
    selection_policy_record,
)
from src.utils.bounded_file import (  # noqa: E402
    read_stable_regular_file,
    sha256_stable_regular_file,
)


CRITICAL_EXIT_WAITING_FOR_BLIND = 3
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
GIT_CHECK_TIMEOUT_SECONDS = 30.0
MAX_DEPLOYMENT_MEASUREMENT_BYTES = 10_000_000
MAX_DEPLOYMENT_ARTIFACT_BYTES = 16_000_000_000
MAX_QUANTIZATION_PARITY_EVIDENCE_BYTES = 1_000_000
MAX_IDENTITY_EVIDENCE_BYTES = 1_000_000
MAX_BAKEOFF_CONFIG_BYTES = 1_000_000
MAX_BAKEOFF_CONFIG_DEPTH = 32
MAX_BAKEOFF_CONFIG_NODES = 10_000
MAX_BAKEOFF_STATE_BYTES = 10_000_000
MAX_CANDIDATE_GATE_BYTES = 10_000_000
MAX_BAKEOFF_REPORT_BYTES = 100_000_000
QUANTIZATION_PARITY_MAX_RELATIVE_DEGRADATION = 0.02
QUANTIZATION_PARITY_MIN_SAMPLES = 32
QUANTIZATION_PARITY_BOOTSTRAP_REPEATS = 1_000
QUANTIZATION_PARITY_SLICES = (
    "drug_name",
    "dose",
    "number",
    "unit",
    "negation",
    "terminology",
    "code_switch",
)
QUANTIZATION_PARITY_METRICS = {
    "mt": ("sacrebleu", "chrf2"),
    "asr": ("wer", "cer", "code_switch_wer"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


LOADED_RUNNER_SHA256 = sha256(Path(__file__).resolve())
TRAIN_STAGE_MAX_ATTEMPTS = 3


def verified_release_git_head(project_root: Path = ROOT) -> str:
    """Require a clean checkout whose HEAD matches tracking and live remote refs."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=GIT_CHECK_TIMEOUT_SECONDS,
        ).strip()
        branch = subprocess.check_output(
            ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=GIT_CHECK_TIMEOUT_SECONDS,
        ).strip()
        remote = subprocess.check_output(
            ["git", "config", "--get", f"branch.{branch}.remote"],
            cwd=project_root,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=GIT_CHECK_TIMEOUT_SECONDS,
        ).strip()
        merge_ref = subprocess.check_output(
            ["git", "config", "--get", f"branch.{branch}.merge"],
            cwd=project_root,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=GIT_CHECK_TIMEOUT_SECONDS,
        ).strip()
        upstream = subprocess.check_output(
            ["git", "rev-parse", "@{upstream}"],
            cwd=project_root,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=GIT_CHECK_TIMEOUT_SECONDS,
        ).strip()
        tracked_status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=project_root,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=GIT_CHECK_TIMEOUT_SECONDS,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Final backup requires a tracked Git checkout with upstream") from exc
    if not GIT_COMMIT_RE.fullmatch(head) or head != upstream:
        raise RuntimeError("Final backup requires local HEAD to match its upstream")
    if tracked_status:
        raise RuntimeError("Final backup requires a clean tracked worktree")
    if not branch or not remote or remote == "." or not merge_ref.startswith("refs/heads/"):
        raise RuntimeError("Final backup requires a named branch with a remote upstream")
    try:
        remote_output = subprocess.check_output(
            ["git", "ls-remote", "--exit-code", remote, merge_ref],
            cwd=project_root,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=GIT_CHECK_TIMEOUT_SECONDS,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Final backup could not verify the live remote branch") from exc
    remote_lines = remote_output.splitlines()
    remote_fields = remote_lines[0].split() if len(remote_lines) == 1 else []
    if (
        len(remote_fields) != 2
        or remote_fields[1] != merge_ref
        or not GIT_COMMIT_RE.fullmatch(remote_fields[0])
        or remote_fields[0] != head
    ):
        raise RuntimeError("Final backup requires local HEAD to match the live remote branch")
    return head


def ensure_verified_final_backup(
    comparison_path: Path,
    *,
    decision: str,
    scope: str,
    backup_root: Path | None = None,
) -> dict[str, Any]:
    """Create or reuse the schema-2 backup bound to one terminal comparison."""
    release_evidence = build_release_evidence(
        comparison_path,
        decision=decision,
        scope=scope,
    )
    git_head = verified_release_git_head()
    comparison_sha256 = str(release_evidence["comparison_sha256"])
    root = backup_root or ROOT / ".backups"
    output_dir = Path(
        os.path.abspath(
            root
            / f"onevoice-model-bakeoff-{comparison_sha256[:16]}-{git_head[:12]}"
        )
    )
    if output_dir.exists():
        verified = verify_backup(output_dir)
    else:
        verified = create_backup(
            output_dir,
            release_evidence=release_evidence,
        )
    if (
        verified.get("schema_version") != BACKUP_SCHEMA_VERSION
        or verified.get("verification") != "pass"
        or verified.get("secret_scan") != "pass"
        or verified.get("git_head") != git_head
        or verified.get("release_evidence") != release_evidence
    ):
        raise ValueError("Final backup does not match the terminal release evidence")
    manifest_path = output_dir / BACKUP_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("Final backup manifest is missing or linked")
    return {
        "path": str(output_dir),
        "schema_version": BACKUP_SCHEMA_VERSION,
        "verification": "pass",
        "secret_scan": "pass",
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256(manifest_path),
        },
        "comparison_sha256": comparison_sha256,
        "git_head": git_head,
        "release_evidence": release_evidence,
    }


def complete_terminal_bakeoff(
    state_path: Path,
    state: dict[str, Any],
    comparison_path: Path,
    *,
    decision: str,
    promotion_allowed: bool,
    scope: str,
) -> dict[str, Any]:
    """Publish the final state only after a release-bound backup verifies."""
    if decision != ("promote" if promotion_allowed else "reject"):
        raise ValueError("Terminal decision disagrees with promotion flag")
    state.update(
        {
            "stage": "verified_backup",
            "execution_status": "backing_up",
            "promotion_allowed": promotion_allowed,
            "decision": decision,
        }
    )
    atomic_json(state_path, state)
    backup = ensure_verified_final_backup(
        comparison_path,
        decision=decision,
        scope=scope,
    )
    state.update(
        {
            "stage": "promotion_or_reject",
            "execution_status": "complete",
            "verified_backup": backup,
        }
    )
    atomic_json(state_path, state)
    return backup


class StageExecutionError(RuntimeError):
    """A launched stage returned non-zero or failed to publish its outputs."""


class JsonDocumentDigestMismatch(ValueError):
    """A strict JSON document did not match its external digest binding."""


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


def read_json_mapping(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Read one stable strict-JSON document whose root is a mapping."""
    payload = read_stable_regular_file(
        path,
        maximum_bytes=maximum_bytes,
        label=label,
    )
    if (
        expected_sha256 is not None
        and hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise JsonDocumentDigestMismatch(f"{label} checksum does not match")
    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError(f"{label} is not valid strict JSON") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} root must be a mapping")
    return parsed


def validate_loaded_runner_generation() -> None:
    actual = sha256(Path(__file__).resolve())
    if actual != LOADED_RUNNER_SHA256:
        raise RuntimeError(
            "Loaded bake-off runner no longer matches the file on disk; "
            "restart only after confirming no GPU child is active"
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def manifest_task(rows: list[dict[str, Any]], path: Path) -> str:
    tasks = set()
    for row in rows:
        if row.get("source_text") is not None and row.get("target_text") is not None:
            tasks.add("mt")
        elif row.get("text") is not None:
            tasks.add("asr")
        else:
            raise ValueError(f"Cannot infer manifest task from {path}")
    if len(tasks) != 1:
        raise ValueError(f"Manifest mixes tasks: {path}")
    return tasks.pop()


def leakage_values(
    rows: list[dict[str, Any]],
    task: str,
    path: Path,
    *,
    strict_selection: bool,
) -> dict[str, set[str]]:
    keys = (
        ("id", "pair_fingerprint")
        if task == "mt"
        else ("id", "text_fingerprint", "audio_sha256", "speaker", "group")
    )
    values = {key: set() for key in keys}
    required = set(keys[:2] if task == "mt" else keys[:3])
    counts = {key: 0 for key in keys}
    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id") or "").strip()
        if not row_id:
            raise ValueError(f"Missing id in {path} row {index}")
        values["id"].add(row_id)
        counts["id"] += 1
        if task == "mt":
            source = str(row.get("source_text") or "").strip()
            target = str(row.get("target_text") or "").strip()
            if not source or not target:
                raise ValueError(f"Missing MT payload in {path} row {index}")
            computed = fingerprint_text(source + "\x1f" + target)
            declared = str(row.get("pair_fingerprint") or "").strip()
            if declared and declared != computed:
                raise ValueError(f"pair_fingerprint mismatch in {path} row {index}")
            values["pair_fingerprint"].add(computed)
            counts["pair_fingerprint"] += 1
        else:
            transcript = str(row.get("text") or "").strip()
            if not transcript:
                raise ValueError(f"Missing ASR text in {path} row {index}")
            computed = fingerprint_text(transcript)
            declared = str(row.get("text_fingerprint") or "").strip()
            if declared and declared != computed:
                raise ValueError(f"text_fingerprint mismatch in {path} row {index}")
            values["text_fingerprint"].add(computed)
            counts["text_fingerprint"] += 1
            for key in ("audio_sha256", "speaker", "group"):
                value = str(row.get(key) or "").strip()
                if value:
                    values[key].add(value)
                    counts[key] += 1
        if strict_selection:
            for key in required:
                if counts[key] != index:
                    raise ValueError(f"Selection manifest is missing {key}: {path} row {index}")
    if strict_selection:
        for key in required:
            if len(values[key]) != len(rows):
                raise ValueError(f"Selection manifest contains duplicate {key}: {path}")
    return values


def command_digest(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()


def output_evidence(paths: list[Path]) -> list[dict[str, Any]]:
    """Fingerprint file outputs while retaining directory existence checks.

    Training directories are verified by ``completed_adapter`` against their
    signed round archives. Small stage products such as benchmark reports,
    blind gates and deployment drafts are hashed here so a completed stage
    cannot silently trust a replaced file on resume.
    """
    evidence = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_symlink():
            raise ValueError(f"Stage output must not be a symlink: {path}")
        if not path.is_file() and not path.is_dir():
            raise ValueError(f"Stage output must be a regular file or directory: {path}")
        record: dict[str, Any] = {
            "path": str(path.resolve()),
            "kind": "directory" if path.is_dir() else "file",
        }
        if path.is_file():
            record.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
        evidence.append(record)
    return evidence


def resume_scoring_command_compatible(
    previous_command: Any,
    current_command: list[str],
) -> bool:
    """Permit the recovery-only flag without invalidating completed GPU work."""
    if not isinstance(previous_command, list) or not all(
        isinstance(item, str) for item in previous_command
    ):
        return False
    if not any(Path(item).name == "run_baseline_benchmarks.py" for item in current_command):
        return False
    previous_flags = previous_command.count("--resume-scoring")
    current_flags = current_command.count("--resume-scoring")
    if {previous_flags, current_flags} != {0, 1}:
        return False
    return (
        [item for item in previous_command if item != "--resume-scoring"]
        == [item for item in current_command if item != "--resume-scoring"]
    )


def resolved_command_executable(command_token: str) -> Path | None:
    """Resolve a command executable using the same working directory as stages."""
    executable = Path(command_token)
    if not executable.is_absolute():
        executable = ROOT / executable
    try:
        if not executable.is_file():
            return None
        resolved = executable.resolve(strict=True)
        return resolved if resolved.is_file() else None
    except (OSError, RuntimeError):
        return None


def interpreter_alias_only_command_change(
    previous_command: Any,
    current_command: list[str],
) -> bool:
    """Accept only an equivalent spelling of the command executable.

    Completed training stages created before a recovery runner upgrade do not
    carry ``--bakeoff-runner-sha256``.  They are still immutable evidence, but
    may have recorded the shared virtualenv interpreter through a different
    relative alias.  Treat that spelling-only difference as compatible only
    when both paths currently resolve, strictly, to the same regular file and
    every remaining command token is byte-for-byte identical.
    """
    if (
        not isinstance(previous_command, list)
        or not previous_command
        or not current_command
        or not all(isinstance(item, str) for item in previous_command)
        or not all(isinstance(item, str) for item in current_command)
    ):
        return False
    previous = list(previous_command)
    current = list(current_command)
    if previous[0] == current[0] or previous[1:] != current[1:]:
        return False
    previous_executable = resolved_command_executable(previous[0])
    current_executable = resolved_command_executable(current[0])
    return (
        previous_executable is not None
        and current_executable is not None
        and previous_executable == current_executable
    )


def runner_generation_only_command_change(
    previous_command: Any,
    current_command: list[str],
) -> bool:
    """Allow immutable completed evidence across a runner-only code upgrade.

    Benchmark reports retain the runner hash that produced them. On resume, a
    newer runner may safely skip that completed stage only when the command
    differs by the 64-character ``--bakeoff-runner-sha256`` value and,
    optionally, by two interpreter spellings that resolve to the same regular
    file. ``run_stage`` still verifies the recorded output fingerprint before
    returning.
    """
    if not isinstance(previous_command, list) or not all(
        isinstance(item, str) for item in previous_command
    ):
        return False
    option = "--bakeoff-runner-sha256"
    if previous_command.count(option) != 1 or current_command.count(option) != 1:
        return False
    if not any(Path(item).name == "run_baseline_benchmarks.py" for item in current_command):
        return False
    previous = list(previous_command)
    current = list(current_command)
    previous_index = previous.index(option) + 1
    current_index = current.index(option) + 1
    if previous_index >= len(previous) or current_index >= len(current):
        return False
    old_hash = previous[previous_index]
    new_hash = current[current_index]
    if not all(
        len(value) == 64 and set(value.lower()) <= set("0123456789abcdef")
        for value in (old_hash, new_hash)
    ):
        return False
    if old_hash == new_hash:
        return False
    previous[previous_index] = "<runner-generation>"
    current[current_index] = "<runner-generation>"
    if previous[0] != current[0]:
        previous_python = resolved_command_executable(previous[0])
        current_python = resolved_command_executable(current[0])
        if (
            previous_python is None
            or current_python is None
            or previous_python != current_python
        ):
            return False
        previous[0] = "<python-executable>"
        current[0] = "<python-executable>"
    return previous == current


def audit_completed_stage_resume(
    state: dict[str, Any],
    current_python: str,
) -> dict[str, Any]:
    """Read-only audit of completed stage commands and recorded outputs.

    The audit applies the only two execution-identity changes accepted during
    resume: an equivalent interpreter spelling and the loaded benchmark-runner
    generation. It never mutates state or launches a subprocess.
    """
    failures: list[str] = []
    details: list[dict[str, Any]] = []
    stages = state.get("stages")
    if not isinstance(stages, dict):
        return {
            "status": "fail",
            "completed_stage_count": 0,
            "stages": [],
            "failures": ["state.stages must be a mapping"],
        }
    if not isinstance(current_python, str) or not current_python.strip():
        return {
            "status": "fail",
            "completed_stage_count": 0,
            "stages": [],
            "failures": ["current Python executable must be a non-empty string"],
        }

    for name in sorted(stages):
        record = stages[name]
        if not isinstance(record, dict) or record.get("status") != "complete":
            continue
        stage_failures: list[str] = []
        compatibility = "invalid"
        stored_command = record.get("command")
        if (
            not isinstance(stored_command, list)
            or not stored_command
            or not all(isinstance(item, str) for item in stored_command)
        ):
            stage_failures.append("stored command is invalid")
        else:
            stored_digest = command_digest(stored_command)
            if record.get("command_sha256") != stored_digest:
                stage_failures.append("stored command digest does not match command")
            current_command = list(stored_command)
            current_command[0] = current_python
            runner_option = "--bakeoff-runner-sha256"
            is_benchmark = any(
                Path(item).name == "run_baseline_benchmarks.py"
                for item in current_command
            )
            if is_benchmark:
                if current_command.count(runner_option) != 1:
                    stage_failures.append(
                        "benchmark runner generation binding is missing or ambiguous"
                    )
                else:
                    runner_index = current_command.index(runner_option) + 1
                    if runner_index >= len(current_command) or not SHA256_RE.fullmatch(
                        current_command[runner_index].lower()
                    ):
                        stage_failures.append(
                            "benchmark runner generation value is missing or invalid"
                        )
                    else:
                        current_command[runner_index] = LOADED_RUNNER_SHA256
            if not stage_failures:
                if stored_command == current_command:
                    compatibility = "exact"
                elif interpreter_alias_only_command_change(
                    stored_command,
                    current_command,
                ):
                    compatibility = "interpreter_alias"
                elif runner_generation_only_command_change(
                    stored_command,
                    current_command,
                ):
                    compatibility = "runner_generation"
                else:
                    stage_failures.append(
                        "current execution identity is not resume-compatible"
                    )

        recorded_evidence = record.get("output_evidence")
        if not isinstance(recorded_evidence, list) or not recorded_evidence:
            stage_failures.append("recorded output evidence is missing")
        elif any(
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not item["path"]
            for item in recorded_evidence
        ):
            stage_failures.append("recorded output evidence is invalid")
        else:
            try:
                current_evidence = output_evidence(
                    [Path(item["path"]) for item in recorded_evidence]
                )
            except (FileNotFoundError, OSError, ValueError) as exc:
                stage_failures.append(f"output evidence cannot be verified: {exc}")
            else:
                if current_evidence != recorded_evidence:
                    stage_failures.append("recorded output evidence changed")

        details.append(
            {
                "stage": name,
                "command_compatibility": compatibility,
                "status": "pass" if not stage_failures else "fail",
                "failures": stage_failures,
            }
        )
        failures.extend(f"{name}: {failure}" for failure in stage_failures)

    return {
        "status": "pass" if not failures else "fail",
        "completed_stage_count": len(details),
        "stages": details,
        "failures": failures,
    }


def pid_is_live(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def matching_linux_command_pids(command: Any) -> list[int]:
    if os.name != "posix" or not isinstance(command, list) or not command:
        return []
    expected = [str(item) for item in command]
    matches = []
    for cmdline_path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            observed = [
                item.decode(errors="surrogateescape")
                for item in cmdline_path.read_bytes().split(b"\0")
                if item
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if observed == expected:
            matches.append(int(cmdline_path.parent.name))
    return sorted(matches)


def assert_interrupted_stage_is_not_live(previous: dict[str, Any] | None) -> None:
    if not isinstance(previous, dict) or previous.get("status") != "running":
        return
    pid = previous.get("pid")
    if isinstance(pid, int) and not isinstance(pid, bool):
        if pid_is_live(pid):
            raise RuntimeError(f"Refusing duplicate stage launch; PID {pid} is still live")
        return
    matches = matching_linux_command_pids(previous.get("command"))
    if matches:
        raise RuntimeError(
            "Refusing duplicate legacy stage launch; matching PIDs are still live: "
            + ", ".join(str(item) for item in matches)
        )


def atomic_json(
    path: Path,
    payload: dict[str, Any],
    *,
    maximum_bytes: int = MAX_BAKEOFF_STATE_BYTES,
) -> None:
    write_durable_json(
        path,
        payload,
        maximum_bytes=maximum_bytes,
        label="Bake-off JSON",
    )


def invocation_binding(
    program_path: Path,
    config_path: Path,
    scope: str,
    research_approvals: set[str],
) -> dict[str, Any]:
    return {
        "current_program_state": str(program_path.resolve()),
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256(config_path.resolve()),
        },
        "scope": scope,
        "research_approvals": sorted(research_approvals),
    }


def bind_or_validate_invocation(
    state: dict[str, Any], binding: dict[str, Any]
) -> None:
    recorded = state.get("invocation_binding")
    if recorded is None:
        completed = [
            name
            for name, stage in state.get("stages", {}).items()
            if isinstance(stage, dict) and stage.get("status") == "complete"
        ]
        if completed:
            raise ValueError(
                "Legacy bake-off state has completed stages but no invocation binding: "
                + ", ".join(sorted(completed))
            )
        state["invocation_binding"] = binding
        return
    if recorded != binding:
        raise ValueError("Bake-off invocation changed for the existing state directory")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self._composition_depth = 0
        self._composed_nodes = 0

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "model-bakeoff config aliases are not allowed",
                self.peek_event().start_mark,
            )
        if self._composition_depth >= MAX_BAKEOFF_CONFIG_DEPTH:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "model-bakeoff config nesting is too deep",
                self.peek_event().start_mark,
            )
        if self._composed_nodes >= MAX_BAKEOFF_CONFIG_NODES:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                "model-bakeoff config contains too many nodes",
                self.peek_event().start_mark,
            )
        self._composition_depth += 1
        self._composed_nodes += 1
        try:
            return super().compose_node(parent, index)
        finally:
            self._composition_depth -= 1


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found a duplicate key",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _validate_config_tree(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Model-bakeoff config contains a non-finite number")
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Model-bakeoff config mapping keys must be strings")
        for item in value.values():
            _validate_config_tree(item)
        return
    if isinstance(value, list):
        for item in value:
            _validate_config_tree(item)
        return
    raise ValueError("Model-bakeoff config contains an unsupported YAML type")


def load_config(path: Path) -> dict[str, Any]:
    payload = read_stable_regular_file(
        path,
        maximum_bytes=MAX_BAKEOFF_CONFIG_BYTES,
        label="Model-bakeoff config",
    )
    try:
        config = yaml.load(payload.decode("utf-8"), Loader=_UniqueKeySafeLoader)
    except (UnicodeDecodeError, yaml.YAMLError, RecursionError):
        raise ValueError("Model-bakeoff config is not valid strict YAML") from None
    if not isinstance(config, dict):
        raise ValueError("Model-bakeoff config root must be a mapping")
    _validate_config_tree(config)
    if config.get("version") != 1:
        raise ValueError("Unsupported model-bakeoff config version")
    return config


def validate_resources(config: dict[str, Any]) -> None:
    resources = config["resources"]
    validate_resource_limits(resources)
    if not config["principles"].get("one_gpu_child_at_a_time"):
        raise ValueError("Bake-off requires one GPU child at a time")
    deployment_min_runs = config["promotion_gate"].get("deployment_min_runs")
    if (
        isinstance(deployment_min_runs, bool)
        or not isinstance(deployment_min_runs, int)
        or deployment_min_runs < 1
    ):
        raise ValueError("Deployment benchmark requires a positive integer run count")
    expected_metrics = {
        "mt_metrics": {"sacrebleu", "chrf2"},
        "asr_metrics": {"wer", "cer", "code_switch_wer"},
    }
    for key, expected in expected_metrics.items():
        configured = config["promotion_gate"].get(key)
        if (
            not isinstance(configured, list)
            or len(configured) != len(expected)
            or set(configured) != expected
        ):
            raise ValueError(f"Promotion gate {key} must contain exactly {sorted(expected)}")
    deployment_metrics = config["promotion_gate"].get("deployment_metrics")
    configured_deployment_metrics = {
        "latency_p50_ms",
        "latency_p95_ms",
        "peak_ram_bytes",
        "peak_vram_bytes",
        "model_bytes",
    }
    if (
        not isinstance(deployment_metrics, list)
        or len(deployment_metrics) != len(configured_deployment_metrics)
        or set(deployment_metrics) != configured_deployment_metrics
    ):
        raise ValueError(
            "Promotion gate deployment_metrics must contain exactly "
            f"{sorted(configured_deployment_metrics)}"
        )
    code_switch_limit = float(config["promotion_gate"].get("asr_code_switch_wer_max", math.nan))
    if not math.isfinite(code_switch_limit) or not 0.0 <= code_switch_limit <= 1.0:
        raise ValueError("ASR code-switch WER limit must be finite and in [0, 1]")


def hyperparameter_profiles(
    config: dict[str, Any], task: str
) -> list[dict[str, Any]]:
    search = config.get("hyperparameter_search")
    expected_policy = "best_safety_eligible_profile_per_candidate_direction"
    if not isinstance(search, dict) or search.get("selection") != expected_policy:
        raise ValueError("Unsupported or missing hyperparameter search policy")
    profiles = search.get("profiles", {}).get(task)
    if not isinstance(profiles, list) or len(profiles) < 2:
        raise ValueError(f"{task} requires at least two hyperparameter profiles")
    ids = []
    validated = []
    for raw in profiles:
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid {task} hyperparameter profile")
        profile = deepcopy(raw)
        profile_id = str(profile.get("id") or "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", profile_id):
            raise ValueError(f"Invalid {task} hyperparameter profile id: {profile_id!r}")
        try:
            learning_rate = float(profile["learning_rate"])
            rank = int(profile["lora_rank"])
            alpha = int(profile["lora_alpha"])
            dropout = float(profile["lora_dropout"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {task} hyperparameter profile: {profile_id}") from exc
        if (
            not math.isfinite(learning_rate)
            or learning_rate <= 0.0
            or isinstance(profile["lora_rank"], bool)
            or isinstance(profile["lora_alpha"], bool)
            or rank <= 0
            or alpha <= 0
            or not math.isfinite(dropout)
            or not 0.0 <= dropout < 1.0
        ):
            raise ValueError(f"Unsafe {task} hyperparameter profile: {profile_id}")
        ids.append(profile_id)
        validated.append(
            {
                "id": profile_id,
                "learning_rate": learning_rate,
                "lora_rank": rank,
                "lora_alpha": alpha,
                "lora_dropout": dropout,
            }
        )
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate {task} hyperparameter profile ids")
    return validated


def validate_selection_artifacts(config: dict[str, Any]) -> dict[str, str]:
    hashes = {}
    forbidden = {(ROOT / value).resolve() for value in config["data"]["forbidden_selection_inputs"]}
    selection_values: dict[str, dict[str, set[str]]] = {}
    for task in ("mt", "asr"):
        record = config["data"]["selection_dev"][task]
        path = (ROOT / record["path"]).resolve()
        if path in forbidden:
            raise ValueError(f"Locked test cannot be used for selection: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"Selection checksum mismatch for {task}: {actual}")
        hashes[task] = actual
        rows = read_jsonl(path)
        if manifest_task(rows, path) != task:
            raise ValueError(f"Selection task mismatch for {task}: {path}")
        selection_values[task] = leakage_values(
            rows,
            task,
            path,
            strict_selection=True,
        )
    for task in ("mt", "asr"):
        train_path = (ROOT / config["data"]["train"][task]).resolve()
        if not train_path.is_file():
            raise FileNotFoundError(train_path)
        train_rows = read_jsonl(train_path)
        if manifest_task(train_rows, train_path) != task:
            raise ValueError(f"Training task mismatch for {task}: {train_path}")
        train_values = leakage_values(
            train_rows,
            task,
            train_path,
            strict_selection=False,
        )
        for key, selected in selection_values[task].items():
            overlap = selected & train_values[key]
            if overlap:
                raise ValueError(
                    f"Selection/train leakage for {task}.{key}: {len(overlap)}"
                )
    for path in sorted(forbidden):
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        task = manifest_task(rows, path)
        forbidden_values = leakage_values(
            rows,
            task,
            path,
            strict_selection=False,
        )
        for key, selected in selection_values[task].items():
            overlap = selected & forbidden_values[key]
            if overlap:
                raise ValueError(
                    f"Selection/{path.name} leakage for {task}.{key}: {len(overlap)}"
                )
    accuracy_record = config["data"]["accuracy_program"]
    accuracy_path = (ROOT / accuracy_record["path"]).resolve()
    if not accuracy_path.is_file():
        raise FileNotFoundError(accuracy_path)
    accuracy_hash = sha256(accuracy_path)
    if accuracy_hash != accuracy_record["sha256"]:
        raise ValueError(f"Accuracy policy checksum mismatch: {accuracy_hash}")
    hashes["accuracy_program"] = accuracy_hash
    return hashes


def validate_candidate_matrix(config: dict[str, Any]) -> None:
    ids = []
    for task in ("mt", "asr"):
        candidates = config["candidates"][task]
        if sum(candidate.get("role") == "candidate_a" for candidate in candidates) != 1:
            raise ValueError(f"{task} must have exactly one Candidate A")
        for candidate in candidates:
            ids.append(candidate["id"])
            if len(str(candidate.get("revision") or "")) != 40:
                raise ValueError(f"Candidate must pin a 40-character revision: {candidate['id']}")
            if candidate.get("status") not in {
                "planned",
                "in_progress",
                "waiting_for_current_pipeline",
            }:
                raise ValueError(f"Invalid pre-benchmark status: {candidate['id']}")
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate IDs must be unique")


def find_current_program_state(config: dict[str, Any], explicit: Path | None) -> Path:
    if explicit:
        return Path(os.path.abspath(explicit))
    matches = sorted(
        ROOT.glob(config["prerequisite"]["current_program_state_glob"]),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError("No current GPU program state found")
    return matches[-1]


def wait_for_current_program(path: Path, poll_seconds: float, should_wait: bool) -> dict[str, Any]:
    while True:
        try:
            state = read_json_mapping(
                path,
                maximum_bytes=MAX_BAKEOFF_STATE_BYTES,
                label="Current GPU program state",
            )
        except RuntimeError:
            if not should_wait:
                raise
            time.sleep(poll_seconds)
            continue
        status = state.get("execution_status")
        if status == "complete":
            return state
        if status == "error":
            raise RuntimeError(f"Current GPU program failed: {state.get('error')}")
        if not should_wait:
            raise RuntimeError(f"Current GPU program is not complete: {status!r}")
        time.sleep(poll_seconds)


def candidate_output_dir(
    current: dict[str, Any], task: str, override: Path | None = None
) -> Path:
    output = override
    if output is None:
        stage = current.get("stages", {}).get(f"{task}_candidate")
        command = stage.get("command") if isinstance(stage, dict) else None
        if not isinstance(command, list) or "--output-dir" not in command:
            raise ValueError(f"Current program lacks {task} Candidate A output binding")
        index = command.index("--output-dir")
        if index + 1 >= len(command):
            raise ValueError(f"Current program has an invalid {task} --output-dir binding")
        output = Path(str(command[index + 1]))
    return resolve_regular_directory_under(
        output,
        project_root=ROOT,
        allowed_root=ROOT / "gpu-runs",
        label="Candidate A output",
    )


def validate_candidate_a_locked_evaluation(
    current: dict[str, Any],
    config: dict[str, Any],
    task: str,
    override: Path | None = None,
) -> dict[str, Any]:
    """Bind a terminal locked-test gate before any bake-off GPU work.

    Candidate A may pass or fail its locked clinical policy and still remain a
    reference. Infrastructure ``error`` is not a benchmark result and must
    stop the bake-off. New evidence requires a canonical member manifest; the
    explicitly documented MT legacy bundle remains reference-only.
    """
    output = candidate_output_dir(current, task, override)
    gate_path = output / "candidate_gate.json"
    if gate_path.is_symlink() or not gate_path.is_file():
        raise FileNotFoundError(f"Candidate A gate is incomplete: {gate_path}")
    gate = read_json_mapping(
        gate_path,
        maximum_bytes=MAX_CANDIDATE_GATE_BYTES,
        label=f"Candidate A {task} gate",
    )
    status = gate.get("status")
    if status not in {"pass", "fail"} or gate.get("error") not in {None, ""}:
        raise ValueError(f"Candidate A {task} locked evaluation is not terminal: {gate_path}")
    if bool(gate.get("promotion_allowed")) != (status == "pass"):
        raise ValueError(f"Candidate A {task} promotion flag is inconsistent: {gate_path}")
    checks = gate.get("checks")
    resources = gate.get("resource_runs")
    if not isinstance(checks, list) or not checks or any(not isinstance(row, dict) for row in checks):
        raise ValueError(f"Candidate A {task} gate lacks check evidence: {gate_path}")
    if not isinstance(resources, dict) or not resources:
        raise ValueError(f"Candidate A {task} gate lacks resource evidence: {gate_path}")

    candidate_a = next(
        candidate
        for candidate in config["candidates"][task]
        if candidate.get("role") == "candidate_a"
    )
    if (
        gate.get("base_model") != candidate_a["model"]
        or gate.get("base_model_revision") != candidate_a["revision"]
    ):
        raise ValueError(f"Candidate A {task} gate model binding is inconsistent: {gate_path}")
    required_checks = {
        *(f"clinical_{name}" for name in config["promotion_gate"]["critical_slices"]),
        *(f"clinical_{name}" for name in config["promotion_gate"]["policy_slices"]),
    }
    aggregate_checks = (
        {"mt_sacrebleu", "mt_chrf2"}
        if task == "mt"
        else {"asr_vi_wer", "asr_code_switch_wer"}
    )
    check_names = [str(row.get("name") or "") for row in checks]
    normalized_checks = {
        name.removesuffix("_failure_rate") for name in check_names if name
    }
    if (
        len(check_names) != len(set(check_names))
        or not aggregate_checks <= set(check_names)
        or not required_checks <= normalized_checks
        or any(not isinstance(row.get("pass"), bool) for row in checks)
    ):
        raise ValueError(f"Candidate A {task} gate has incomplete checks: {gate_path}")
    checks_pass = all(bool(row["pass"]) for row in checks)
    if checks_pass != (status == "pass"):
        raise ValueError(f"Candidate A {task} gate status disagrees with its checks: {gate_path}")

    for run_name in ("aggregate", "clinical"):
        run = resources.get(run_name)
        if not isinstance(run, dict) or run.get("return_code") != 0:
            raise ValueError(f"Candidate A {task} {run_name} resource run is incomplete")
    resource_limits = gate.get("resource_limits")
    configured_limits = config["resources"]
    if not isinstance(resource_limits, dict) or any(
        float(resource_limits.get(name, math.nan)) != float(configured_limits[name])
        for name in (
            "gpu_memory_fraction",
            "utilization_percent",
            "hard_utilization_percent",
            "resume_percent",
        )
    ):
        raise ValueError(f"Candidate A {task} resource limits do not match policy")
    if task == "mt":
        locked_hashes = [gate.get("locked_safety_sha256")]
    else:
        locked = gate.get("locked_hashes")
        locked_hashes = (
            [locked.get("test"), locked.get("safety")]
            if isinstance(locked, dict)
            else []
        )
    if not locked_hashes or any(
        not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)
        for digest in locked_hashes
    ):
        raise ValueError(f"Candidate A {task} locked input hashes are incomplete")

    adapter = Path(str(current.get(f"{task}_adapter") or "")).resolve()
    if Path(str(gate.get("adapter") or "")).resolve() != adapter:
        raise ValueError(f"Candidate A {task} gate adapter does not match program state")

    archive = output.with_suffix(".tar.gz")
    sidecar, manifest_path = evidence_sidecars(archive)
    if not archive.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Candidate A {task} archive/sidecar is incomplete: {archive}")
    digest = sha256(archive)
    sidecar_fields = sidecar.read_text(encoding="utf-8").strip().split()
    if sidecar_fields != [digest, archive.name]:
        raise ValueError(f"Candidate A {task} archive checksum mismatch: {archive}")

    policy = config["prerequisite"]["candidate_a_locked_evaluation"][task]
    canonical_manifest = manifest_path.is_file()
    archive_record: dict[str, Any] = {
        "path": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": digest,
        "content_manifest_verified": canonical_manifest,
    }
    if canonical_manifest:
        manifest = verify_evidence_archive(archive)
        member_name = f"{output.name}/candidate_gate.json"
        members = {str(row["path"]): row for row in manifest["files"]}
        member = members.get(member_name)
        if (
            member is None
            or int(member["bytes"]) != gate_path.stat().st_size
            or str(member["sha256"]) != sha256(gate_path)
        ):
            raise ValueError(f"Candidate A {task} gate does not match its archive")
        archive_record["manifest_sha256"] = sha256(manifest_path)
    elif bool(policy.get("require_content_manifest", True)):
        raise ValueError(f"Candidate A {task} archive lacks a canonical content manifest")
    elif not bool(policy.get("legacy_reference_only", False)):
        raise ValueError(f"Candidate A {task} legacy evidence is not explicitly reference-only")

    return {
        "task": task,
        "status": status,
        "promotion_allowed": status == "pass",
        "output_dir": str(output),
        "gate": {
            "path": str(gate_path),
            "bytes": gate_path.stat().st_size,
            "sha256": sha256(gate_path),
        },
        "archive": archive_record,
        "legacy_reference_only": not canonical_manifest,
    }


def candidate_a_resume_output(
    state: dict[str, Any], task: str, explicit: Path | None = None
) -> Path | None:
    """Reuse the already locked Candidate A output when resuming a bake-off.

    The returned path is only a hint: ``validate_candidate_a_locked_evaluation``
    still revalidates the gate, model binding, hashes, archive, and manifest before
    any GPU work is allowed. An explicit CLI override remains authoritative.
    """
    if explicit is not None:
        return explicit
    record = state.get("candidate_a_locked_evaluations", {}).get(task)
    if not isinstance(record, dict) or record.get("task") != task:
        return None
    output_dir = record.get("output_dir")
    return Path(str(output_dir)) if output_dir else None


def tree_manifest(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"Candidate adapter root cannot be a symlink: {path}")
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = []
    for item in sorted(
        path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()
    ):
        if item.is_symlink():
            raise ValueError(f"Candidate adapter cannot contain symlinks: {item}")
        if item.is_file():
            files.append(
                {
                    "path": item.relative_to(path).as_posix(),
                    "bytes": item.stat().st_size,
                    "sha256": sha256(item),
                }
            )
    if not files:
        raise ValueError(f"Cannot freeze empty candidate directory: {path}")
    manifest_digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"root": str(path.resolve()), "files": files, "manifest_sha256": manifest_digest}


def deployment_expectations(
    winner_specs: list[tuple[str, str | None, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Bind hardware evidence to the exact selected adapter trees."""
    adapter_hashes: dict[Path, str] = {}
    expected = []
    for task, direction, winner in winner_specs:
        adapter = Path(winner["adapter"]).resolve()
        if adapter not in adapter_hashes:
            adapter_hashes[adapter] = str(tree_manifest(adapter)["manifest_sha256"])
        recorded_manifest = str(winner.get("adapter_manifest_sha256") or "")
        if recorded_manifest and recorded_manifest != adapter_hashes[adapter]:
            raise ValueError(
                f"Selected winner adapter changed after selection: {winner['candidate_id']}"
            )
        expected.append(
            {
                "task": task,
                "direction": direction,
                "candidate_id": winner["candidate_id"],
                "adapter_manifest_sha256": adapter_hashes[adapter],
            }
        )
    return expected


def percentile_linear(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def qcs6490_identity_failures(identity: Any) -> list[str]:
    """Validate raw Linux device-tree evidence captured on the target board."""
    if not isinstance(identity, dict):
        return ["payload_invalid"]
    failures: list[str] = []
    if identity.get("version") != 1:
        failures.append("version_invalid")
    if identity.get("capture_source") != "linux_sysfs_device_tree":
        failures.append("capture_source_invalid")

    captured_at = str(identity.get("captured_at") or "")
    try:
        captured_time = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        if captured_time.tzinfo is None:
            raise ValueError("timezone required")
    except ValueError:
        failures.append("captured_at_invalid")

    architecture = str(identity.get("architecture") or "").strip().lower()
    if architecture not in {"aarch64", "arm64"}:
        failures.append("architecture_not_arm64")

    board_model = str(identity.get("board_model") or "").strip().upper()
    if not (
        "QCS6490" in board_model
        or "RB3 GEN 2" in board_model
        or "RB3GEN2" in board_model
    ):
        failures.append("board_model_not_qcs6490")

    compatible_value = identity.get("device_tree_compatible")
    if isinstance(compatible_value, str):
        compatible = [compatible_value]
    elif isinstance(compatible_value, list):
        compatible = [str(value) for value in compatible_value]
    else:
        compatible = []
    compatible_text = " ".join(compatible).upper()
    if "QCOM" not in compatible_text or (
        "QCS6490" not in compatible_text and "RB3GEN2" not in compatible_text
    ):
        failures.append("compatible_not_qcs6490")
    return failures


def deployment_required_metrics(configured: list[str]) -> list[str]:
    """Add code-mandatory physical metrics without changing active config SHA."""
    mandatory = (
        "power_avg_mw",
        "power_p95_mw",
        "temperature_peak_c",
    )
    return list(dict.fromkeys([*configured, *mandatory]))


def _parity_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _relative_metric_degradation(
    reference: float,
    quantized: float,
    *,
    greater_is_better: bool,
) -> float | None:
    if reference == 0.0:
        if greater_is_better and quantized >= 0.0:
            return 0.0
        return 0.0 if quantized == 0.0 else None
    delta = reference - quantized if greater_is_better else quantized - reference
    return delta / abs(reference)


def quantization_parity_evidence_failures(
    evidence: Any,
    *,
    expected: dict[str, Any] | None = None,
    artifact_sha256: str | None = None,
    artifact_bytes: int | None = None,
) -> list[str]:
    """Validate immutable float-vs-compiled parity evidence and safety gates."""
    if not isinstance(evidence, dict):
        return ["payload_invalid"]
    failures: list[str] = []
    expected_fields = {
        "version",
        "status",
        "evidence_source",
        "evaluated_at",
        "task",
        "direction",
        "candidate_id",
        "adapter_manifest_sha256",
        "artifact_sha256",
        "artifact_bytes",
        "manifest_sha256",
        "manifest_bytes",
        "reference_predictions_sha256",
        "reference_predictions_bytes",
        "reference_provenance_sha256",
        "reference_provenance_bytes",
        "quantized_predictions_sha256",
        "quantized_predictions_bytes",
        "quantized_provenance_sha256",
        "quantized_provenance_bytes",
        "decoding",
        "samples",
        "bootstrap",
        "metrics",
        "safety_slices",
    }
    if set(evidence) != expected_fields:
        failures.append("fields_invalid")
    if evidence.get("version") != 2:
        failures.append("version_invalid")
    if evidence.get("status") != "pass":
        failures.append("status_not_pass")
    if evidence.get("evidence_source") != "onevoice_quantization_parity":
        failures.append("evidence_source_invalid")

    evaluated_at = str(evidence.get("evaluated_at") or "")
    try:
        evaluated_time = datetime.fromisoformat(evaluated_at.replace("Z", "+00:00"))
        if evaluated_time.tzinfo is None:
            raise ValueError("timezone required")
    except ValueError:
        failures.append("evaluated_at_invalid")

    task = str(evidence.get("task") or "")
    direction = evidence.get("direction")
    candidate_id = str(evidence.get("candidate_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", candidate_id):
        failures.append("candidate_id_invalid")
    if task not in QUANTIZATION_PARITY_METRICS:
        failures.append("task_invalid")
    elif task == "mt" and direction not in {"en_to_vi", "vi_to_en"}:
        failures.append("direction_invalid")
    elif task == "asr" and direction is not None:
        failures.append("direction_invalid")

    if expected is not None:
        for field in ("task", "direction", "candidate_id", "adapter_manifest_sha256"):
            if evidence.get(field) != expected.get(field):
                failures.append(f"{field}_mismatch")
    if artifact_sha256 is not None and evidence.get("artifact_sha256") != artifact_sha256:
        failures.append("artifact_sha256_mismatch")
    if artifact_bytes is not None and evidence.get("artifact_bytes") != artifact_bytes:
        failures.append("artifact_bytes_mismatch")

    for field in (
        "adapter_manifest_sha256",
        "artifact_sha256",
        "manifest_sha256",
        "reference_predictions_sha256",
        "reference_provenance_sha256",
        "quantized_predictions_sha256",
        "quantized_provenance_sha256",
    ):
        if not SHA256_RE.fullmatch(str(evidence.get(field) or "")):
            failures.append(f"{field}_invalid")
    for field in (
        "artifact_bytes",
        "manifest_bytes",
        "reference_predictions_bytes",
        "reference_provenance_bytes",
        "quantized_predictions_bytes",
        "quantized_provenance_bytes",
    ):
        value = evidence.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            failures.append(f"{field}_invalid")

    decoding = evidence.get("decoding")
    if not isinstance(decoding, dict) or set(decoding) != {"num_beams", "do_sample"}:
        failures.append("decoding_invalid")
    elif (
        isinstance(decoding.get("num_beams"), bool)
        or not isinstance(decoding.get("num_beams"), int)
        or decoding["num_beams"] < 1
        or decoding.get("do_sample") is not False
    ):
        failures.append("decoding_invalid")

    samples = evidence.get("samples")
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples < QUANTIZATION_PARITY_MIN_SAMPLES
        or samples > 100_000
    ):
        failures.append("samples_insufficient")

    bootstrap = evidence.get("bootstrap")
    if not isinstance(bootstrap, dict):
        failures.append("bootstrap_missing")
    else:
        if set(bootstrap) != {"unit", "clusters", "repeats", "seed"}:
            failures.append("bootstrap_fields_invalid")
        unit = bootstrap.get("unit")
        expected_units = {"group", "speaker"} if task == "asr" else {"row"}
        if unit not in expected_units:
            failures.append("bootstrap_unit_invalid")
        clusters = bootstrap.get("clusters")
        minimum_clusters = 32 if task == "asr" else QUANTIZATION_PARITY_MIN_SAMPLES
        if (
            isinstance(clusters, bool)
            or not isinstance(clusters, int)
            or clusters < minimum_clusters
            or (isinstance(samples, int) and clusters > samples)
        ):
            failures.append("bootstrap_clusters_insufficient")
        elif task == "mt" and clusters != samples:
            failures.append("bootstrap_clusters_mismatch")
        repeats = bootstrap.get("repeats")
        if (
            isinstance(repeats, bool)
            or not isinstance(repeats, int)
            or repeats < QUANTIZATION_PARITY_BOOTSTRAP_REPEATS
        ):
            failures.append("bootstrap_repeats_insufficient")
        if bootstrap.get("seed") != 20261005:
            failures.append("bootstrap_seed_invalid")

    metrics = evidence.get("metrics")
    expected_metrics = set(QUANTIZATION_PARITY_METRICS.get(task, ()))
    if not isinstance(metrics, dict) or set(metrics) != expected_metrics:
        failures.append("metrics_set_invalid")
    else:
        for name in sorted(expected_metrics):
            metric = metrics.get(name)
            prefix = f"metric_{name}"
            if not isinstance(metric, dict):
                failures.append(f"{prefix}_invalid")
                continue
            if set(metric) != {
                "reference",
                "quantized",
                "relative_degradation",
                "relative_degradation_bootstrap_95ci",
                "maximum_relative_degradation",
            }:
                failures.append(f"{prefix}_fields_invalid")
            reference = _parity_number(metric.get("reference"))
            quantized = _parity_number(metric.get("quantized"))
            relative = _parity_number(metric.get("relative_degradation"))
            if reference is None or reference < 0:
                failures.append(f"{prefix}_reference_invalid")
            if quantized is None or quantized < 0:
                failures.append(f"{prefix}_quantized_invalid")
            if relative is None:
                failures.append(f"{prefix}_relative_degradation_invalid")
            if metric.get("maximum_relative_degradation") != (
                QUANTIZATION_PARITY_MAX_RELATIVE_DEGRADATION
            ):
                failures.append(f"{prefix}_threshold_invalid")
            interval = metric.get("relative_degradation_bootstrap_95ci")
            parsed_interval: tuple[float, float] | None = None
            if isinstance(interval, list) and len(interval) == 2:
                lower = _parity_number(interval[0])
                upper = _parity_number(interval[1])
                if lower is not None and upper is not None and lower <= upper:
                    parsed_interval = (lower, upper)
            if parsed_interval is None:
                failures.append(f"{prefix}_bootstrap_95ci_invalid")
            elif parsed_interval[1] > QUANTIZATION_PARITY_MAX_RELATIVE_DEGRADATION:
                failures.append(f"{prefix}_bootstrap_95ci_above_policy")
            if reference is not None and quantized is not None:
                calculated = _relative_metric_degradation(
                    reference,
                    quantized,
                    greater_is_better=task == "mt",
                )
                if calculated is None:
                    failures.append(f"{prefix}_relative_degradation_undefined")
                elif relative is not None:
                    tolerance = max(1e-9, abs(calculated) * 1e-6)
                    if abs(relative - calculated) > tolerance:
                        failures.append(f"{prefix}_relative_degradation_mismatch")
                    elif relative > QUANTIZATION_PARITY_MAX_RELATIVE_DEGRADATION:
                        failures.append(f"{prefix}_relative_degradation_above_policy")

    slices = evidence.get("safety_slices")
    if not isinstance(slices, dict) or set(slices) != set(QUANTIZATION_PARITY_SLICES):
        failures.append("safety_slices_set_invalid")
    else:
        for name in QUANTIZATION_PARITY_SLICES:
            record = slices.get(name)
            prefix = f"safety_{name}"
            if not isinstance(record, dict):
                failures.append(f"{prefix}_invalid")
                continue
            if set(record) != {
                "samples",
                "reference_failures",
                "quantized_failures",
            }:
                failures.append(f"{prefix}_fields_invalid")
            slice_samples = record.get("samples")
            if (
                isinstance(slice_samples, bool)
                or not isinstance(slice_samples, int)
                or slice_samples < 1
                or (isinstance(samples, int) and slice_samples > samples)
            ):
                failures.append(f"{prefix}_samples_invalid")
            for field in ("reference_failures", "quantized_failures"):
                value = record.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value != 0:
                    failures.append(f"{prefix}_{field}_nonzero")
    return failures


def validate_deployment_report(
    report: dict[str, Any],
    expected_winners: list[dict[str, Any]],
    required_metrics: list[str],
    min_runs: int,
    project_root: Path = ROOT,
) -> tuple[bool, list[str]]:
    """Validate fresh physical-board evidence for every selected winner."""
    required_metrics = deployment_required_metrics(required_metrics)
    failures: list[str] = []
    if report.get("version") != 1:
        failures.append("version:invalid")
    if report.get("status") != "pass":
        failures.append("status:not_pass")
    if report.get("target") != "QCS6490":
        failures.append("target:not_qcs6490")
    if report.get("measurement_source") != "physical_board":
        failures.append("measurement_source:not_physical_board")

    selection = report.get("selection_comparison")
    if not isinstance(selection, dict) or set(selection) != {"path", "sha256"}:
        failures.append("selection_comparison:invalid")
    else:
        selection_sha = selection.get("sha256")
        selection_path: Path | None = None
        if not isinstance(selection_sha, str) or not SHA256_RE.fullmatch(
            selection_sha
        ):
            failures.append("selection_comparison:sha256_invalid")
        try:
            selection_path = resolve_regular_file_without_links(
                selection.get("path"),
                label="Selection comparison",
                maximum_bytes=MAX_BAKEOFF_REPORT_BYTES,
            )
        except FileNotFoundError:
            failures.append("selection_comparison:missing_file")
        except ValueError:
            failures.append("selection_comparison:path_unsafe")
        if selection_path is not None:
            try:
                read_json_mapping(
                    selection_path,
                    maximum_bytes=MAX_BAKEOFF_REPORT_BYTES,
                    label="Selection comparison",
                    expected_sha256=(
                        selection_sha
                        if isinstance(selection_sha, str)
                        and SHA256_RE.fullmatch(selection_sha)
                        else None
                    ),
                )
            except JsonDocumentDigestMismatch:
                failures.append("selection_comparison:sha256_mismatch")
            except (FileNotFoundError, RuntimeError, ValueError):
                failures.append("selection_comparison:invalid_json")

    gate = report.get("deployment_gate")
    expected_gate_metrics = deployment_required_metrics(required_metrics)
    if not isinstance(gate, dict):
        failures.append("deployment_gate:missing")
    else:
        if gate.get("required_metrics") != expected_gate_metrics:
            failures.append("deployment_gate:required_metrics_mismatch")
        if gate.get("minimum_measurement_runs") != min_runs:
            failures.append("deployment_gate:minimum_measurement_runs_mismatch")
        parity_policy = gate.get("quantization_parity")
        expected_parity_policy = {
            "maximum_relative_degradation": (
                QUANTIZATION_PARITY_MAX_RELATIVE_DEGRADATION
            ),
            "minimum_samples": QUANTIZATION_PARITY_MIN_SAMPLES,
            "minimum_bootstrap_repeats": QUANTIZATION_PARITY_BOOTSTRAP_REPEATS,
            "asr_minimum_independent_groups": 32,
            "required_zero_failure_slices": list(QUANTIZATION_PARITY_SLICES),
        }
        if parity_policy != expected_parity_policy:
            failures.append("deployment_gate:quantization_parity_policy_mismatch")

    measured_at = str(report.get("measured_at") or "")
    measured_time: datetime | None = None
    try:
        measured_time = datetime.fromisoformat(measured_at.replace("Z", "+00:00"))
        if measured_time.tzinfo is None:
            raise ValueError("timezone required")
    except ValueError:
        measured_time = None
        failures.append("measured_at:invalid")

    device = report.get("device")
    device_identity_sha = ""
    if not isinstance(device, dict):
        failures.append("device:missing")
    else:
        if str(device.get("chipset") or "").upper() != "QCS6490":
            failures.append("device.chipset:not_qcs6490")
        board = str(device.get("board") or "").strip()
        if not board:
            failures.append("device.board:missing")
        elif "6490" not in board.upper() and "RB3 GEN 2" not in board.upper():
            failures.append("device.board:not_qcs6490_board")
        if not str(device.get("os") or "").strip():
            failures.append("device.os:missing")

        identity_value = str(device.get("identity_evidence_path") or "").strip()
        identity_sha = str(device.get("identity_evidence_sha256") or "")
        device_identity_sha = identity_sha
        if not identity_value:
            failures.append("device.identity_evidence_path:missing")
        else:
            identity_root = (
                project_root / "data" / "reports" / "model_bakeoff" / "board-evidence"
            )
            try:
                identity_path = resolve_regular_file_under(
                    identity_value,
                    project_root=project_root,
                    allowed_root=identity_root,
                    label="QCS6490 identity evidence",
                    maximum_bytes=MAX_IDENTITY_EVIDENCE_BYTES,
                )
            except FileNotFoundError:
                failures.append("device.identity_evidence_path:missing_file")
                identity_path = None
            except ValueError:
                failures.append("device.identity_evidence_path:unsafe")
                identity_path = None
            if identity_path is not None:
                if identity_path.stat().st_size < 2:
                    failures.append("device.identity_evidence:invalid_size")
                else:
                    if not SHA256_RE.fullmatch(identity_sha):
                        failures.append("device.identity_evidence_sha256:invalid")
                    try:
                        identity = read_json_mapping(
                            identity_path,
                            maximum_bytes=MAX_IDENTITY_EVIDENCE_BYTES,
                            label="QCS6490 identity evidence",
                            expected_sha256=(
                                identity_sha
                                if SHA256_RE.fullmatch(identity_sha)
                                else None
                            ),
                        )
                    except JsonDocumentDigestMismatch:
                        failures.append("device.identity_evidence_sha256:mismatch")
                    except (FileNotFoundError, RuntimeError, ValueError):
                        failures.append("device.identity_evidence:invalid_json")
                    else:
                        for failure in qcs6490_identity_failures(identity):
                            failures.append(f"device.identity_evidence:{failure}")
                        evidence_board = str(identity.get("board_model") or "").strip()
                        if evidence_board and board != evidence_board:
                            failures.append("device.board:identity_mismatch")
                        evidence_architecture = str(
                            identity.get("architecture") or ""
                        ).strip()
                        if evidence_architecture and str(
                            device.get("architecture") or ""
                        ).strip() != evidence_architecture:
                            failures.append("device.architecture:identity_mismatch")
                        try:
                            captured_time = datetime.fromisoformat(
                                str(identity.get("captured_at") or "").replace(
                                    "Z", "+00:00"
                                )
                            )
                            if captured_time.tzinfo is None:
                                captured_time = None
                        except ValueError:
                            captured_time = None
                        if (
                            measured_time is not None
                            and captured_time is not None
                            and abs((measured_time - captured_time).total_seconds())
                            > 86_400
                        ):
                            failures.append("device.identity_evidence:not_same_session")

    expected_by_key = {
        (item["task"], item.get("direction")): item for item in expected_winners
    }
    records = report.get("winners")
    if not isinstance(records, list):
        return False, failures + ["winners:missing"]
    record_by_key: dict[tuple[str, str | None], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            failures.append("winners:invalid_record")
            continue
        task = str(record.get("task") or "")
        direction_value = record.get("direction")
        direction = str(direction_value) if direction_value is not None else None
        key = (task, direction)
        if key in record_by_key:
            failures.append(f"winner:{task}/{direction or 'vi'}:duplicate")
        record_by_key[key] = record
    if set(record_by_key) != set(expected_by_key):
        failures.append("winners:set_mismatch")

    for key, expected in expected_by_key.items():
        task, direction = key
        label = f"winner:{task}/{direction or 'vi'}"
        record = record_by_key.get(key)
        if record is None:
            continue
        if record.get("candidate_id") != expected["candidate_id"]:
            failures.append(f"{label}:candidate_mismatch")
        if record.get("adapter_manifest_sha256") != expected["adapter_manifest_sha256"]:
            failures.append(f"{label}:adapter_checksum_mismatch")
        artifact_sha = str(record.get("artifact_sha256") or "")
        if not SHA256_RE.fullmatch(artifact_sha):
            failures.append(f"{label}:artifact_sha256_invalid")
        parity_value = str(record.get("parity_evidence_path") or "").strip()
        parity_sha = str(record.get("parity_evidence_sha256") or "")
        parity: dict[str, Any] | None = None
        if not parity_value:
            failures.append(f"{label}:parity_evidence_path_missing")
        else:
            parity_root = (
                project_root
                / "data/reports/model_bakeoff/board-evidence/parity"
            )
            try:
                parity_path = resolve_regular_file_under(
                    parity_value,
                    project_root=project_root,
                    allowed_root=parity_root,
                    label="Quantization parity evidence",
                    maximum_bytes=MAX_QUANTIZATION_PARITY_EVIDENCE_BYTES,
                )
            except FileNotFoundError:
                failures.append(f"{label}:parity_evidence_missing")
                parity_path = None
            except ValueError as exc:
                suffix = "too_large" if "exceeds" in str(exc) else "path_unsafe"
                failures.append(f"{label}:parity_evidence_{suffix}")
                parity_path = None
            if parity_path is not None:
                if parity_path.stat().st_size < 2:
                    failures.append(f"{label}:parity_evidence_invalid_size")
                    parity_path = None
            if parity_path is not None:
                if not SHA256_RE.fullmatch(parity_sha):
                    failures.append(f"{label}:parity_evidence_sha256_invalid")
                try:
                    loaded_parity = read_json_mapping(
                        parity_path,
                        maximum_bytes=MAX_QUANTIZATION_PARITY_EVIDENCE_BYTES,
                        label="Quantization parity evidence",
                        expected_sha256=(
                            parity_sha if SHA256_RE.fullmatch(parity_sha) else None
                        ),
                    )
                except JsonDocumentDigestMismatch:
                    failures.append(f"{label}:parity_evidence_sha256_mismatch")
                except (FileNotFoundError, RuntimeError, ValueError):
                    failures.append(f"{label}:parity_evidence_invalid_json")
                else:
                    parity = loaded_parity
        if parity is not None:
            for parity_failure in quantization_parity_evidence_failures(
                parity,
                expected=expected,
                artifact_sha256=artifact_sha,
                artifact_bytes=record.get("model_bytes"),
            ):
                failures.append(f"{label}:parity_evidence_{parity_failure}")
            parity_fields = {
                "evaluated_at": "parity_evaluated_at",
                "manifest_sha256": "parity_manifest_sha256",
                "reference_predictions_sha256": (
                    "parity_reference_predictions_sha256"
                ),
                "reference_provenance_sha256": (
                    "parity_reference_provenance_sha256"
                ),
                "quantized_predictions_sha256": (
                    "parity_quantized_predictions_sha256"
                ),
                "quantized_provenance_sha256": (
                    "parity_quantized_provenance_sha256"
                ),
                "decoding": "parity_decoding",
                "samples": "parity_samples",
                "bootstrap": "parity_bootstrap",
                "metrics": "parity_metrics",
                "safety_slices": "parity_safety_slices",
            }
            for evidence_field, record_field in parity_fields.items():
                if parity.get(evidence_field) != record.get(record_field):
                    failures.append(
                        f"{label}:parity_evidence_{evidence_field}_mismatch"
                    )
        measurement_value = str(record.get("measurement_evidence_path") or "").strip()
        measurement_sha = str(record.get("measurement_evidence_sha256") or "")
        measurement: dict[str, Any] | None = None
        if not measurement_value:
            failures.append(f"{label}:measurement_evidence_path_missing")
        else:
            measurement_root = (
                project_root
                / "data/reports/model_bakeoff/board-evidence/measurements"
            )
            try:
                measurement_path = resolve_regular_file_under(
                    measurement_value,
                    project_root=project_root,
                    allowed_root=measurement_root,
                    label="Physical measurement evidence",
                    maximum_bytes=MAX_DEPLOYMENT_MEASUREMENT_BYTES,
                )
            except FileNotFoundError:
                failures.append(f"{label}:measurement_evidence_missing")
                measurement_path = None
            except ValueError as exc:
                suffix = "too_large" if "exceeds" in str(exc) else "path_unsafe"
                failures.append(f"{label}:measurement_evidence_{suffix}")
                measurement_path = None
            if measurement_path is not None:
                if measurement_path.stat().st_size < 2:
                    failures.append(f"{label}:measurement_evidence_invalid_size")
                    measurement_path = None
            if measurement_path is not None:
                if not SHA256_RE.fullmatch(measurement_sha):
                    failures.append(f"{label}:measurement_evidence_sha256_invalid")
                try:
                    loaded_measurement = read_json_mapping(
                        measurement_path,
                        maximum_bytes=MAX_DEPLOYMENT_MEASUREMENT_BYTES,
                        label="Physical measurement evidence",
                        expected_sha256=(
                            measurement_sha
                            if SHA256_RE.fullmatch(measurement_sha)
                            else None
                        ),
                    )
                except JsonDocumentDigestMismatch:
                    failures.append(f"{label}:measurement_evidence_sha256_mismatch")
                except (FileNotFoundError, RuntimeError, ValueError):
                    failures.append(f"{label}:measurement_evidence_invalid_json")
                else:
                    measurement = loaded_measurement
        if measurement is not None:
            if measurement.get("version") != 1:
                failures.append(f"{label}:measurement_evidence_version_invalid")
            if measurement.get("capture_source") != "physical_qcs6490":
                failures.append(f"{label}:measurement_evidence_source_invalid")
            bindings = {
                "task": record.get("task"),
                "direction": record.get("direction"),
                "candidate_id": expected["candidate_id"],
                "adapter_manifest_sha256": expected["adapter_manifest_sha256"],
                "identity_evidence_sha256": device_identity_sha,
                "artifact_sha256": artifact_sha,
                "artifact_bytes": record.get("model_bytes"),
            }
            for field, bound_value in bindings.items():
                if measurement.get(field) != bound_value:
                    failures.append(f"{label}:measurement_evidence_{field}_mismatch")
            measurement_fields = (
                "latency_samples_ms",
                "power_sensor",
                "power_samples_mw",
                "temperature_sensor",
                "temperature_samples_c",
                "peak_ram_bytes",
                "peak_vram_bytes",
            )
            for field in measurement_fields:
                if measurement.get(field) != record.get(field):
                    failures.append(f"{label}:measurement_evidence_{field}_mismatch")
            if measurement.get("captured_at") != record.get(
                "measurement_captured_at"
            ):
                failures.append(
                    f"{label}:measurement_evidence_captured_at_mismatch"
                )
            captured_at = str(measurement.get("captured_at") or "")
            try:
                captured_time = datetime.fromisoformat(
                    captured_at.replace("Z", "+00:00")
                )
                if captured_time.tzinfo is None:
                    raise ValueError("timezone required")
            except ValueError:
                captured_time = None
                failures.append(f"{label}:measurement_evidence_timestamp_invalid")
            if (
                measured_time is not None
                and captured_time is not None
                and abs((measured_time - captured_time).total_seconds()) > 86_400
            ):
                failures.append(f"{label}:measurement_evidence_not_same_session")
        runs = record.get("measurement_runs")
        if isinstance(runs, bool) or not isinstance(runs, int) or runs < min_runs:
            failures.append(f"{label}:insufficient_measurement_runs")

        latency_samples = record.get("latency_samples_ms")
        parsed_samples: list[float] = []
        if not isinstance(latency_samples, list):
            failures.append(f"{label}:latency_samples_missing")
        else:
            for value in latency_samples:
                try:
                    sample = float(value)
                except (TypeError, ValueError):
                    sample = math.nan
                if not math.isfinite(sample) or sample <= 0:
                    parsed_samples = []
                    failures.append(f"{label}:latency_samples_invalid")
                    break
                parsed_samples.append(sample)
            if parsed_samples and (
                len(parsed_samples) < min_runs or runs != len(parsed_samples)
            ):
                failures.append(f"{label}:latency_sample_count_mismatch")

        sensor_samples: dict[str, list[float]] = {}
        sensor_contracts = {
            "power": ("power_samples_mw", "power_sensor"),
            "temperature": ("temperature_samples_c", "temperature_sensor"),
        }
        for sensor, (sample_key, source_key) in sensor_contracts.items():
            source = record.get(source_key)
            if not isinstance(source, str) or not source.strip():
                failures.append(f"{label}:{source_key}_missing")
            values = record.get(sample_key)
            parsed: list[float] = []
            if not isinstance(values, list):
                failures.append(f"{label}:{sample_key}_missing")
            else:
                for value in values:
                    try:
                        sample = float(value)
                    except (TypeError, ValueError):
                        sample = math.nan
                    invalid = not math.isfinite(sample)
                    if sensor == "power":
                        invalid = invalid or sample <= 0
                    else:
                        invalid = invalid or sample <= -273.15
                    if invalid:
                        parsed = []
                        failures.append(f"{label}:{sample_key}_invalid")
                        break
                    parsed.append(sample)
                if parsed and len(parsed) < min_runs:
                    failures.append(f"{label}:{sample_key}_insufficient")
            sensor_samples[sensor] = parsed

        numeric: dict[str, float] = {}
        for metric in required_metrics:
            value = record.get(metric)
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = math.nan
            if not math.isfinite(number):
                failures.append(f"{label}:{metric}_invalid")
                continue
            if metric.endswith("_bytes") and not number.is_integer():
                failures.append(f"{label}:{metric}_invalid")
                continue
            if metric == "peak_vram_bytes":
                if number < 0:
                    failures.append(f"{label}:{metric}_invalid")
            elif metric == "temperature_peak_c":
                if number <= -273.15:
                    failures.append(f"{label}:{metric}_invalid")
            elif number <= 0:
                failures.append(f"{label}:{metric}_invalid")
            numeric[metric] = number
        if (
            "latency_p50_ms" in numeric
            and "latency_p95_ms" in numeric
            and numeric["latency_p95_ms"] < numeric["latency_p50_ms"]
        ):
            failures.append(f"{label}:latency_percentiles_reversed")
        if parsed_samples and "latency_p50_ms" in numeric and "latency_p95_ms" in numeric:
            calculated = {
                "latency_p50_ms": percentile_linear(parsed_samples, 0.50),
                "latency_p95_ms": percentile_linear(parsed_samples, 0.95),
            }
            for metric, expected_value in calculated.items():
                tolerance = max(1e-6, abs(expected_value) * 1e-6)
                if abs(numeric[metric] - expected_value) > tolerance:
                    failures.append(f"{label}:{metric}_does_not_match_samples")
        power_samples = sensor_samples["power"]
        temperature_samples = sensor_samples["temperature"]
        calculated_sensor_metrics: dict[str, float] = {}
        if power_samples:
            calculated_sensor_metrics.update(
                {
                    "power_avg_mw": sum(power_samples) / len(power_samples),
                    "power_p95_mw": percentile_linear(power_samples, 0.95),
                }
            )
        if temperature_samples:
            calculated_sensor_metrics["temperature_peak_c"] = max(
                temperature_samples
            )
        for metric, expected_value in calculated_sensor_metrics.items():
            if metric not in numeric:
                continue
            tolerance = max(1e-6, abs(expected_value) * 1e-6)
            if abs(numeric[metric] - expected_value) > tolerance:
                failures.append(f"{label}:{metric}_does_not_match_samples")

        artifact_value = str(record.get("artifact_path") or "").strip()
        if not artifact_value:
            failures.append(f"{label}:artifact_path_missing")
            continue
        try:
            artifact_path = resolve_regular_file_under(
                artifact_value,
                project_root=project_root,
                allowed_root=project_root / "models",
                label="Compiled deployment artifact",
                maximum_bytes=MAX_DEPLOYMENT_ARTIFACT_BYTES,
            )
        except FileNotFoundError:
            failures.append(f"{label}:artifact_missing")
            artifact_path = None
        except ValueError:
            failures.append(f"{label}:artifact_path_unsafe")
            artifact_path = None
        if artifact_path is not None:
            try:
                observed_sha, observed_bytes = sha256_stable_regular_file(
                    artifact_path,
                    maximum_bytes=MAX_DEPLOYMENT_ARTIFACT_BYTES,
                    label="Compiled deployment artifact",
                )
            except (FileNotFoundError, RuntimeError, ValueError):
                failures.append(f"{label}:artifact_unstable")
            else:
                if SHA256_RE.fullmatch(artifact_sha) and observed_sha != artifact_sha:
                    failures.append(f"{label}:artifact_checksum_mismatch")
                if "model_bytes" in numeric and observed_bytes != int(
                    numeric["model_bytes"]
                ):
                    failures.append(f"{label}:model_bytes_mismatch")
    return not failures, failures


def validate_deployment_draft(
    draft: dict[str, Any],
    expected_winners: list[dict[str, Any]],
    comparison_path: Path,
) -> tuple[bool, list[str]]:
    """Verify that pending physical measurements remain bound to this selection."""
    failures: list[str] = []
    if draft.get("version") != 1:
        failures.append("version:invalid")
    if draft.get("status") != "pending_physical_measurement":
        failures.append("status:not_pending")
    if draft.get("target") != "QCS6490":
        failures.append("target:not_qcs6490")
    if draft.get("measurement_source") != "physical_board":
        failures.append("measurement_source:not_physical_board")
    selection = draft.get("selection_comparison")
    if not isinstance(selection, dict):
        failures.append("selection_comparison:missing")
    else:
        try:
            resolved_comparison = resolve_regular_file_without_links(
                comparison_path,
                label="Selection comparison",
                maximum_bytes=MAX_BAKEOFF_REPORT_BYTES,
            )
            comparison_sha256, _ = sha256_stable_regular_file(
                resolved_comparison,
                maximum_bytes=MAX_BAKEOFF_REPORT_BYTES,
                label="Selection comparison",
            )
        except (FileNotFoundError, RuntimeError, ValueError):
            failures.append("selection_comparison:file_invalid")
            resolved_comparison = None
            comparison_sha256 = None
        if resolved_comparison is not None and selection.get("path") != str(
            resolved_comparison
        ):
            failures.append("selection_comparison:path_mismatch")
        if (
            comparison_sha256 is not None
            and selection.get("sha256") != comparison_sha256
        ):
            failures.append("selection_comparison:checksum_mismatch")

    expected_by_key = {
        (item["task"], item.get("direction")): item for item in expected_winners
    }
    records = draft.get("winners")
    if not isinstance(records, list):
        return False, failures + ["winners:missing"]
    record_by_key: dict[tuple[str, str | None], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            failures.append("winners:invalid_record")
            continue
        direction_value = record.get("direction")
        key = (
            str(record.get("task") or ""),
            str(direction_value) if direction_value is not None else None,
        )
        if key in record_by_key:
            failures.append(f"winner:{key[0]}/{key[1] or 'vi'}:duplicate")
        record_by_key[key] = record
    if set(record_by_key) != set(expected_by_key):
        failures.append("winners:set_mismatch")
    for key, expected in expected_by_key.items():
        record = record_by_key.get(key)
        if record is None:
            continue
        label = f"winner:{key[0]}/{key[1] or 'vi'}"
        if record.get("candidate_id") != expected["candidate_id"]:
            failures.append(f"{label}:candidate_mismatch")
        if record.get("adapter_manifest_sha256") != expected["adapter_manifest_sha256"]:
            failures.append(f"{label}:adapter_checksum_mismatch")
        for sample_key in (
            "latency_samples_ms",
            "power_samples_mw",
            "temperature_samples_c",
        ):
            if not isinstance(record.get(sample_key), list):
                failures.append(f"{label}:{sample_key}_missing")
        for source_key in ("power_sensor", "temperature_sensor"):
            if not isinstance(record.get(source_key), str):
                failures.append(f"{label}:{source_key}_missing")
        if not isinstance(record.get("parity_evidence_path"), str):
            failures.append(f"{label}:parity_evidence_path_missing")
        if not isinstance(record.get("measurement_evidence_path"), str):
            failures.append(f"{label}:measurement_evidence_path_missing")
    return not failures, failures


def freeze_candidate_a(current: dict[str, Any], output: Path) -> dict[str, Any]:
    adapters = {"mt": current.get("mt_adapter"), "asr": current.get("asr_adapter")}
    missing = [task for task, value in adapters.items() if not value]
    if missing:
        raise ValueError(f"Current program lacks Candidate A adapters: {missing}")
    frozen = {
        "created_at": utc_now(),
        "source_program_state": current,
        "promotion_allowed": False,
        "policy": "Candidate A is a frozen reference; passing its old gate does not promote it.",
        "adapters": {task: tree_manifest(Path(value)) for task, value in adapters.items()},
    }
    atomic_json(output, frozen)
    return frozen


def validate_candidate_a_freeze(
    current: dict[str, Any], output: Path
) -> dict[str, Any]:
    """Re-hash an existing immutable freeze before resume or GPU work."""
    if output.is_symlink() or not output.is_file():
        raise FileNotFoundError(f"Candidate A freeze is not a regular file: {output}")
    frozen = read_json_mapping(
        output,
        maximum_bytes=MAX_BAKEOFF_STATE_BYTES,
        label="Candidate A freeze",
    )
    if frozen.get("promotion_allowed") is not False:
        raise ValueError("Candidate A freeze cannot authorize promotion")
    source = frozen.get("source_program_state")
    adapters = frozen.get("adapters")
    if not isinstance(source, dict) or not isinstance(adapters, dict):
        raise ValueError("Candidate A freeze is incomplete")
    for task in ("mt", "asr"):
        current_adapter = str(current.get(f"{task}_adapter") or "")
        if not current_adapter or source.get(f"{task}_adapter") != current_adapter:
            raise ValueError(f"Candidate A {task} freeze is bound to a different adapter")
        recorded = adapters.get(task)
        actual = tree_manifest(Path(current_adapter))
        if recorded != actual:
            raise ValueError(f"Candidate A {task} adapter changed after freeze")
    return frozen


def critical_safety_pass(report: dict[str, Any], required: list[str]) -> tuple[bool, list[str]]:
    categories = report.get("categories") or {}
    failures = []
    for name in required:
        record = categories.get(name)
        try:
            samples = int(record.get("samples", 0)) if record else 0
        except (TypeError, ValueError):
            samples = 0
        if not record or samples < 1:
            failures.append(f"{name}:missing")
            continue
        try:
            failure_rate = float(record.get("safety_failure_rate", 1.0))
        except (TypeError, ValueError):
            failure_rate = math.nan
        if not math.isfinite(failure_rate):
            failures.append(f"{name}:invalid")
        elif failure_rate != 0.0:
            failures.append(f"{name}:nonzero")
    return not failures, failures


def normalized_interval(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        lower, upper = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        return None
    return lower, upper


def interval_stronger(
    challenger: Any, baseline: Any, *, greater_is_better: bool
) -> bool:
    challenger_bounds = normalized_interval(challenger)
    baseline_bounds = normalized_interval(baseline)
    if challenger_bounds is None or baseline_bounds is None:
        return False
    if greater_is_better:
        return challenger_bounds[0] > baseline_bounds[1]
    return challenger_bounds[1] < baseline_bounds[0]


def multi_metric_stronger(
    challenger: dict[str, Any], baseline: dict[str, Any]
) -> bool:
    """Require a 95% CI win and reject a significant regression on any CI metric."""
    if not challenger.get("evidence_valid") or not baseline.get("evidence_valid"):
        return False
    challenger_metrics = challenger.get("metrics") or {}
    baseline_metrics = baseline.get("metrics") or {}
    if set(challenger_metrics) != set(baseline_metrics) or not challenger_metrics:
        return False
    improvement = False
    for name, challenger_metric in challenger_metrics.items():
        baseline_metric = baseline_metrics[name]
        challenger_interval = challenger_metric.get("confidence_interval_95")
        baseline_interval = baseline_metric.get("confidence_interval_95")
        if challenger_interval is None and baseline_interval is None:
            continue
        if challenger_interval is None or baseline_interval is None:
            return False
        greater = bool(challenger_metric["greater_is_better"])
        if interval_stronger(
            baseline_interval,
            challenger_interval,
            greater_is_better=greater,
        ):
            return False
        if interval_stronger(
            challenger_interval,
            baseline_interval,
            greater_is_better=greater,
        ):
            improvement = True
    return improvement


def run_stage(
    name: str,
    command: list[str],
    state_path: Path,
    state: dict[str, Any],
    log_path: Path,
    expected_outputs: list[Path],
    resource_limits: dict[str, Any] | None = None,
) -> None:
    validate_loaded_runner_generation()
    digest = command_digest(command)
    previous = state.setdefault("stages", {}).get(name)
    assert_interrupted_stage_is_not_live(previous)
    if previous and previous.get("status") == "complete":
        command_changed = previous.get("command_sha256") != digest
        recovery_only_upgrade = (
            command_changed
            and resume_scoring_command_compatible(previous.get("command"), command)
        )
        immutable_command_compatibility = (
            command_changed
            and (
                interpreter_alias_only_command_change(previous.get("command"), command)
                or runner_generation_only_command_change(
                    previous.get("command"), command
                )
            )
        )
        if command_changed and not (
            recovery_only_upgrade or immutable_command_compatibility
        ):
            raise ValueError(f"Cannot resume {name}: command changed")
        current_evidence = output_evidence(expected_outputs)
        recorded_evidence = previous.get("output_evidence")
        if immutable_command_compatibility and recorded_evidence is None:
            raise ValueError(f"Cannot resume {name}: output evidence missing")
        if recorded_evidence is not None and recorded_evidence != current_evidence:
            raise ValueError(f"Cannot resume {name}: output evidence changed")
        if immutable_command_compatibility:
            return
        if not recovery_only_upgrade:
            return
        # A legacy benchmark did not bind its report to verified predictions.
        # Re-run only the CPU scoring path; --resume-scoring verifies the exact
        # prediction checkpoint and skips model loading and inference.
    previous_attempt = 0
    attempt_history: list[dict[str, Any]] = []
    if isinstance(previous, dict):
        raw_attempt = previous.get("attempt", 1)
        if isinstance(raw_attempt, bool) or not isinstance(raw_attempt, int) or raw_attempt < 1:
            raise ValueError(f"Stage {name} has an invalid attempt counter")
        previous_attempt = raw_attempt
        raw_history = previous.get("attempt_history", [])
        if not isinstance(raw_history, list) or any(
            not isinstance(item, dict) for item in raw_history
        ):
            raise ValueError(f"Stage {name} has invalid attempt history")
        attempt_history = list(raw_history)
        if previous.get("status") in {"error", "running"}:
            attempt_history.append(
                {
                    "attempt": previous_attempt,
                    "status": previous.get("status"),
                    "return_code": previous.get("return_code"),
                    "started_at": previous.get("started_at"),
                    "completed_at": previous.get("completed_at"),
                    "command_sha256": previous.get("command_sha256"),
                    "log": previous.get("log"),
                }
            )
    state["stage"] = name
    stage_record = {
        "status": "running",
        "attempt": previous_attempt + 1,
        "attempt_history": attempt_history,
        "started_at": utc_now(),
        "command": command,
        "command_sha256": digest,
        "log": str(log_path),
    }
    if resource_limits is not None:
        stage_record.update(
            {
                "pid": os.getpid(),
                "waiting_for_gpu_capacity": True,
            }
        )
    state["stages"][name] = stage_record
    atomic_json(state_path, state)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if resource_limits is not None:
        spawn_capacity_path = log_path.with_suffix(".pre_spawn.jsonl")
        spawn_capacity = wait_for_gpu_spawn_capacity(
            resource_limits,
            spawn_capacity_path,
        )
        state["stages"][name].update(
            {
                "spawn_capacity": spawn_capacity,
                "waiting_for_gpu_capacity": False,
            }
        )
        atomic_json(state_path, state)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        state["stages"][name]["pid"] = process.pid
        state["stages"][name]["process_group"] = process.pid
        atomic_json(state_path, state)
        if resource_limits is None:
            return_code = process.wait()
        else:
            from scripts.run_gpu_rounds import monitor_process

            monitor_path = log_path.with_suffix(".resources.jsonl")
            monitor = monitor_process(process, monitor_path, resource_limits)
            return_code = int(monitor["return_code"])
    record = state["stages"][name]
    record.update({"return_code": return_code, "completed_at": utc_now()})
    if return_code or not all(path.exists() for path in expected_outputs):
        record["status"] = "error"
        atomic_json(state_path, state)
        raise StageExecutionError(f"Stage {name} failed; see {log_path}")
    try:
        record["output_evidence"] = output_evidence(expected_outputs)
    except (FileNotFoundError, ValueError):
        record["status"] = "error"
        atomic_json(state_path, state)
        raise
    record["status"] = "complete"
    atomic_json(state_path, state)


def benchmark_command(
    python: str,
    task: str,
    candidate: dict[str, Any],
    manifest: Path,
    output_dir: Path,
    stem: str,
    adapter: Path | None,
    direction: str | None,
) -> list[str]:
    command = [
        python,
        str(ROOT / "scripts/run_baseline_benchmarks.py"),
        "--task",
        task,
        "--model",
        candidate["model"],
        "--model-revision",
        candidate["revision"],
        "--manifest",
        str(manifest),
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
        "--bakeoff-runner-sha256",
        LOADED_RUNNER_SHA256,
    ]
    if adapter:
        command += ["--adapter", str(adapter)]
    if task == "mt":
        command += [
            "--mt-model-family",
            candidate["model_family"],
            "--mt-direction",
            str(direction or "joint"),
        ]
    else:
        command += ["--language", "vi"]
    return command


def candidate_units(config: dict[str, Any], task: str) -> list[dict[str, Any]]:
    units = []
    for candidate in config["candidates"][task]:
        if candidate["role"] == "candidate_a":
            continue
        directions = candidate.get("directions") or [None]
        for direction in directions:
            unit = deepcopy(candidate)
            unit["direction"] = direction
            unit["unit_id"] = candidate["id"] + (f"__{direction}" if direction else "")
            units.append(unit)
    return units


def write_runtime_round_config(
    path: Path,
    config: dict[str, Any],
    task: str,
    candidate: dict[str, Any],
    direction: str | None,
    round_name: str,
    steps: int,
    profile: dict[str, Any] | None = None,
) -> None:
    halving = config["successive_halving"]
    limit_train = (
        int(halving["pilot"]["max_train_samples"])
        if round_name == "pilot"
        else 0
    )
    common = {
        "base_model": candidate["model"],
        "base_model_revision": candidate["revision"],
        "train_manifest": str(ROOT / config["data"]["train"][task]),
        "validation_manifest": str(ROOT / config["data"]["selection_dev"][task]["path"]),
        "method": "lora",
        "precision": "bf16",
        "batch_size": 2 if task == "mt" else 1,
        "eval_batch_size": 2 if task == "mt" else 1,
        "gradient_accumulation_steps": 16 if task == "mt" else 32,
        "clinical_oversample_factor": 2,
        "limit_train": limit_train,
        "limit_validation": 0,
        "eval_steps": 100 if steps > 0 else 250,
        "save_steps": 100 if steps > 0 else 250,
        "logging_steps": 25,
    }
    if task == "mt":
        common.update({"model_family": candidate["model_family"], "direction": direction})
    selected_profile = profile or hyperparameter_profiles(config, task)[0]
    runtime = {
        "version": 1,
        "limits": config["resources"],
        "tasks": {
            task: {
                "module": (
                    "src.training.finetune_mt_medical"
                    if task == "mt"
                    else "src.training.finetune_whisper_vi"
                ),
                "selection_metric": "eval_loss" if task == "mt" else "eval_wer",
                "greater_is_better": False,
                "common_args": common,
                "rounds": [
                    {
                        "name": round_name,
                        "max_steps": steps,
                        "epochs": 1.0 if task == "mt" else 5.0,
                        "learning_rate": selected_profile["learning_rate"],
                        "lora_rank": selected_profile["lora_rank"],
                        "lora_alpha": selected_profile["lora_alpha"],
                        "lora_dropout": selected_profile["lora_dropout"],
                    }
                ],
            }
        },
    }
    rendered = yaml.safe_dump(runtime, sort_keys=False, allow_unicode=True)
    if path.is_file() and path.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"Runtime config changed during resume: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def completed_adapter(output_root: Path, task: str) -> Path:
    runs = sorted(output_root.glob(f"{task}-*"), key=lambda path: path.stat().st_mtime)
    complete = []
    for run in runs:
        summary_path = run / "summary.json"
        if not summary_path.is_file():
            continue
        summary = read_json_mapping(
            summary_path,
            maximum_bytes=MAX_BAKEOFF_STATE_BYTES,
            label=f"{task} training summary",
        )
        if summary.get("status") == "complete":
            complete.append((run, summary))
    if len(complete) != 1:
        raise ValueError(f"Expected one complete {task} run under {output_root}, found {len(complete)}")
    run, _ = complete[0]
    evidence = verified_training_result(output_root, task=task)
    if Path(evidence["run_root"]).resolve() != run.resolve():
        raise ValueError(f"Verified training evidence selected an unexpected run: {run}")
    return Path(evidence["adapter"])


def verified_resume_checkpoint(output_root: Path, task: str) -> Path | None:
    """Return the newest immutable checkpoint from an interrupted GPU stage.

    Resuming is allowed only when the watcher index, archive, checksum sidecar,
    content manifest, Trainer state, and the live checkpoint tree all agree.
    This keeps optimizer/scheduler progress without trusting a partial or
    replaced directory merely because it has a ``checkpoint-*`` name.
    """
    root = output_root.resolve()
    candidates: list[tuple[int, int, Path]] = []
    for run_root in sorted(root.glob(f"{task}-*")):
        resolved_run = run_root.resolve()
        for index_path in sorted(
            run_root.glob("checkpoint-archives/*/checkpoint_archives.json")
        ):
            try:
                index = read_json_mapping(
                    index_path,
                    maximum_bytes=MAX_BAKEOFF_STATE_BYTES,
                    label="Checkpoint archive index",
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                raise ValueError(f"Invalid checkpoint index: {index_path}") from exc
            records = index.get("checkpoints")
            if not isinstance(records, list):
                raise ValueError(f"Checkpoint index has no checkpoints list: {index_path}")
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError(f"Invalid checkpoint record: {index_path}")
                archive = Path(str(record.get("archive", ""))).resolve()
                checkpoint = Path(str(record.get("checkpoint", ""))).resolve()
                if archive.parent != index_path.parent.resolve():
                    raise ValueError(f"Checkpoint archive escaped its index directory: {archive}")
                try:
                    checkpoint.relative_to(resolved_run)
                except ValueError as exc:
                    raise ValueError(
                        f"Checkpoint escaped its training run: {checkpoint}"
                    ) from exc
                verification = verify_checkpoint_download(archive, index_path)
                complete = completed_checkpoint(checkpoint)
                if complete is None or complete[0] != verification["checkpoint_step"]:
                    raise ValueError(f"Live checkpoint is incomplete: {checkpoint}")
                manifest = verify_evidence_archive(archive)
                live_records = [
                    {key: item[key] for key in ("path", "bytes", "sha256")}
                    for item in checkpoint_file_records(checkpoint)
                ]
                if live_records != manifest["files"]:
                    raise ValueError(
                        f"Live checkpoint does not match verified archive: {checkpoint}"
                    )
                candidates.append(
                    (
                        int(verification["checkpoint_step"]),
                        run_root.stat().st_mtime_ns,
                        checkpoint,
                    )
                )
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def train_unit(
    *,
    unit: dict[str, Any],
    task: str,
    round_name: str,
    steps: int,
    config: dict[str, Any],
    python: str,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
    profile: dict[str, Any],
) -> Path:
    unit_id = unit["unit_id"]
    profile_id = str(profile["id"])
    output_root = state_dir / "training" / task / unit_id / round_name / profile_id
    runtime_config = (
        state_dir
        / "runtime-configs"
        / f"{task}-{unit_id}-{round_name}-{profile_id}.yaml"
    )
    write_runtime_round_config(
        runtime_config,
        config,
        task,
        unit,
        unit.get("direction"),
        round_name,
        steps,
        profile,
    )
    base_command = [
        python,
        str(ROOT / "scripts/run_gpu_rounds.py"),
        "--task",
        task,
        "--config",
        str(runtime_config),
        "--output-root",
        str(output_root),
    ]
    stage = f"{task}_{unit_id}_{round_name}_{profile_id}_train"
    while True:
        command = list(base_command)
        previous = state.get("stages", {}).get(stage)
        previous_attempt = (
            int(previous.get("attempt", 1))
            if isinstance(previous, dict)
            and previous.get("status") in {"error", "running"}
            else 0
        )
        if isinstance(previous, dict) and previous.get("status") in {"error", "running"}:
            resume_checkpoint = verified_resume_checkpoint(output_root, task)
            if resume_checkpoint is not None:
                command.extend(
                    [
                        "--round-name",
                        round_name,
                        "--resume-from-checkpoint",
                        str(resume_checkpoint),
                    ]
                )
        attempt = previous_attempt + 1
        log_name = f"{stage}.log" if attempt == 1 else f"{stage}.attempt-{attempt}.log"
        try:
            run_stage(
                stage,
                command,
                state_path,
                state,
                state_dir / "logs" / log_name,
                [output_root],
            )
            break
        except StageExecutionError:
            current = state.get("stages", {}).get(stage, {})
            if int(current.get("attempt", 0)) >= TRAIN_STAGE_MAX_ATTEMPTS:
                raise
            if verified_resume_checkpoint(output_root, task) is None:
                raise
    return completed_adapter(output_root, task)


def benchmark_unit(
    *,
    unit: dict[str, Any],
    task: str,
    label: str,
    adapter: Path | None,
    config: dict[str, Any],
    python: str,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
    direction: str | None,
) -> dict[str, Any]:
    output_dir = state_dir / "selection-reports" / task / unit["unit_id"] / label
    stem = "selection"
    report_path = output_dir / f"{stem}.json"
    command = benchmark_command(
        python,
        task,
        unit,
        ROOT / config["data"]["selection_dev"][task]["path"],
        output_dir,
        stem,
        adapter,
        direction,
    )
    stage = f"{task}_{unit['unit_id']}_{label}_selection"
    run_stage(
        stage,
        command,
        state_path,
        state,
        state_dir / "logs" / f"{stage}.log",
        [report_path],
        resource_limits=config["resources"],
    )
    return read_json_mapping(
        report_path,
        maximum_bytes=MAX_BAKEOFF_REPORT_BYTES,
        label=f"{task} selection report",
    )


def report_score(
    report: dict[str, Any],
    task: str,
    direction: str | None,
    required_slices: list[str],
    required_metrics: list[str] | None = None,
    asr_code_switch_wer_max: float = 0.21,
) -> dict[str, Any]:
    safety_pass, failures = critical_safety_pass(report, required_slices)
    metric_view = report
    if task == "mt" and direction:
        metric_view = report.get("directions", {}).get(direction) or {}
    metric_specs: list[tuple[str, bool, bool]]
    if task == "mt":
        default_metrics = ["sacrebleu", "chrf2"]
        primary_metric = "chrf2"
        metric_specs = [(name, True, True) for name in (required_metrics or default_metrics)]
    else:
        default_metrics = ["wer", "cer", "code_switch_wer"]
        primary_metric = "wer"
        metric_specs = [
            (name, False, name == "wer") for name in (required_metrics or default_metrics)
        ]
        code_switch = report.get("slices", {}).get("code_switch", {}).get("True")
        if code_switch is None:
            safety_pass = False
            failures.append("code_switch_wer:missing")
            code_switch_wer = None
        else:
            try:
                code_switch_wer = float(code_switch.get("wer", 1.0))
            except (TypeError, ValueError):
                code_switch_wer = math.nan
            if not math.isfinite(code_switch_wer):
                safety_pass = False
                failures.append("code_switch_wer:missing_or_invalid")
            elif code_switch_wer > asr_code_switch_wer_max:
                safety_pass = False
                failures.append("code_switch_wer:above_policy")
        metric_view = {**metric_view, "code_switch_wer": code_switch_wer}

    clinical_safety_pass = safety_pass
    metrics: dict[str, dict[str, Any]] = {}
    for metric_name, greater_is_better, requires_interval in metric_specs:
        metric_value = None
        try:
            candidate_metric = float(metric_view.get(metric_name))
            if math.isfinite(candidate_metric):
                metric_value = candidate_metric
        except (TypeError, ValueError):
            pass
        interval_value = normalized_interval(
            metric_view.get(f"{metric_name}_bootstrap_95ci")
        )
        if metric_value is None:
            failures.append(f"{metric_name}:missing_or_invalid")
        if requires_interval and interval_value is None:
            failures.append(f"{metric_name}_bootstrap_95ci:missing_or_invalid")
        metrics[metric_name] = {
            "value": metric_value,
            "confidence_interval_95": list(interval_value) if interval_value else None,
            "greater_is_better": greater_is_better,
            "evidence_valid": metric_value is not None
            and (not requires_interval or interval_value is not None),
        }
    evidence_valid = bool(metrics) and all(item["evidence_valid"] for item in metrics.values())
    safety_pass = clinical_safety_pass and evidence_valid
    primary = metrics.get(primary_metric) or {
        "value": None,
        "confidence_interval_95": None,
        "greater_is_better": task == "mt",
    }
    return {
        "metric_name": primary_metric,
        "metric": primary["value"],
        "confidence_interval_95": primary["confidence_interval_95"],
        "greater_is_better": primary["greater_is_better"],
        "metrics": metrics,
        "ranking_key": [
            item["value"] if item["greater_is_better"] else -item["value"]
            for item in metrics.values()
        ]
        if evidence_valid
        else None,
        "safety_pass": safety_pass,
        "clinical_safety_pass": clinical_safety_pass,
        "evidence_valid": evidence_valid,
        "safety_failures": failures,
        "report": report,
    }


def rank_scores(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [entry for entry in entries if entry["score"]["safety_pass"]]
    if not eligible:
        return []
    return sorted(
        eligible,
        key=lambda entry: entry["score"]["ranking_key"],
        reverse=True,
    )


def halving_groups(
    entries: list[dict[str, Any]], task: str
) -> list[list[dict[str, Any]]]:
    if task != "mt":
        return [entries]
    directions = sorted(
        {
            str(entry.get("candidate", {}).get("direction") or "")
            for entry in entries
        }
    )
    return [
        [
            entry
            for entry in entries
            if str(entry.get("candidate", {}).get("direction") or "") == direction
        ]
        for direction in directions
    ]


def select_semifinalists(
    pilot: list[dict[str, Any]], task: str, keep: int
) -> list[dict[str, Any]]:
    return [
        entry
        for group in halving_groups(pilot, task)
        for entry in rank_scores(group)[:keep]
    ]


def select_finalists(
    semifinal: list[dict[str, Any]], task: str
) -> list[dict[str, Any]]:
    finalists = []
    for group in halving_groups(semifinal, task):
        ranked = rank_scores(group)
        if not ranked:
            continue
        best = ranked[0]
        finalists.extend(
            entry
            for entry in ranked
            if entry is best or not multi_metric_stronger(best["score"], entry["score"])
        )
    return finalists


def strongest_eligible_challenger(
    ranked: list[dict[str, Any]], baseline_score: dict[str, Any]
) -> dict[str, Any] | None:
    """Return the best ranked challenger that can actually replace Candidate A."""
    if not ranked:
        return None
    if not baseline_score.get("clinical_safety_pass"):
        return ranked[0]
    return next(
        (
            entry
            for entry in ranked
            if multi_metric_stronger(entry["score"], baseline_score)
        ),
        None,
    )


def run_task_bakeoff(
    *,
    task: str,
    config: dict[str, Any],
    research_approvals: set[str],
    frozen: dict[str, Any],
    python: str,
    state_dir: Path,
    state_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    required = config["promotion_gate"]["critical_slices"] + config["promotion_gate"]["policy_slices"]
    required_metrics = config["promotion_gate"].get(
        f"{task}_metrics",
        ["sacrebleu", "chrf2"] if task == "mt" else ["wer", "cer", "code_switch_wer"],
    )
    asr_code_switch_wer_max = float(
        config["promotion_gate"].get("asr_code_switch_wer_max", 0.21)
    )
    candidate_a = next(
        item for item in config["candidates"][task] if item["role"] == "candidate_a"
    )
    candidate_a = deepcopy(candidate_a)
    candidate_a["unit_id"] = candidate_a["id"]
    candidate_a_adapter = Path(frozen["adapters"][task]["root"])
    candidate_a_directions = ["en_to_vi", "vi_to_en"] if task == "mt" else [None]
    candidate_a_reports = {}
    for direction in candidate_a_directions:
        key = direction or "vi"
        candidate_a_reports[key] = benchmark_unit(
            unit=candidate_a,
            task=task,
            label=f"candidate_a_frozen_{key}",
            adapter=candidate_a_adapter,
            config=config,
            python=python,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
            direction=direction,
        )

    units = []
    excluded = []
    for unit in candidate_units(config, task):
        allowed, reason = license_gate(unit, research_approvals)
        if allowed:
            units.append(unit)
        else:
            excluded.append({"unit": unit["unit_id"], "reason": reason})

    zero_shot = []
    pilot = []
    pilot_profiles = []
    profiles = hyperparameter_profiles(config, task) if units else []
    for unit in units:
        zero_report = benchmark_unit(
            unit=unit,
            task=task,
            label="zero_shot",
            adapter=None,
            config=config,
            python=python,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
            direction=unit.get("direction"),
        )
        zero_shot.append(
            {
                "unit": unit["unit_id"],
                "score": report_score(
                    zero_report,
                    task,
                    unit.get("direction"),
                    required,
                    required_metrics,
                    asr_code_switch_wer_max,
                ),
            }
        )
        profile_entries = []
        for profile in profiles:
            profile_id = str(profile["id"])
            adapter = train_unit(
                unit=unit,
                task=task,
                round_name="pilot",
                steps=int(config["successive_halving"]["pilot"]["train_steps"]),
                config=config,
                python=python,
                state_dir=state_dir,
                state_path=state_path,
                state=state,
                profile=profile,
            )
            report = benchmark_unit(
                unit=unit,
                task=task,
                label=f"pilot_{profile_id}",
                adapter=adapter,
                config=config,
                python=python,
                state_dir=state_dir,
                state_path=state_path,
                state=state,
                direction=unit.get("direction"),
            )
            entry = {
                "unit": unit["unit_id"],
                "candidate": unit,
                "adapter": str(adapter),
                "profile": profile,
                "score": report_score(
                    report,
                    task,
                    unit.get("direction"),
                    required,
                    required_metrics,
                    asr_code_switch_wer_max,
                ),
            }
            profile_entries.append(entry)
            pilot_profiles.append(entry)
        ranked_profiles = rank_scores(profile_entries)
        if ranked_profiles:
            pilot.append(ranked_profiles[0])

    semifinalists = select_semifinalists(
        pilot,
        task,
        int(config["successive_halving"]["semifinal"]["keep"]),
    )
    semifinal = []
    for entry in semifinalists:
        unit = entry["candidate"]
        profile = entry["profile"]
        adapter = train_unit(
            unit=unit,
            task=task,
            round_name="semifinal",
            steps=int(config["successive_halving"]["semifinal"]["train_steps"]),
            config=config,
            python=python,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
            profile=profile,
        )
        report = benchmark_unit(
            unit=unit,
            task=task,
            label=f"semifinal_{profile['id']}",
            adapter=adapter,
            config=config,
            python=python,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
            direction=unit.get("direction"),
        )
        semifinal.append(
            {
                "unit": unit["unit_id"],
                "candidate": unit,
                "adapter": str(adapter),
                "profile": profile,
                "score": report_score(
                    report,
                    task,
                    unit.get("direction"),
                    required,
                    required_metrics,
                    asr_code_switch_wer_max,
                ),
            }
        )

    finalists = select_finalists(semifinal, task)

    full = []
    for entry in finalists:
        unit = entry["candidate"]
        profile = entry["profile"]
        adapter = train_unit(
            unit=unit,
            task=task,
            round_name="full",
            steps=int(config["successive_halving"]["full"]["train_steps"]),
            config=config,
            python=python,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
            profile=profile,
        )
        report = benchmark_unit(
            unit=unit,
            task=task,
            label=f"full_{profile['id']}",
            adapter=adapter,
            config=config,
            python=python,
            state_dir=state_dir,
            state_path=state_path,
            state=state,
            direction=unit.get("direction"),
        )
        full.append(
            {
                "unit": unit["unit_id"],
                "candidate": unit,
                "adapter": str(adapter),
                "profile": profile,
                "score": report_score(
                    report,
                    task,
                    unit.get("direction"),
                    required,
                    required_metrics,
                    asr_code_switch_wer_max,
                ),
            }
        )

    directions = ["en_to_vi", "vi_to_en"] if task == "mt" else [None]
    winners = {}
    for direction in directions:
        baseline_score = report_score(
            candidate_a_reports[direction or "vi"],
            task,
            direction,
            required,
            required_metrics,
            asr_code_switch_wer_max,
        )
        challengers = [
            entry
            for entry in full
            if task != "mt" or entry["candidate"].get("direction") == direction
        ]
        ranked = rank_scores(challengers)
        if not baseline_score["evidence_valid"]:
            raise ValueError(
                f"Candidate A evidence is invalid for {task}/{direction or 'vi'}"
            )
        if not baseline_score["clinical_safety_pass"] and not ranked:
            raise RuntimeError(
                f"No safety-eligible winner for {task}/{direction or 'vi'}"
            )
        winner = {
            "candidate_id": candidate_a["id"],
            "adapter": str(candidate_a_adapter),
            "adapter_manifest_sha256": tree_manifest(candidate_a_adapter)[
                "manifest_sha256"
            ],
            "direction": direction,
            "score": baseline_score,
            "decision": "candidate_a_retained",
            "profile": None,
        }
        challenger = strongest_eligible_challenger(ranked, baseline_score)
        if challenger is not None:
            challenger_score = challenger["score"]
            winner = {
                "candidate_id": challenger["candidate"]["id"],
                "adapter": challenger["adapter"],
                "adapter_manifest_sha256": tree_manifest(Path(challenger["adapter"]))[
                    "manifest_sha256"
                ],
                "direction": direction,
                "profile": challenger["profile"],
                "score": challenger_score,
                "decision": (
                    "challenger_selected_after_candidate_a_safety_failure"
                    if not baseline_score["clinical_safety_pass"]
                    else "challenger_stronger_beyond_95ci"
                ),
            }
        winners[direction or "vi"] = winner
    return {
        "task": task,
        "excluded_by_license": excluded,
        "zero_shot": zero_shot,
        "pilot_profiles": pilot_profiles,
        "pilot": pilot,
        "semifinal": semifinal,
        "full": full,
        "winners": winners,
    }


def preflight(config: dict[str, Any], research_approvals: set[str]) -> dict[str, Any]:
    validate_resources(config)
    profiles = {
        task: hyperparameter_profiles(config, task) for task in ("mt", "asr")
    }
    validate_candidate_matrix(config)
    hashes = validate_selection_artifacts(config)
    licenses = license_decisions(config, research_approvals)
    return {
        "status": "pass",
        "selection_sha256": hashes,
        "licenses": licenses,
        "workflow": config["workflow"],
        "resource_policy": config["resources"],
        "hyperparameter_profiles": profiles,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/model_bakeoff.yaml")
    parser.add_argument("--current-program-state", type=Path)
    parser.add_argument("--mt-candidate-output", type=Path)
    parser.add_argument("--asr-candidate-output", type=Path)
    parser.add_argument("--state-dir", type=Path, default=ROOT / "gpu-runs/model-bakeoff")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--wait-current", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--resume-audit",
        action="store_true",
        help="Read-only validation of completed stage commands and output evidence.",
    )
    parser.add_argument("--scope", choices=["research", "production"], default="research")
    parser.add_argument(
        "--approve-research-license",
        action="append",
        default=[],
        help="Explicit candidate ID approval for a research-only license; never enables production.",
    )
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    config = load_config(args.config)
    report = preflight(config, set(args.approve_research_license))
    if args.resume_audit:
        if args.execute:
            parser.error("--resume-audit cannot be combined with --execute")
        state_path = args.state_dir.resolve() / "bakeoff_state.json"
        if not state_path.is_file():
            parser.error(f"resume state does not exist: {state_path}")
        try:
            resume_state = read_json_mapping(
                state_path,
                maximum_bytes=MAX_BAKEOFF_STATE_BYTES,
                label="Bake-off resume state",
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))
        resume_audit = audit_completed_stage_resume(resume_state, args.python)
        print(
            json.dumps(
                {**report, "resume_audit": resume_audit},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if resume_audit["status"] == "pass" else 2
    if args.preflight or not args.execute:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    program_path = find_current_program_state(config, args.current_program_state)
    current = wait_for_current_program(program_path, args.poll_seconds, args.wait_current)
    approvals = set(args.approve_research_license)
    binding = invocation_binding(program_path, args.config, args.scope, approvals)
    state_dir = args.state_dir.resolve()
    state_path = state_dir / "bakeoff_state.json"
    if state_path.is_file():
        state = read_json_mapping(
            state_path,
            maximum_bytes=MAX_BAKEOFF_STATE_BYTES,
            label="Bake-off state",
        )
    else:
        state = {
            "created_at": utc_now(),
            "execution_status": "running",
            "promotion_allowed": False,
            "current_program_state": str(program_path),
            "stages": {},
        }
        atomic_json(state_path, state)
    bind_or_validate_invocation(state, binding)
    candidate_a_locked_evaluations = {
        "mt": validate_candidate_a_locked_evaluation(
            current,
            config,
            "mt",
            candidate_a_resume_output(state, "mt", args.mt_candidate_output),
        ),
        "asr": validate_candidate_a_locked_evaluation(
            current,
            config,
            "asr",
            candidate_a_resume_output(state, "asr", args.asr_candidate_output),
        ),
    }
    freeze_path = state_dir / "candidate_a_freeze.json"
    if not freeze_path.is_file():
        frozen = freeze_candidate_a(current, freeze_path)
    else:
        frozen = validate_candidate_a_freeze(current, freeze_path)
    state["stage"] = "freeze_candidate_A"
    state["candidate_a_freeze"] = str(freeze_path)
    state["candidate_a_locked_evaluations"] = candidate_a_locked_evaluations
    state["scope"] = args.scope
    atomic_json(state_path, state)

    state["stage"] = "mt_multi_model_bakeoff"
    atomic_json(state_path, state)
    mt_result = run_task_bakeoff(
        task="mt",
        config=config,
        research_approvals=approvals,
        frozen=frozen,
        python=args.python,
        state_dir=state_dir,
        state_path=state_path,
        state=state,
    )
    state["stage"] = "asr_multi_model_bakeoff"
    atomic_json(state_path, state)
    asr_result = run_task_bakeoff(
        task="asr",
        config=config,
        research_approvals=approvals,
        frozen=frozen,
        python=args.python,
        state_dir=state_dir,
        state_path=state_path,
        state=state,
    )
    comparison = {
        "version": 1,
        "updated_at": utc_now(),
        "status": "selection_complete",
        "scope": args.scope,
        "promotion_allowed": False,
        "candidate_a_freeze": str(freeze_path),
        "candidate_a_locked_evaluations": candidate_a_locked_evaluations,
        "selection_sha256": report["selection_sha256"],
        "selection_policy": selection_policy_record(config),
        "research_license_approvals": sorted(approvals),
        "license_decisions": report["licenses"],
        "results": {"mt": mt_result, "asr": asr_result},
        "blind_test_v2": "pending",
    }
    comparison_path = ROOT / "data/reports/model_bakeoff/comparison.json"
    selection_snapshot_path = (
        ROOT
        / "data/reports/model_bakeoff"
        / f"selection_comparison-{selection_identity_sha256(comparison)[:16]}.json"
    )
    if selection_snapshot_path.is_symlink():
        raise ValueError("Immutable selection snapshot cannot be a symlink")
    if selection_snapshot_path.exists() and not selection_snapshot_path.is_file():
        raise ValueError("Immutable selection snapshot must be a regular file")
    if selection_snapshot_path.is_file():
        selection_snapshot = read_json_mapping(
            selection_snapshot_path,
            maximum_bytes=MAX_BAKEOFF_REPORT_BYTES,
            label="Immutable selection snapshot",
        )
        if selection_snapshot.get("status") != "selection_complete":
            raise ValueError("Immutable selection snapshot has an invalid status")
        if selection_identity(selection_snapshot) != selection_identity(comparison):
            raise ValueError("Immutable selection snapshot does not match resumed winners")
    else:
        selection_snapshot = deepcopy(comparison)
        atomic_json(
            selection_snapshot_path,
            selection_snapshot,
            maximum_bytes=MAX_BAKEOFF_REPORT_BYTES,
        )
    comparison["selection_snapshot"] = {
        "path": str(selection_snapshot_path.resolve()),
        "sha256": sha256(selection_snapshot_path),
    }
    atomic_json(
        comparison_path,
        comparison,
        maximum_bytes=MAX_BAKEOFF_REPORT_BYTES,
    )

    blind_lock_path = ROOT / config["data"]["blind_test_v2_lock"]
    blind_lock = read_json_mapping(
        blind_lock_path,
        maximum_bytes=MAX_BAKEOFF_STATE_BYTES,
        label="Blind v2 lock",
    )
    if blind_lock.get("status") == "awaiting_unseen_data":
        state["stage"] = "blind_locked_test_v2"
        state["execution_status"] = "waiting_for_blind_test_v2"
        state["next_stage"] = "blind_locked_test_v2"
        state["selection_comparison"] = str(selection_snapshot_path)
        atomic_json(state_path, state)
        print(
            json.dumps(
                {
                    **report,
                    "state": str(state_path),
                    "candidate_a_freeze": str(freeze_path),
                    "selection_comparison": str(selection_snapshot_path),
                    "next_stage": state["next_stage"],
                    "note": "Selection is complete; promotion is blocked until unseen blind v2 data is checksum-locked.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return CRITICAL_EXIT_WAITING_FOR_BLIND

    if blind_lock.get("status") not in {"locked_unopened", "partially_opened", "opened"}:
        raise ValueError(f"Unexpected blind v2 state: {blind_lock.get('status')!r}")
    state["stage"] = "blind_locked_test_v2"
    blind_results = []
    blind_output = state_dir / "blind-v2"
    winner_specs = [
        ("mt", direction, winner)
        for direction, winner in mt_result["winners"].items()
    ] + [("asr", None, asr_result["winners"]["vi"])]
    for task, direction, winner in winner_specs:
        key = direction or "vi"
        stem = f"blind_v2_{task}_{key}_{winner['candidate_id']}"
        gate_path = blind_output / f"{stem}_gate.json"
        command = [
            args.python,
            str(ROOT / "scripts/run_blind_candidate_suite.py"),
            "--action",
            "evaluate",
            "--task",
            task,
            "--candidate",
            winner["candidate_id"],
            "--adapter",
            winner["adapter"],
            "--scope",
            args.scope,
            "--selection-comparison",
            str(selection_snapshot_path),
            "--output-dir",
            str(blind_output),
        ]
        if direction:
            command += ["--direction", direction]
        stage = f"blind_v2_{task}_{key}"
        run_stage(
            stage,
            command,
            state_path,
            state,
            state_dir / "logs" / f"{stage}.log",
            [gate_path],
        )
        blind_results.append(
            read_json_mapping(
                gate_path,
                maximum_bytes=MAX_BAKEOFF_REPORT_BYTES,
                label=f"Blind v2 {task} gate",
            )
        )
    comparison["blind_test_v2"] = blind_results
    comparison["status"] = "blind_complete"
    atomic_json(
        comparison_path,
        comparison,
        maximum_bytes=MAX_BAKEOFF_REPORT_BYTES,
    )
    blind_pass = all(item.get("promotion_allowed") is True for item in blind_results)
    if not blind_pass:
        comparison.update(
            {
                "status": "complete",
                "deployment": None,
                "deployment_gate_failures": ["not_run_blind_failed"],
                "promotion_allowed": False,
                "production_license_gate": False,
            }
        )
        atomic_json(
            comparison_path,
            comparison,
            maximum_bytes=MAX_BAKEOFF_REPORT_BYTES,
        )
        backup = complete_terminal_bakeoff(
            state_path,
            state,
            comparison_path,
            decision="reject",
            promotion_allowed=False,
            scope=args.scope,
        )
        print(
            json.dumps(
                {
                    **report,
                    "state": str(state_path),
                    "candidate_a_freeze": str(freeze_path),
                    "comparison": str(comparison_path),
                    "decision": state["decision"],
                    "verified_backup": backup,
                    "note": "Deployment was not run because the blind hard gate failed.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    expected_deployment = deployment_expectations(winner_specs)
    deployment_dir = ROOT / "data/reports/model_bakeoff"
    deployment_path = deployment_dir / "deployment_selected_winners.json"
    if not deployment_path.is_file():
        deployment_draft_path = deployment_dir / "deployment_selected_winners.draft.json"
        if not deployment_draft_path.is_file():
            draft_command = [
                args.python,
                str(ROOT / "scripts/prepare_deployment_benchmark.py"),
                "--action",
                "template",
                "--selection-comparison",
                str(comparison_path),
                "--config",
                str(args.config),
                "--draft",
                str(deployment_draft_path),
            ]
            run_stage(
                "prepare_deployment_draft",
                draft_command,
                state_path,
                state,
                state_dir / "logs/prepare_deployment_draft.log",
                [deployment_draft_path],
            )
        deployment_draft = read_json_mapping(
            deployment_draft_path,
            maximum_bytes=MAX_BAKEOFF_STATE_BYTES,
            label="Deployment draft",
        )
        draft_pass, draft_failures = validate_deployment_draft(
            deployment_draft,
            expected_deployment,
            comparison_path,
        )
        if not draft_pass:
            raise ValueError(
                "Deployment draft does not match selected winners: "
                + ", ".join(draft_failures)
            )
        state["stage"] = "deployment_benchmark"
        state["execution_status"] = "waiting_for_qcs6490_benchmark"
        state["next_stage"] = "deployment_benchmark"
        state["deployment_draft_template"] = {
            "path": str(deployment_draft_path),
            "created_sha256": sha256(deployment_draft_path),
        }
        atomic_json(state_path, state)
        return CRITICAL_EXIT_WAITING_FOR_BLIND
    deployment = read_json_mapping(
        deployment_path,
        maximum_bytes=MAX_BAKEOFF_STATE_BYTES,
        label="Deployment report",
    )
    deployment_pass, deployment_failures = validate_deployment_report(
        deployment,
        expected_deployment,
        list(config["promotion_gate"]["deployment_metrics"]),
        int(config["promotion_gate"]["deployment_min_runs"]),
    )
    winner_ids = {
        winner["candidate_id"]
        for _, _, winner in winner_specs
    }
    production_licenses = all(
        next(
            candidate
            for task in ("mt", "asr")
            for candidate in config["candidates"][task]
            if candidate["id"] == winner_id
        )["license"].get("production_eligible")
        for winner_id in winner_ids
    )
    promotion_allowed = deployment_pass and (
        args.scope == "research" or production_licenses
    )
    comparison.update(
        {
            "status": "complete",
            "deployment": deployment,
            "deployment_gate_failures": deployment_failures,
            "promotion_allowed": promotion_allowed,
            "production_license_gate": production_licenses,
        }
    )
    atomic_json(
        comparison_path,
        comparison,
        maximum_bytes=MAX_BAKEOFF_REPORT_BYTES,
    )
    decision = "promote" if promotion_allowed else "reject"
    backup = complete_terminal_bakeoff(
        state_path,
        state,
        comparison_path,
        decision=decision,
        promotion_allowed=promotion_allowed,
        scope=args.scope,
    )
    print(
        json.dumps(
            {
                **report,
                "state": str(state_path),
                "candidate_a_freeze": str(freeze_path),
                "comparison": str(comparison_path),
                "decision": state["decision"],
                "verified_backup": backup,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if promotion_allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())

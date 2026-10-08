"""Mine a review-only terminology challenge set from real validation errors.

The builder accepts prediction JSONL files directly or prediction members from
verified ``tar.gz`` artifacts.  Every prediction is matched back to a validation
manifest before it can become a challenge.  MT rows require a detected clinical
safety failure; ASR rows require a non-zero word error.  The resulting rows are
always pending human review and can never be consumed as automatic corrections.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from jiwer import process_words


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.safety_guard import (  # noqa: E402
    safety_issue_codes,
    validate_translation,
)
from src.utils.text_normalization import normalize_for_wer  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_SIZE = 128
DEFAULT_SOURCE_LIST = ROOT / "configs/terminology_challenge_sources.txt"
MIN_CHALLENGE_ROWS = 100
MAX_CHALLENGE_ROWS = 200
MAX_SOURCE_BYTES = 64 * 1024 * 1024
LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
CRITICAL_CATEGORIES = {
    "drug_name",
    "dose",
    "number",
    "unit",
    "negation",
    "terminology",
    "code_switch",
}
CRITICAL_ISSUES = {
    "empty_translation",
    "identifier_mismatch",
    "negation_binding_mismatch",
    "negation_mismatch",
    "number_mismatch",
    "quantity_binding_ambiguous",
    "quantity_binding_mismatch",
    "quantity_mismatch",
    "terminology_missing",
    "unit_mismatch",
}


@dataclass(frozen=True)
class SourceSpec:
    label: str
    path: Path
    member: str | None = None


@dataclass(frozen=True)
class LoadedSource:
    spec: SourceSpec
    rows: list[dict[str, Any]]
    evidence: dict[str, Any]
    provenance: dict[str, Any] | None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_source_spec(raw: str) -> SourceSpec:
    label, separator, location = raw.partition("=")
    if not separator or not LABEL_RE.fullmatch(label):
        raise ValueError(
            "Prediction source must be LABEL=PATH or LABEL=ARCHIVE::MEMBER "
            "with a lowercase filesystem-safe label"
        )
    path_text, member_separator, member = location.partition("::")
    if not path_text or (member_separator and not member):
        raise ValueError(f"Invalid prediction source location: {raw}")
    if member_separator:
        _safe_member_name(member)
    return SourceSpec(
        label=label,
        path=Path(path_text),
        member=member if member_separator else None,
    )


def load_source_list(path: Path) -> tuple[list[SourceSpec], dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Prediction source list does not exist: {resolved}")
    data = resolved.read_bytes()
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Prediction source list is not UTF-8: {resolved}") from exc
    specs = [
        parse_source_spec(line.strip())
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not specs:
        raise ValueError(f"Prediction source list is empty: {resolved}")
    return specs, {
        "path": display_path(resolved),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "sources": len(specs),
    }


def _safe_member_name(member: str) -> PurePosixPath:
    if "\\" in member:
        raise ValueError(f"Archive member must use POSIX separators: {member}")
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe archive member path: {member}")
    return path


def _read_regular_member(archive: tarfile.TarFile, member: str) -> bytes:
    _safe_member_name(member)
    try:
        info = archive.getmember(member)
    except KeyError as exc:
        raise ValueError(f"Archive member is missing: {member}") from exc
    if not info.isfile():
        raise ValueError(f"Archive member is not a regular file: {member}")
    if info.size > MAX_SOURCE_BYTES:
        raise ValueError(f"Archive member exceeds {MAX_SOURCE_BYTES} bytes: {member}")
    handle = archive.extractfile(info)
    if handle is None:
        raise ValueError(f"Could not read archive member: {member}")
    data = handle.read(MAX_SOURCE_BYTES + 1)
    if len(data) != info.size or len(data) > MAX_SOURCE_BYTES:
        raise ValueError(f"Archive member size mismatch: {member}")
    return data


def _decode_jsonl(data: bytes, identity: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Prediction source is not UTF-8: {identity}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {identity}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Prediction row is not an object at {identity}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"Prediction source is empty: {identity}")
    return rows


def _verify_archive_provenance(
    provenance: dict[str, Any], data: bytes, rows: list[dict[str, Any]], identity: str
) -> None:
    if provenance.get("schema_version") != 1:
        raise ValueError(f"Unsupported prediction provenance schema: {identity}")
    prediction = provenance.get("predictions")
    if not isinstance(prediction, dict):
        raise ValueError(f"Prediction provenance is missing predictions: {identity}")
    expected = {
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "rows": len(rows),
    }
    for key, value in expected.items():
        if prediction.get(key) != value:
            raise ValueError(f"Prediction provenance {key} mismatch: {identity}")


def load_source(spec: SourceSpec) -> LoadedSource:
    path = spec.path.resolve()
    if not path.is_file():
        raise ValueError(f"Prediction source does not exist: {path}")
    if spec.member is None:
        artifact_bytes = path.read_bytes()
        if len(artifact_bytes) > MAX_SOURCE_BYTES:
            raise ValueError(f"Prediction source exceeds {MAX_SOURCE_BYTES} bytes: {path}")
        rows = _decode_jsonl(artifact_bytes, display_path(path))
        return LoadedSource(
            spec=spec,
            rows=rows,
            evidence={
                "label": spec.label,
                "path": display_path(path),
                "bytes": len(artifact_bytes),
                "sha256": sha256_bytes(artifact_bytes),
                "rows": len(rows),
            },
            provenance=None,
        )

    member = str(_safe_member_name(spec.member))
    with tarfile.open(path, mode="r:gz") as archive:
        data = _read_regular_member(archive, member)
        provenance_member = f"{member}.provenance.json"
        provenance_data = _read_regular_member(archive, provenance_member)
    rows = _decode_jsonl(data, f"{display_path(path)}::{member}")
    try:
        provenance = json.loads(provenance_data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid prediction provenance: {path}::{provenance_member}") from exc
    if not isinstance(provenance, dict):
        raise ValueError(f"Prediction provenance is not an object: {path}")
    _verify_archive_provenance(provenance, data, rows, f"{display_path(path)}::{member}")
    return LoadedSource(
        spec=spec,
        rows=rows,
        evidence={
            "label": spec.label,
            "path": display_path(path),
            "archive_bytes": path.stat().st_size,
            "archive_sha256": sha256_file(path),
            "member": member,
            "member_bytes": len(data),
            "member_sha256": sha256_bytes(data),
            "rows": len(rows),
            "provenance_member": provenance_member,
            "provenance_sha256": sha256_bytes(provenance_data),
        },
        provenance=provenance,
    )


def load_validation_manifest(
    path: Path, task: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Validation manifest does not exist: {resolved}")
    data = resolved.read_bytes()
    rows = _decode_jsonl(data, display_path(resolved))
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id") or "").strip()
        if not row_id:
            raise ValueError(f"{task.upper()} validation row is missing id")
        if row_id in indexed:
            raise ValueError(f"Duplicate {task.upper()} validation id: {row_id}")
        role = str(row.get("role") or "").strip().lower()
        if role != "validation":
            raise ValueError(f"{task.upper()} manifest contains non-validation row: {row_id}")
        indexed[row_id] = row
    return indexed, {
        "task": task,
        "path": display_path(resolved),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "rows": len(rows),
    }


def infer_task(rows: Iterable[dict[str, Any]], label: str) -> str:
    tasks = {"mt" if row.get("direction") is not None else "asr" for row in rows}
    if len(tasks) != 1:
        raise ValueError(f"Prediction source mixes tasks: {label}")
    return tasks.pop()


def _expected_prediction(
    task: str, prediction: dict[str, Any], manifest_row: dict[str, Any]
) -> tuple[str | None, str | None, str]:
    if task == "asr":
        return None, None, str(manifest_row.get("text") or "")
    direction = str(prediction.get("direction") or "")
    if direction == "en_to_vi":
        return direction, str(manifest_row.get("source_text") or ""), str(
            manifest_row.get("target_text") or ""
        )
    if direction == "vi_to_en":
        return direction, str(manifest_row.get("target_text") or ""), str(
            manifest_row.get("source_text") or ""
        )
    raise ValueError(f"Unsupported MT direction: {direction or '<missing>'}")


def _manifest_terminology(row: dict[str, Any], direction: str | None) -> dict[str, Any]:
    terminology = row.get("terminology") or {}
    if direction and isinstance(terminology, dict) and direction in terminology:
        terminology = terminology[direction]
    return terminology if isinstance(terminology, dict) else {}


def _asr_terms(reference: str, hypothesis: str) -> list[str]:
    reference_words = normalize_for_wer(reference).split()
    hypothesis_words = normalize_for_wer(hypothesis).split()
    matcher = SequenceMatcher(a=reference_words, b=hypothesis_words, autojunk=False)
    terms = []
    for operation, start, end, _, _ in matcher.get_opcodes():
        if operation != "equal" and start < end:
            terms.append(" ".join(reference_words[start:end][:8]))
    return list(dict.fromkeys(term for term in terms if term))[:8]


def _asr_word_error_rate(reference: str, hypothesis: str) -> float:
    result = process_words(normalize_for_wer(reference), normalize_for_wer(hypothesis))
    return float(result.wer)


def _candidate_key(task: str, row_id: str, direction: str | None) -> str:
    return f"{task}\x1f{direction or 'transcription'}\x1f{row_id}"


def _observation(
    task: str,
    label: str,
    prediction: dict[str, Any],
    source: str | None,
    reference: str,
    direction: str | None,
    terminology: dict[str, Any],
) -> dict[str, Any]:
    hypothesis = prediction.get("hypothesis")
    if not isinstance(hypothesis, str):
        raise ValueError(f"Prediction hypothesis must be a string: {label}")
    if task == "asr":
        word_error_rate = _asr_word_error_rate(reference, hypothesis)
        return {
            "model_run": label,
            "hypothesis": hypothesis,
            "error_detected": word_error_rate > 0.0,
            "word_error_rate": word_error_rate,
            "candidate_terms": _asr_terms(reference, hypothesis),
            "safety_issues": [],
        }
    assert source is not None and direction is not None
    source_language, target_language = direction.split("_to_")
    check = validate_translation(
        source,
        hypothesis,
        source_language,
        target_language,
        terminology=terminology,
    )
    return {
        "model_run": label,
        "hypothesis": hypothesis,
        "error_detected": not check.safe,
        "word_error_rate": None,
        "candidate_terms": sorted(str(term) for term in terminology),
        "safety_issues": list(check.issues),
    }


def mine_candidates(
    sources: list[LoadedSource],
    manifests: dict[str, dict[str, dict[str, Any]]],
    manifest_evidence: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    source_labels: set[str] = set()
    for loaded in sorted(sources, key=lambda item: item.spec.label):
        label = loaded.spec.label
        if label in source_labels:
            raise ValueError(f"Duplicate prediction source label: {label}")
        source_labels.add(label)
        task = infer_task(loaded.rows, label)
        manifest_sha256 = manifest_evidence[task]["sha256"]
        if loaded.provenance is not None:
            specification = loaded.provenance.get("specification") or {}
            provenance_manifest = specification.get("manifest") or {}
            if specification.get("task") != task:
                raise ValueError(f"Prediction provenance task mismatch: {label}")
            if provenance_manifest.get("sha256") != manifest_sha256:
                raise ValueError(f"Prediction provenance manifest mismatch: {label}")

        seen_keys: set[str] = set()
        for prediction in loaded.rows:
            row_id = str(prediction.get("id") or "").strip()
            if not row_id or row_id not in manifests[task]:
                raise ValueError(
                    f"Prediction id is absent from {task} validation: {label}:{row_id}"
                )
            manifest_row = manifests[task][row_id]
            direction, source, reference = _expected_prediction(task, prediction, manifest_row)
            if not reference or prediction.get("reference") != reference:
                raise ValueError(f"Prediction reference mismatch: {label}:{row_id}")
            if task == "mt" and prediction.get("source") != source:
                raise ValueError(f"Prediction source mismatch: {label}:{row_id}")
            key = _candidate_key(task, row_id, direction)
            if key in seen_keys:
                raise ValueError(f"Duplicate prediction key in source: {label}:{row_id}")
            seen_keys.add(key)
            terminology = _manifest_terminology(manifest_row, direction)
            observation = _observation(
                task,
                label,
                prediction,
                source,
                reference,
                direction,
                terminology,
            )
            candidate = candidates.setdefault(
                key,
                {
                    "schema_version": SCHEMA_VERSION,
                    "challenge_id": stable_digest(key)[:24],
                    "task": task,
                    "direction": direction,
                    "source_id": row_id,
                    "input_text": source,
                    "reference": reference,
                    "categories": sorted(
                        {str(value) for value in manifest_row.get("categories", [])}
                    ),
                    "candidate_terms": [],
                    "hazard_level": None,
                    "hazard_basis": [],
                    "review_status": "pending",
                    "reviewer": None,
                    "reviewed_at": None,
                    "evaluation_only": True,
                    "auto_correction_eligible": False,
                    "observations": [],
                    "provenance": {
                        "manifest_path": manifest_evidence[task]["path"],
                        "manifest_sha256": manifest_sha256,
                        "prediction_sources": [],
                    },
                },
            )
            candidate["observations"].append(observation)
            candidate["provenance"]["prediction_sources"].append(label)
            candidate["candidate_terms"].extend(observation["candidate_terms"])
            if task == "asr":
                for field in (
                    "audio_path",
                    "audio_sha256",
                    "speaker",
                    "group",
                    "accent",
                    "source",
                ):
                    candidate[field] = manifest_row.get(field)

    mined = []
    for candidate in candidates.values():
        observations = sorted(candidate["observations"], key=lambda item: item["model_run"])
        if not any(observation["error_detected"] for observation in observations):
            continue
        candidate["observations"] = observations
        candidate["provenance"]["prediction_sources"] = sorted(
            set(candidate["provenance"]["prediction_sources"])
        )
        candidate["candidate_terms"] = sorted(set(candidate["candidate_terms"]))
        issue_codes = sorted(
            {
                code
                for observation in observations
                for code in safety_issue_codes(observation["safety_issues"])
            }
        )
        categories = set(candidate["categories"])
        critical_issues = sorted(set(issue_codes) & CRITICAL_ISSUES)
        if critical_issues:
            candidate["hazard_level"] = "critical"
            candidate["hazard_basis"] = critical_issues
        elif issue_codes:
            candidate["hazard_level"] = "high"
            candidate["hazard_basis"] = issue_codes
        elif categories & CRITICAL_CATEGORIES:
            candidate["hazard_level"] = "high"
            candidate["hazard_basis"] = sorted(categories & CRITICAL_CATEGORIES)
        else:
            candidate["hazard_level"] = "moderate"
            candidate["hazard_basis"] = ["asr_word_error"]
        mined.append(candidate)
    return mined


def _candidate_score(candidate: dict[str, Any]) -> tuple[int, int, float, str]:
    issue_count = sum(
        len(observation["safety_issues"]) for observation in candidate["observations"]
    )
    error_runs = sum(
        bool(observation["error_detected"]) for observation in candidate["observations"]
    )
    max_wer = max(
        (
            float(observation["word_error_rate"] or 0.0)
            for observation in candidate["observations"]
        ),
        default=0.0,
    )
    return issue_count, error_runs, max_wer, candidate["challenge_id"]


def select_challenges(candidates: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    if not MIN_CHALLENGE_ROWS <= size <= MAX_CHALLENGE_ROWS:
        raise ValueError(
            f"Challenge size must be between {MIN_CHALLENGE_ROWS} and {MAX_CHALLENGE_ROWS}"
        )
    if len(candidates) < size:
        raise ValueError(
            f"Only {len(candidates)} verified validation errors are available for {size} rows"
        )
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        primary_basis = candidate["hazard_basis"][0]
        stratum = f"{candidate['task']}|{candidate['direction'] or 'transcription'}|{primary_basis}"
        strata[stratum].append(candidate)
    for rows in strata.values():
        rows.sort(key=_candidate_score, reverse=True)
    selected = []
    offset = 0
    names = sorted(strata)
    while len(selected) < size:
        added = False
        for name in names:
            if offset < len(strata[name]) and len(selected) < size:
                selected.append(strata[name][offset])
                added = True
        if not added:
            break
        offset += 1
    if len(selected) != size:
        raise ValueError(f"Could only select {len(selected)} of {size} challenge rows")
    return sorted(selected, key=lambda row: row["challenge_id"])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> bytes:
    data = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def build_report(
    output: Path,
    output_data: bytes,
    rows: list[dict[str, Any]],
    sources: list[LoadedSource],
    manifests: dict[str, dict[str, Any]],
    source_list: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_task: dict[str, int] = defaultdict(int)
    by_hazard: dict[str, int] = defaultdict(int)
    by_direction: dict[str, int] = defaultdict(int)
    for row in rows:
        by_task[row["task"]] += 1
        by_hazard[row["hazard_level"]] += 1
        by_direction[row["direction"] or "transcription"] += 1
        if row["review_status"] != "pending" or row["auto_correction_eligible"]:
            raise ValueError("New challenge rows must remain pending and evaluation-only")
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "awaiting_human_review",
        "policy": {
            "source_role": "validation_only",
            "evaluation_only": True,
            "auto_correction_eligible": False,
            "minimum_rows": MIN_CHALLENGE_ROWS,
            "maximum_rows": MAX_CHALLENGE_ROWS,
        },
        "artifact": {
            "path": display_path(output),
            "bytes": len(output_data),
            "sha256": sha256_bytes(output_data),
            "rows": len(rows),
        },
        "coverage": {
            "by_task": dict(sorted(by_task.items())),
            "by_direction": dict(sorted(by_direction.items())),
            "by_hazard_level": dict(sorted(by_hazard.items())),
        },
        "manifests": {task: manifests[task] for task in sorted(manifests)},
        "prediction_sources": [
            loaded.evidence for loaded in sorted(sources, key=lambda item: item.spec.label)
        ],
    }
    if source_list is not None:
        report["source_list"] = source_list
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-source",
        action="append",
        help="LABEL=PATH or LABEL=ARCHIVE::MEMBER; repeat for each model run",
    )
    parser.add_argument(
        "--source-list",
        type=Path,
        default=DEFAULT_SOURCE_LIST,
        help="UTF-8 source-spec list used when --prediction-source is omitted",
    )
    parser.add_argument(
        "--mt-manifest",
        type=Path,
        default=ROOT / "data/eval/mt_selection_dev.jsonl",
    )
    parser.add_argument(
        "--asr-manifest",
        type=Path,
        default=ROOT / "data/processed/manifests/asr--validation-local.jsonl",
    )
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/eval/terminology_challenge_set.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data/eval/terminology_challenge_set.report.json",
    )
    args = parser.parse_args()

    if args.prediction_source:
        specs = [parse_source_spec(raw) for raw in args.prediction_source]
        source_list_evidence = None
    else:
        specs, source_list_evidence = load_source_list(args.source_list)
    sources = [load_source(spec) for spec in specs]
    mt_manifest, mt_evidence = load_validation_manifest(args.mt_manifest, "mt")
    asr_manifest, asr_evidence = load_validation_manifest(args.asr_manifest, "asr")
    manifests = {"mt": mt_manifest, "asr": asr_manifest}
    manifest_evidence = {"mt": mt_evidence, "asr": asr_evidence}
    candidates = mine_candidates(sources, manifests, manifest_evidence)
    selected = select_challenges(candidates, args.size)
    output_data = write_jsonl(args.output, selected)
    report = build_report(
        args.output,
        output_data,
        selected,
        sources,
        manifest_evidence,
        source_list_evidence,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

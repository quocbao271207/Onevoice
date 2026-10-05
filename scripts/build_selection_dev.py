"""Build deterministic model-selection manifests from validation data only.

These files may be opened repeatedly during successive halving.  They are not
the blind test and never draw rows from either locked test manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.quality import fingerprint_text  # noqa: E402


SEED = 20261005
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
UNIT_RE = re.compile(
    r"(?:\b(?:mg|mcg|g|kg|ml|mmhg|bpm|gram|gam|microgram|nanogram|đơn vị|units?)\b|%)",
    re.I,
)
NEGATION_RE = re.compile(r"\b(?:không|chưa|chẳng|not|no|without|deny|denies)\b", re.I)
CODE_SWITCH_RE = re.compile(
    r"\b(?:covid|hiv|aids|mri|ct|x[- ]?ray|spo2|ecg|pcr|insulin|metformin|paracetamol)\b",
    re.I,
)
DRUG_RE = re.compile(
    r"\b(?:thuốc|drug|dose|liều|insulin|metformin|paracetamol|amoxicillin|aspirin|epinephrine)\b",
    re.I,
)
NEGATION_TOKENS = ("không", "chưa", "chẳng", "not", "no", "without", "deny", "denies")
MEDICAL_TERMS = (
    "insulin",
    "metformin",
    "paracetamol",
    "amoxicillin",
    "aspirin",
    "epinephrine",
    "adrenaline",
    "hiv",
    "covid",
    "viêm",
    "nhiễm",
    "ung thư",
    "diabetes",
    "hypertension",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def row_leakage_values(task: str, row: dict[str, Any]) -> dict[str, str]:
    row_id = str(row.get("id") or "").strip()
    if not row_id:
        raise ValueError("Selection source row is missing id")
    values = {"id": row_id}
    if task == "mt":
        source = str(row.get("source_text") or "").strip()
        target = str(row.get("target_text") or "").strip()
        if not source or not target:
            raise ValueError(f"Selection source MT row {row_id} is missing text")
        values["pair_fingerprint"] = fingerprint_text(source + "\x1f" + target)
        return values
    transcript = str(row.get("text") or "").strip()
    if not transcript:
        raise ValueError(f"Selection source ASR row {row_id} is missing text")
    values["text_fingerprint"] = fingerprint_text(transcript)
    for key in ("audio_sha256", "speaker", "group"):
        value = str(row.get(key) or "").strip()
        if value:
            values[key] = value
    return values


def collect_leakage_values(
    task: str, rows: list[dict[str, Any]]
) -> dict[str, set[str]]:
    keys = (
        ("id", "pair_fingerprint")
        if task == "mt"
        else ("id", "text_fingerprint", "audio_sha256", "speaker", "group")
    )
    values = {key: set() for key in keys}
    for row in rows:
        for key, value in row_leakage_values(task, row).items():
            values[key].add(value)
    return values


def filter_disjoint(
    task: str,
    rows: list[dict[str, Any]],
    excluded: dict[str, set[str]],
) -> list[dict[str, Any]]:
    kept = []
    for row in rows:
        values = row_leakage_values(task, row)
        if any(value in excluded[key] for key, value in values.items()):
            continue
        kept.append(row)
    return kept


def stable_key(row: dict[str, Any], seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}\x1f{row['id']}".encode()).hexdigest()


def round_robin_strata(
    rows: list[dict[str, Any]], size: int, stratum: Callable[[dict[str, Any]], str]
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    for values in groups.values():
        values.sort(key=stable_key)
    selected: list[dict[str, Any]] = []
    offset = 0
    names = sorted(groups)
    while len(selected) < min(size, len(rows)):
        added = False
        for name in names:
            if offset < len(groups[name]) and len(selected) < size:
                selected.append(groups[name][offset])
                added = True
        if not added:
            break
        offset += 1
    return selected


def lexical_categories(text: str) -> list[str]:
    categories = []
    if DRUG_RE.search(text):
        categories.append("drug_name")
    if NUMBER_RE.search(text):
        categories.append("number")
    if UNIT_RE.search(text):
        categories.extend(["dose", "unit"])
    if NEGATION_RE.search(text):
        categories.append("negation")
    if CODE_SWITCH_RE.search(text):
        categories.append("code_switch")
    if any(term in text.lower() for term in MEDICAL_TERMS):
        categories.append("terminology")
    return sorted(set(categories))


def phrases(pattern: re.Pattern[str], text: str) -> list[str]:
    return list(dict.fromkeys(match.group(0) for match in pattern.finditer(text)))


def asr_expectations(text: str) -> dict[str, list[list[str]]]:
    lower = text.lower()
    numbers = phrases(NUMBER_RE, text)
    units = phrases(UNIT_RE, text)
    code_switch = phrases(CODE_SWITCH_RE, text)
    drugs = [term for term in MEDICAL_TERMS[:7] if term in lower]
    terminology = [term for term in MEDICAL_TERMS if term in lower]
    negations = [term for term in NEGATION_TOKENS if re.search(rf"\b{re.escape(term)}\b", lower)]
    expectations: dict[str, list[list[str]]] = {}
    if drugs:
        expectations["drug_name"] = [[term] for term in drugs]
    if numbers:
        expectations["number"] = [[term] for term in numbers]
    if units:
        expectations["unit"] = [[term] for term in units]
    if numbers and units:
        expectations["dose"] = [[term] for term in numbers + units]
    if negations:
        expectations["negation"] = [[term] for term in negations]
    if terminology:
        expectations["terminology"] = [[term] for term in terminology]
    if code_switch:
        expectations["code_switch"] = [[term] for term in code_switch]
    return expectations


def mt_terminology(row: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    en = row["source_text"].lower()
    vi = row["target_text"].lower()
    shared = [term for term in MEDICAL_TERMS if term in en and term in vi]
    return {
        "en_to_vi": {term: [term] for term in shared},
        "vi_to_en": {term: [term] for term in shared},
    }


def build_mt(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    enriched = []
    for source in rows:
        row = dict(source)
        categories = lexical_categories(f"{row['source_text']} {row['target_text']}")
        terminology = mt_terminology(row)
        if not terminology["en_to_vi"]:
            categories = [name for name in categories if name not in {"drug_name", "terminology"}]
        row["categories"] = categories
        row["terminology"] = terminology
        row["selection_provenance"] = "mt--validation.jsonl"
        enriched.append(row)
    return round_robin_strata(
        enriched,
        size,
        lambda row: "+".join(row["categories"]),
    )


def accent_region(value: Any) -> str:
    text = str(value or "unknown").lower()
    if "north" in text:
        return "north"
    if "central" in text:
        return "central"
    if "south" in text:
        return "south"
    return "unknown"


def build_asr(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    enriched = []
    for source in rows:
        row = dict(source)
        metadata = row.get("metadata") or {}
        expectations = asr_expectations(row["text"])
        categories = sorted(expectations)
        row["categories"] = categories
        row["safety_expectations"] = expectations
        row["selection_dimensions"] = {
            "accent_region": accent_region(row.get("accent")),
            "role": str(metadata.get("role") or "unknown").lower(),
            "recording_condition": str(metadata.get("rec_condition") or "unknown").lower(),
            "noise": str(metadata.get("rec_condition") or "").lower()
            in {"tel", "consultation"},
            "code_switch": "code_switch" in categories,
        }
        row["selection_provenance"] = "asr--validation-local.jsonl"
        enriched.append(row)
    return round_robin_strata(
        enriched,
        size,
        lambda row: "|".join(str(value) for value in row["selection_dimensions"].values()),
    )


def ensure_no_overlap(
    task: str,
    selected: list[dict[str, Any]],
    excluded: dict[str, set[str]],
) -> None:
    selected_values = collect_leakage_values(task, selected)
    for key, values in selected_values.items():
        overlap = values & excluded[key]
        if overlap:
            raise ValueError(f"Selection leakage for {task}.{key}: {len(overlap)} rows")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mt-size", type=int, default=256)
    parser.add_argument("--asr-size", type=int, default=384)
    args = parser.parse_args()
    if min(args.mt_size, args.asr_size) < 1:
        parser.error("selection sizes must be positive")

    config = yaml.safe_load((ROOT / "configs/model_bakeoff.yaml").read_text(encoding="utf-8"))
    excluded_rows: dict[str, list[dict[str, Any]]] = {"mt": [], "asr": []}
    exclusion_paths = [
        *(ROOT / path for path in config["data"]["train"].values()),
        *(ROOT / path for path in config["data"]["forbidden_selection_inputs"]),
    ]
    for path in exclusion_paths:
        rows = read_jsonl(path)
        excluded_rows[manifest_task(rows, path)].extend(rows)
    excluded = {
        task: collect_leakage_values(task, rows) for task, rows in excluded_rows.items()
    }

    mt_source = read_jsonl(ROOT / "data/processed/manifests/mt--validation.jsonl")
    asr_source = read_jsonl(ROOT / "data/processed/manifests/asr--validation-local.jsonl")
    mt_candidates = filter_disjoint("mt", mt_source, excluded["mt"])
    asr_candidates = filter_disjoint("asr", asr_source, excluded["asr"])
    if len(mt_candidates) < args.mt_size or len(asr_candidates) < args.asr_size:
        raise ValueError("Not enough disjoint validation rows for requested selection sizes")
    mt = build_mt(mt_candidates, args.mt_size)
    asr = build_asr(asr_candidates, args.asr_size)
    ensure_no_overlap("mt", mt, excluded["mt"])
    ensure_no_overlap("asr", asr, excluded["asr"])
    outputs = {
        ROOT / "data/eval/mt_selection_dev.jsonl": mt,
        ROOT / "data/eval/asr_selection_dev.jsonl": asr,
    }
    report = {}
    for path, rows in outputs.items():
        write_jsonl(path, rows)
        report[str(path.relative_to(ROOT))] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    report["excluded_validation_rows"] = {
        "mt": len(mt_source) - len(mt_candidates),
        "asr": len(asr_source) - len(asr_candidates),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

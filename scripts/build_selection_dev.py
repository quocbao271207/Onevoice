"""Build deterministic model-selection manifests from validation data only.

These files may be opened repeatedly during successive halving.  They are not
the blind test and never draw rows from either locked test manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
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


def ensure_no_test_overlap(
    selected: list[dict[str, Any]], test_rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> None:
    for key in keys:
        selected_values = {row.get(key) for row in selected if row.get(key)}
        test_values = {row.get(key) for row in test_rows if row.get(key)}
        overlap = selected_values & test_values
        if overlap:
            raise ValueError(f"Selection/test leakage for {key}: {len(overlap)} rows")


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

    mt = build_mt(
        read_jsonl(ROOT / "data/processed/manifests/mt--validation.jsonl"), args.mt_size
    )
    asr = build_asr(
        read_jsonl(ROOT / "data/processed/manifests/asr--validation-local.jsonl"),
        args.asr_size,
    )
    ensure_no_test_overlap(
        mt,
        read_jsonl(ROOT / "data/processed/manifests/mt--test.jsonl"),
        ("id", "pair_fingerprint"),
    )
    ensure_no_test_overlap(
        asr,
        read_jsonl(ROOT / "data/processed/manifests/asr--test-local.jsonl"),
        ("id", "text_fingerprint", "audio_sha256"),
    )
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
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

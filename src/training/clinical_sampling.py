"""Deterministic clinical-risk sampling for ASR and MT training only."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from src.pipeline.safety_guard import extract_quantities, has_negation


DRUG_TERMS = (
    "acetaminophen",
    "amoxicillin",
    "aspirin",
    "epinephrine",
    "heparin",
    "insulin",
    "metformin",
    "morphine",
    "paracetamol",
    "warfarin",
)

SPECIALTY_TERMS = (
    "anaphylactic shock",
    "cardiac arrest",
    "myocardial infarction",
    "pneumothorax",
    "sốc phản vệ",
    "ngừng tim",
    "nhồi máu cơ tim",
    "tràn khí màng phổi",
)

ASCII_MEDICAL_RE = re.compile(
    r"(?i)\b(?:ct|mri|covid-?19|hiv|spo2|ecg|bpm|mmhg|mg|mcg|ml|iu)\b"
)
VIETNAMESE_CHAR_RE = re.compile(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", re.I)
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\s*\d+)?(?!\w)")


def clinical_risk_tags(text: str, language: str | None = None) -> set[str]:
    """Return explainable tags used to up-weight safety-critical train rows."""
    folded = (text or "").casefold()
    tags: set[str] = set()
    if any(term in folded for term in DRUG_TERMS):
        tags.add("drug")
    if any(term in folded for term in SPECIALTY_TERMS):
        tags.add("specialty_term")
    if extract_quantities(text):
        tags.update(("dose", "number", "unit"))
    elif NUMBER_RE.search(text or ""):
        tags.add("number")
    if language in {"vi", "en"} and has_negation(text, language):
        tags.add("negation")
    if VIETNAMESE_CHAR_RE.search(text or "") and ASCII_MEDICAL_RE.search(text or ""):
        tags.add("code_switch")
    return tags


def row_risk_tags(row: dict[str, Any], task: str) -> set[str]:
    if task == "asr":
        tags = clinical_risk_tags(row.get("text", ""), "vi")
        if row.get("language") == "vi-code-switch":
            tags.add("code_switch")
        return tags
    if task == "mt":
        return clinical_risk_tags(row.get("source_text", ""), "en") | clinical_risk_tags(
            row.get("target_text", ""), "vi"
        )
    raise ValueError(f"Unsupported task: {task}")


def _stable_key(row: dict[str, Any], seed: int) -> str:
    identity = str(row.get("id") or row.get("pair_fingerprint") or row)
    return hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).hexdigest()


def prioritize_rows(rows: Iterable[dict[str, Any]], task: str, seed: int) -> list[dict[str, Any]]:
    """Place high-risk rows first for bounded smoke runs without random leakage."""
    return sorted(
        rows,
        key=lambda row: (-len(row_risk_tags(row, task)), _stable_key(row, seed)),
    )


def oversample_clinical_rows(
    rows: Iterable[dict[str, Any]],
    task: str,
    factor: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Duplicate risk-bearing train rows deterministically; never use on validation/test."""
    if factor < 1:
        raise ValueError("clinical oversample factor must be at least 1")
    ordered = prioritize_rows(rows, task, seed)
    output = list(ordered)
    if factor == 1:
        return output
    risky = [row for row in ordered if row_risk_tags(row, task)]
    for _ in range(factor - 1):
        output.extend(risky)
    return output

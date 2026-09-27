from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from scripts.evaluate_benchmarks import score_mt
from src.pipeline.safety_guard import extract_quantities, negation_count, validate_translation
from src.training.clinical_sampling import (
    clinical_risk_tags,
    oversample_clinical_rows,
    prioritize_rows,
)


ROOT = Path(__file__).resolve().parents[1]
SAFETY_SUITE = ROOT / "data" / "eval" / "medical_safety_mt.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_clinical_sampling_is_deterministic_and_train_only_by_contract():
    rows = [
        {"id": "plain", "text": "bệnh nhân ổn định", "language": "vi"},
        {"id": "dose", "text": "tiêm insulin 6 đơn vị", "language": "vi-code-switch"},
    ]
    first = prioritize_rows(rows, "asr", seed=7)
    second = prioritize_rows(reversed(rows), "asr", seed=7)
    assert [row["id"] for row in first] == ["dose", "plain"]
    assert [row["id"] for row in first] == [row["id"] for row in second]
    oversampled = oversample_clinical_rows(rows, "asr", factor=2, seed=7)
    assert [row["id"] for row in oversampled].count("dose") == 2
    assert [row["id"] for row in oversampled].count("plain") == 1


def test_risk_tags_cover_drugs_dose_numbers_units_negation_and_code_switch():
    tags = clinical_risk_tags("Không tiêm insulin 6 đơn vị nếu SpO2 dưới 92%.", "vi")
    assert {"drug", "dose", "number", "unit", "negation", "code_switch"} <= tags


def test_quantity_aliases_are_bilingual_and_composite():
    source = "Infuse heparin at 18 units/kg/hour for 6 hours."
    target = "Truyền heparin 18 đơn vị/kg/giờ trong 6 giờ."
    assert extract_quantities(source) == [("18", "unit/kg/h"), ("6", "h")]
    assert extract_quantities(target) == [("18", "unit/kg/h"), ("6", "h")]
    assert validate_translation(source, target, "en", "vi").safe


def test_quantity_parser_accepts_words_spacing_and_per_notation():
    assert extract_quantities("five milligrams for seven days") == [("5", "mg"), ("7", "day")]
    assert extract_quantities("18 units per kg/h and 5 ml/ hour") == [
        ("18", "unit/kg/h"),
        ("5", "ml/h"),
    ]


def test_article_before_number_is_not_a_medical_identifier():
    result = validate_translation(
        "A 3 mm nodule was found.",
        "Đã phát hiện một nốt 3 mm.",
        "en",
        "vi",
    )
    assert not any(issue.startswith("identifier_mismatch") for issue in result.issues)


def test_negation_count_detects_dropped_clause():
    assert negation_count("Do not give aspirin and do not start warfarin.", "en") == 2
    result = validate_translation(
        "Do not give aspirin and do not start warfarin.",
        "Không dùng aspirin và bắt đầu warfarin.",
        "en",
        "vi",
    )
    assert "negation_mismatch" in result.issues
    assert negation_count("Don't give aspirin.", "en") == 1
    assert negation_count("Đừng dùng aspirin.", "vi") == 1


def test_terminology_allows_reviewed_aliases_but_fails_missing_drug():
    safe = validate_translation(
        "Give epinephrine now.",
        "Dùng adrenaline ngay.",
        "en",
        "vi",
        terminology={"epinephrine": ["epinephrine", "adrenaline"]},
    )
    unsafe = validate_translation(
        "Give epinephrine now.",
        "Dùng thuốc ngay.",
        "en",
        "vi",
        terminology={"epinephrine": ["epinephrine", "adrenaline"]},
    )
    assert safe.safe
    assert any(issue.startswith("terminology_missing") for issue in unsafe.issues)


def test_locked_safety_suite_has_every_required_slice_and_unique_ids():
    rows = read_jsonl(SAFETY_SUITE)
    required = {"drug_name", "dose", "number", "unit", "negation", "terminology", "code_switch"}
    categories = {category for row in rows for category in row["categories"]}
    assert required <= categories
    assert len(rows) >= 16
    assert len({row["id"] for row in rows}) == len(rows)


def test_locked_safety_suite_checksum_matches_artifact_lock():
    lock = yaml.safe_load((ROOT / "configs" / "artifact_lock.yaml").read_text(encoding="utf-8"))
    assert hashlib.sha256(SAFETY_SUITE.read_bytes()).hexdigest() == lock["evaluation"][
        "medical_safety_mt_sha256"
    ]


def test_locked_safety_suite_has_no_exact_pair_overlap_with_training_manifest():
    train_path = ROOT / "data" / "processed" / "manifests" / "mt--train.jsonl"
    if not train_path.is_file():
        return
    train_pairs = {
        hashlib.sha256(
            f"{row['source_text'].strip()}\n{row['target_text'].strip()}".encode("utf-8")
        ).hexdigest()
        for row in read_jsonl(train_path)
    }
    suite_pairs = {
        hashlib.sha256(
            f"{row['source_text'].strip()}\n{row['target_text'].strip()}".encode("utf-8")
        ).hexdigest()
        for row in read_jsonl(SAFETY_SUITE)
    }
    assert not (train_pairs & suite_pairs)


def test_mt_report_exposes_category_gates():
    row = read_jsonl(SAFETY_SUITE)[0]
    prediction = {
        "id": row["id"],
        "direction": "en_to_vi",
        "source": row["source_text"],
        "reference": row["target_text"],
        "hypothesis": row["target_text"],
        "categories": row["categories"],
        "terminology": row["terminology"],
    }
    report = score_mt([prediction])
    assert report["clinical_safety_gate"]["pass"]
    assert report["categories"]["drug_name"]["safety_failure_rate"] == 0.0

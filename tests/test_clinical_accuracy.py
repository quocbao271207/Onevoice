from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import sacrebleu
import yaml

from scripts.evaluate_benchmarks import (
    _bootstrap_mt_confidence_intervals,
    score_asr,
    score_mt,
)
from scripts.run_baseline_benchmarks import asr_prediction_slices
from src.pipeline.safety_guard import extract_quantities, negation_count, validate_translation
from src.pipeline.mt_engine import MedicalLexicon
from src.training.clinical_sampling import (
    clinical_risk_tags,
    oversample_clinical_rows,
    prioritize_rows,
)


ROOT = Path(__file__).resolve().parents[1]
SAFETY_SUITE = ROOT / "data" / "eval" / "medical_safety_mt.jsonl"
ASR_SAFETY_SUITE = ROOT / "data" / "eval" / "medical_safety_asr_vi.jsonl"


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


def test_safety_guard_binds_each_dose_to_the_correct_drug_after_reordering():
    terminology = {"aspirin": "aspirin", "warfarin": "warfarin"}
    safe = validate_translation(
        "Give aspirin 5 mg and warfarin 10 mg.",
        "Dùng warfarin 10 mg và aspirin 5 mg.",
        "en",
        "vi",
        terminology=terminology,
    )
    swapped = validate_translation(
        "Give aspirin 5 mg and warfarin 10 mg.",
        "Dùng aspirin 10 mg và warfarin 5 mg.",
        "en",
        "vi",
        terminology=terminology,
    )

    assert safe.safe
    assert not swapped.safe
    assert any(issue.startswith("quantity_binding_mismatch") for issue in swapped.issues)


def test_safety_guard_binds_negation_to_the_correct_drug():
    terminology = {"aspirin": "aspirin", "warfarin": "warfarin"}
    result = validate_translation(
        "Do not give aspirin; start warfarin.",
        "Dùng aspirin; không bắt đầu warfarin.",
        "en",
        "vi",
        terminology=terminology,
    )

    assert not result.safe
    assert any(issue.startswith("negation_binding_mismatch") for issue in result.issues)


def test_runtime_lexicon_covers_locked_clinical_drugs():
    lexicon = MedicalLexicon()
    required = {
        "amoxicillin",
        "aspirin",
        "heparin",
        "insulin",
        "metformin",
        "morphine",
        "paracetamol",
        "warfarin",
    }
    assert required <= set(lexicon.en_to_vi)


def test_locked_safety_suite_has_every_required_slice_and_unique_ids():
    rows = read_jsonl(SAFETY_SUITE)
    required = {"drug_name", "dose", "number", "unit", "negation", "terminology", "code_switch"}
    categories = {category for row in rows for category in row["categories"]}
    assert required <= categories
    assert len(rows) >= 16
    assert len({row["id"] for row in rows}) == len(rows)


def test_locked_safety_references_pass_guard_in_both_directions():
    for row in read_jsonl(SAFETY_SUITE):
        for direction, source_key, target_key in (
            ("en_to_vi", "source_text", "target_text"),
            ("vi_to_en", "target_text", "source_text"),
        ):
            source_lang, target_lang = direction.split("_to_")
            result = validate_translation(
                row[source_key],
                row[target_key],
                source_lang,
                target_lang,
                terminology=row["terminology"][direction],
            )
            assert result.safe, (row["id"], direction, result.issues)


def test_locked_safety_suite_checksum_matches_artifact_lock():
    lock = yaml.safe_load((ROOT / "configs" / "artifact_lock.yaml").read_text(encoding="utf-8"))
    assert hashlib.sha256(SAFETY_SUITE.read_bytes()).hexdigest() == lock["evaluation"][
        "medical_safety_mt_sha256"
    ]


def test_locked_asr_safety_suite_is_complete_and_checksum_locked():
    rows = read_jsonl(ASR_SAFETY_SUITE)
    required = {"drug_name", "dose", "number", "unit", "negation", "terminology", "code_switch"}
    categories = {category for row in rows for category in row["categories"]}
    assert required <= categories
    assert len(rows) >= 16
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(set(row["categories"]) == set(row["safety_expectations"]) for row in rows)
    lock = yaml.safe_load((ROOT / "configs" / "artifact_lock.yaml").read_text(encoding="utf-8"))
    assert hashlib.sha256(ASR_SAFETY_SUITE.read_bytes()).hexdigest() == lock["evaluation"][
        "medical_safety_asr_vi_sha256"
    ]


def test_locked_asr_safety_suite_is_exact_subset_of_locked_test():
    test_rows = {
        row["id"]: row
        for row in read_jsonl(ROOT / "data" / "processed" / "manifests" / "asr--test-local.jsonl")
    }
    for row in read_jsonl(ASR_SAFETY_SUITE):
        source = test_rows[row["id"]]
        assert row["text"] == source["text"]
        assert row["audio_path"] == source["audio_path"]
        assert row["language"] == source["language"]


def test_asr_safety_scoring_preserves_explicit_alternatives_and_fails_missing_values():
    perfect = {
        "id": "safe",
        "reference": "Kali clorid 0,15 g.",
        "hypothesis": "Kali clorid 0 15 g",
        "categories": ["drug_name", "dose", "number", "unit"],
        "safety_expectations": {
            "drug_name": [["kali clorid"]],
            "dose": [["0,15 g"]],
            "number": [["0,15"]],
            "unit": [["g"]],
        },
    }
    failed = {
        **perfect,
        "id": "failed",
        "hypothesis": "Kali clorid",
    }
    report = score_asr([perfect, failed])
    assert report["categories"]["drug_name"]["safety_failure_rate"] == 0.0
    assert report["categories"]["dose"]["safety_failure_rate"] == 0.5


def test_asr_blind_dimensions_survive_prediction_and_slice_scoring():
    dimensions = {
        "accent_region": "Central",
        "role": "Doctor",
        "recording_condition": "clinic",
        "noise": True,
    }
    slices = asr_prediction_slices(
        {
            "blind_dimensions": dimensions,
            "speaker": "speaker-central-1",
            "group": "recording-central-1",
        }
    )
    assert slices == {
        "accent": "Central",
        "role": "Doctor",
        "speaker": "speaker-central-1",
        "group": "recording-central-1",
        "recording_condition": "clinic",
        "noise": True,
        "noise_snr": None,
    }

    report = score_asr(
        [
            {
                "id": "blind-asr-1",
                "reference": "Không dùng aspirin.",
                "hypothesis": "Không dùng aspirin.",
                **slices,
            }
        ]
    )
    assert report["slices"]["accent"]["Central"]["samples"] == 1
    assert report["slices"]["role"]["Doctor"]["samples"] == 1
    assert report["slices"]["noise"]["True"]["samples"] == 1
    assert report["wer_bootstrap_unit"] == "group"
    assert report["wer_bootstrap_clusters"] == 1
    assert report["unique_speakers"] == 1
    assert report["unique_groups"] == 1


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


def test_mt_category_gates_fail_swapped_doses_and_negation_scope():
    report = score_mt(
        [
            {
                "id": "swapped-dose",
                "direction": "en_to_vi",
                "source": "Give aspirin 5 mg and warfarin 10 mg.",
                "reference": "Dùng aspirin 5 mg và warfarin 10 mg.",
                "hypothesis": "Dùng aspirin 10 mg và warfarin 5 mg.",
                "categories": ["drug_name", "dose", "number", "unit"],
                "terminology": {
                    "en_to_vi": {"aspirin": "aspirin", "warfarin": "warfarin"}
                },
            },
            {
                "id": "swapped-negation",
                "direction": "en_to_vi",
                "source": "Do not give aspirin; start warfarin.",
                "reference": "Không dùng aspirin; bắt đầu warfarin.",
                "hypothesis": "Dùng aspirin; không bắt đầu warfarin.",
                "categories": ["drug_name", "negation"],
                "terminology": {
                    "en_to_vi": {"aspirin": "aspirin", "warfarin": "warfarin"}
                },
            },
        ]
    )

    assert report["categories"]["dose"]["safety_failure_rate"] == 1.0
    assert report["categories"]["negation"]["safety_failure_rate"] == 1.0
    assert not report["clinical_safety_gate"]["pass"]


def test_empty_mt_hypothesis_fails_every_clinical_hard_gate():
    categories = [
        "drug_name",
        "dose",
        "number",
        "unit",
        "negation",
        "terminology",
        "code_switch",
    ]
    report = score_mt(
        [
            {
                "id": "empty-output",
                "direction": "en_to_vi",
                "source": "Do not give aspirin 5 mg to patient T3.",
                "reference": "Không dùng aspirin 5 mg cho bệnh nhân T3.",
                "hypothesis": "",
                "categories": categories,
                "terminology": {"en_to_vi": {"aspirin": "aspirin"}},
            }
        ]
    )

    assert not report["clinical_safety_gate"]["pass"]
    for category in categories:
        assert report["categories"][category]["safety_failure_rate"] == 1.0
        assert report["categories"][category]["safety_issue_counts"]["empty_translation"] == 1


def test_mt_bootstrap_precomputed_statistics_match_full_rescoring():
    rows = [
        {"hypothesis": "Take aspirin daily.", "reference": "Take aspirin daily."},
        {"hypothesis": "No fever today.", "reference": "No fever today."},
        {"hypothesis": "Dose is 5 mg.", "reference": "Dose is 10 mg."},
        {"hypothesis": "Blood pressure stable.", "reference": "Blood pressure is stable."},
    ]
    repeats = 40
    seed = 17
    rng = np.random.default_rng(seed)
    expected_bleu = []
    expected_chrf = []
    for _ in range(repeats):
        sampled = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        hypotheses = [row["hypothesis"] for row in sampled]
        references = [row["reference"] for row in sampled]
        expected_bleu.append(
            sacrebleu.corpus_bleu(hypotheses, [references], tokenize="intl").score
        )
        expected_chrf.append(sacrebleu.corpus_chrf(hypotheses, [references]).score)

    actual = _bootstrap_mt_confidence_intervals(rows, repeats=repeats, seed=seed)
    assert actual["sacrebleu_bootstrap_95ci"] == pytest.approx(
        [np.quantile(expected_bleu, 0.025), np.quantile(expected_bleu, 0.975)]
    )
    assert actual["chrf2_bootstrap_95ci"] == pytest.approx(
        [np.quantile(expected_chrf, 0.025), np.quantile(expected_chrf, 0.975)]
    )

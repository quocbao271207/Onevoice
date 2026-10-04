"""Score saved model predictions without mixing inference and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import sacrebleu
import numpy as np
from jiwer import cer, process_words, wer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.safety_guard import validate_translation
from src.utils.text_normalization import NORMALIZER_VERSION, normalize_for_wer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


ASR_CRITICAL_CATEGORIES = {
    "drug_name",
    "dose",
    "number",
    "unit",
    "negation",
    "terminology",
    "code_switch",
}


def _contains_normalized_phrase(text: str, phrase: str) -> bool:
    normalized_text = f" {normalize_for_wer(text)} "
    normalized_phrase = normalize_for_wer(phrase)
    return bool(normalized_phrase) and f" {normalized_phrase} " in normalized_text


def score_asr_safety(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score explicit token/phrase preservation expectations for locked ASR audio."""
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for category in row.get("categories", []):
            by_category[str(category)].append(row)

    category_reports: dict[str, dict[str, Any]] = {}
    for category, group in sorted(by_category.items()):
        failures = []
        for row in group:
            expectations = row.get("safety_expectations", {}).get(category)
            if not isinstance(expectations, list) or not expectations:
                missing_groups = [["<missing expectation>"]]
            else:
                missing_groups = []
                for alternatives in expectations:
                    if not isinstance(alternatives, list) or not alternatives:
                        missing_groups.append(["<invalid expectation>"])
                        continue
                    if not any(
                        _contains_normalized_phrase(row["hypothesis"], str(alternative))
                        for alternative in alternatives
                    ):
                        missing_groups.append([str(value) for value in alternatives])
            if missing_groups:
                failures.append(
                    {
                        "id": row.get("id"),
                        "missing_any_of": missing_groups,
                        "reference": row.get("reference"),
                        "hypothesis": row.get("hypothesis"),
                    }
                )
        category_reports[category] = {
            "samples": len(group),
            "failures": len(failures),
            "safety_failure_rate": len(failures) / max(1, len(group)),
            "failure_examples": failures[:20],
        }

    evaluated = ASR_CRITICAL_CATEGORIES & set(category_reports)
    return {
        "categories": category_reports,
        "clinical_safety_gate": {
            "pass": evaluated == ASR_CRITICAL_CATEGORIES
            and all(category_reports[name]["safety_failure_rate"] == 0.0 for name in evaluated),
            "required_categories": sorted(ASR_CRITICAL_CATEGORIES),
            "evaluated_categories": sorted(evaluated),
            "policy": "zero detected critical transcription-preservation failures",
        },
    }


def score_asr(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def score(group: list[dict[str, Any]]) -> dict[str, float]:
        refs = [normalize_for_wer(row["reference"]) for row in group]
        hyps = [normalize_for_wer(row["hypothesis"]) for row in group]
        return {"wer": wer(refs, hyps), "cer": cer(refs, hyps)}

    def bootstrap_wer(group: list[dict[str, Any]], repeats: int = 1000) -> list[float]:
        counts = []
        for row in group:
            result = process_words(
                normalize_for_wer(row["reference"]),
                normalize_for_wer(row["hypothesis"]),
            )
            counts.append((result.substitutions + result.deletions + result.insertions, result.hits + result.substitutions + result.deletions))
        array = np.asarray(counts, dtype=np.float64)
        rng = np.random.default_rng(20260922)
        values = []
        for _ in range(repeats):
            sample = array[rng.integers(0, len(array), len(array))]
            values.append(float(sample[:, 0].sum() / max(1.0, sample[:, 1].sum())))
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

    slices: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key in ("source", "accent", "role", "recording_condition", "code_switch", "noise_snr"):
            if row.get(key) not in (None, ""):
                slices[key][str(row[key])].append(row)
    overall = score(rows)
    report = {
        "task": "asr",
        "normalizer": NORMALIZER_VERSION,
        "samples": len(rows),
        **overall,
        "wer_bootstrap_95ci": bootstrap_wer(rows),
        "slices": {
            key: {value: {"samples": len(group), **score(group)} for value, group in groups.items()}
            for key, groups in slices.items()
        },
    }
    if any(row.get("categories") or row.get("safety_expectations") for row in rows):
        report.update(score_asr_safety(rows))
    return report


def score_mt(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def score(
        group: list[dict[str, Any]],
        relevant_issue_prefixes: set[str] | None = None,
    ) -> dict[str, Any]:
        hypotheses = [row["hypothesis"] for row in group]
        references = [row["reference"] for row in group]
        safety_failures = 0
        issue_counts: dict[str, int] = defaultdict(int)
        failure_examples = []
        for row in group:
            direction = row["direction"]
            source_lang, target_lang = direction.split("_to_")
            terminology = row.get("terminology", {})
            if direction in terminology and isinstance(terminology[direction], dict):
                terminology = terminology[direction]
            check = validate_translation(
                row["source"],
                row["hypothesis"],
                source_lang,
                target_lang,
                terminology=terminology,
            )
            relevant_issues = [
                issue
                for issue in check.issues
                if relevant_issue_prefixes is None
                or issue.split(":", 1)[0] in relevant_issue_prefixes
            ]
            if relevant_issues:
                safety_failures += 1
                for issue in relevant_issues:
                    issue_counts[issue.split(":", 1)[0]] += 1
                if len(failure_examples) < 20:
                    failure_examples.append(
                        {
                            "id": row.get("id"),
                            "direction": direction,
                            "issues": relevant_issues,
                            "source": row["source"],
                            "reference": row["reference"],
                            "hypothesis": row["hypothesis"],
                        }
                    )
        bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="intl")
        chrf = sacrebleu.corpus_chrf(hypotheses, [references])
        return {
            "samples": len(group),
            "sacrebleu": bleu.score,
            "sacrebleu_signature": str(bleu),
            "chrf2": chrf.score,
            "safety_failure_rate": safety_failures / max(1, len(group)),
            "safety_issue_counts": dict(issue_counts),
            "safety_failure_examples": failure_examples,
        }

    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_direction[row["direction"]].append(row)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for category in row.get("categories", []):
            by_category[str(category)].append(row)
    overall = score(rows)
    category_issue_prefixes = {
        "drug_name": {"terminology_missing"},
        "dose": {"number_mismatch", "unit_mismatch", "quantity_mismatch"},
        "number": {"number_mismatch"},
        "unit": {"unit_mismatch", "quantity_mismatch"},
        "negation": {"negation_mismatch"},
        "terminology": {"terminology_missing"},
        "code_switch": {"identifier_mismatch", "terminology_missing"},
    }
    category_scores = {
        category: score(group, category_issue_prefixes.get(category))
        for category, group in sorted(by_category.items())
    }
    critical_categories = {
        "drug_name", "dose", "number", "unit", "negation", "terminology", "code_switch"
    }
    evaluated_critical = critical_categories & set(category_scores)
    clinical_gate_pass = bool(evaluated_critical) and all(
        category_scores[category]["safety_failure_rate"] == 0.0
        for category in evaluated_critical
    )
    return {
        "task": "mt",
        **overall,
        "directions": {direction: score(group) for direction, group in sorted(by_direction.items())},
        "categories": category_scores,
        "clinical_safety_gate": {
            "pass": clinical_gate_pass,
            "required_categories": sorted(critical_categories),
            "evaluated_categories": sorted(evaluated_critical),
            "policy": "zero detected critical preservation failures",
        },
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["asr", "mt"], required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--manifest")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    rows = read_jsonl(args.predictions)
    if not rows:
        raise SystemExit("Predictions file is empty")
    report = score_asr(rows) if args.task == "asr" else score_mt(rows)
    report["predictions"] = str(args.predictions)
    if args.model:
        report["model"] = args.model
    if args.manifest:
        report["manifest"] = args.manifest
    if args.seed is not None:
        report["seed"] = args.seed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

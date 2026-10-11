"""Rescore all existing bake-off model predictions with refined clinical safety guard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.evaluate_benchmarks import read_jsonl, score_asr, score_mt

BAKEOFF_DIR = Path("/workspace/test/Onevoice/gpu-runs/model-bakeoff-20261005")
REPORTS_DIR = BAKEOFF_DIR / "selection-reports"


def rescore_all() -> dict[str, Any]:
    results = {"mt": {}, "asr": {}}

    # 1. MT models
    mt_pred_files = sorted(REPORTS_DIR.glob("mt/**/selection_predictions.jsonl"))
    print(f"Found {len(mt_pred_files)} MT prediction files.")
    for pred_path in mt_pred_files:
        run_name = str(pred_path.relative_to(REPORTS_DIR).parent)
        selection_json_path = pred_path.parent / "selection.json"
        old_data = {}
        if selection_json_path.exists():
            try:
                old_data = json.loads(selection_json_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Error loading {selection_json_path}: {e}")

        rows = read_jsonl(pred_path)
        new_report = score_mt(rows)

        # Compare old vs new
        old_categories = old_data.get("categories", {})
        new_categories = new_report.get("categories", {})
        cat_comparison = {}
        for cat in sorted(set(old_categories) | set(new_categories)):
            old_rate = old_categories.get(cat, {}).get("safety_failure_rate")
            new_rate = new_categories.get(cat, {}).get("safety_failure_rate")
            old_counts = old_categories.get(cat, {}).get("safety_issue_counts", {})
            new_counts = new_categories.get(cat, {}).get("safety_issue_counts", {})
            cat_comparison[cat] = {
                "old_failure_rate": old_rate,
                "new_failure_rate": new_rate,
                "old_issues": old_counts,
                "new_issues": new_counts,
            }

        old_overall_rate = old_data.get("safety_failure_rate")
        new_overall_rate = new_report.get("safety_failure_rate")
        old_gate = old_data.get("clinical_safety_gate", {}).get("pass")
        new_gate = new_report.get("clinical_safety_gate", {}).get("pass")

        results["mt"][run_name] = {
            "samples": len(rows),
            "sacrebleu": new_report.get("sacrebleu"),
            "chrf2": new_report.get("chrf2"),
            "old_safety_failure_rate": old_overall_rate,
            "new_safety_failure_rate": new_overall_rate,
            "old_clinical_gate_pass": old_gate,
            "new_clinical_gate_pass": new_gate,
            "category_comparison": cat_comparison,
            "remaining_failure_examples": new_report.get("safety_failure_examples", [])[:10],
        }
        print(f"MT {run_name}: failure rate {old_overall_rate} -> {new_overall_rate}")

    # 2. ASR models
    asr_pred_files = sorted(REPORTS_DIR.glob("asr/**/selection_predictions.jsonl"))
    print(f"Found {len(asr_pred_files)} ASR prediction files.")
    for pred_path in asr_pred_files:
        run_name = str(pred_path.relative_to(REPORTS_DIR).parent)
        selection_json_path = pred_path.parent / "selection.json"
        old_data = {}
        if selection_json_path.exists():
            try:
                old_data = json.loads(selection_json_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Error loading {selection_json_path}: {e}")

        rows = read_jsonl(pred_path)
        new_report = score_asr(rows)

        old_categories = old_data.get("categories", {})
        new_categories = new_report.get("categories", {})
        cat_comparison = {}
        for cat in sorted(set(old_categories) | set(new_categories)):
            old_rate = old_categories.get(cat, {}).get("safety_failure_rate")
            new_rate = new_categories.get(cat, {}).get("safety_failure_rate")
            old_failures = old_categories.get(cat, {}).get("failures")
            new_failures = new_categories.get(cat, {}).get("failures")
            cat_comparison[cat] = {
                "old_failures": old_failures,
                "new_failures": new_failures,
                "old_failure_rate": old_rate,
                "new_failure_rate": new_rate,
            }

        results["asr"][run_name] = {
            "samples": len(rows),
            "wer": new_report.get("wer"),
            "cer": new_report.get("cer"),
            "old_clinical_gate_pass": old_data.get("clinical_safety_gate", {}).get("pass"),
            "new_clinical_gate_pass": new_report.get("clinical_safety_gate", {}).get("pass"),
            "category_comparison": cat_comparison,
        }
        print(f"ASR {run_name}: gates old={results['asr'][run_name]['old_clinical_gate_pass']} -> new={results['asr'][run_name]['new_clinical_gate_pass']}")

    return results


if __name__ == "__main__":
    report = rescore_all()
    out_path = BAKEOFF_DIR / "rescored_comparison.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved rescored comparison to {out_path}")

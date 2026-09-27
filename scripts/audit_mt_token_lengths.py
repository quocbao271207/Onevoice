"""Measure NLLB token lengths on locked MedEV manifests before GPU training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.finetune_mt_medical import MTTrainingConfig  # noqa: E402


def summarize(values: list[int], max_length: int) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.int32)
    over = int(np.sum(array > max_length))
    return {
        "rows": int(len(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
        "over_max_length": over,
        "over_max_length_percent": round(100.0 * over / max(1, len(array)), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data" / "processed" / "manifests")
    parser.add_argument("--max-length", type=int, default=MTTrainingConfig.max_source_length)
    parser.add_argument("--max-truncation-percent", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "reports" / "eda" / "medev_token_lengths.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MTTrainingConfig.base_model,
        revision=MTTrainingConfig.base_model_revision,
    )
    report = {
        "model": MTTrainingConfig.base_model,
        "revision": MTTrainingConfig.base_model_revision,
        "max_length": args.max_length,
        "max_truncation_percent": args.max_truncation_percent,
        "roles": {},
    }
    failures = []
    for role in ("train", "validation", "test"):
        lengths = {"en": [], "vi": []}
        pending = {"en": [], "vi": []}

        def flush() -> None:
            for language, code in (("en", "eng_Latn"), ("vi", "vie_Latn")):
                if not pending[language]:
                    continue
                tokenizer.src_lang = code
                encoded = tokenizer(pending[language], add_special_tokens=True, truncation=False)
                lengths[language].extend(len(ids) for ids in encoded["input_ids"])
                pending[language].clear()

        path = args.manifest_dir / f"mt--{role}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                pending["en"].append(row["source_text"])
                pending["vi"].append(row["target_text"])
                if len(pending["en"]) >= args.batch_size:
                    flush()
        flush()
        role_report = {language: summarize(values, args.max_length) for language, values in lengths.items()}
        report["roles"][role] = role_report
        for language, summary in role_report.items():
            if float(summary["over_max_length_percent"]) > args.max_truncation_percent:
                failures.append(f"{role}:{language}={summary['over_max_length_percent']}%")

    report["status"] = "pass" if not failures else "review_required"
    report["failures"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())

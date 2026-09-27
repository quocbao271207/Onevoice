"""Deep structural/language/number audit for aligned MedEV EN↔VI pairs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.quality import vietnamese_mark_ratio  # noqa: E402


NUMBER_RE = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)?")
VI_MARK_CHARS = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")


def comparison_text(text: str) -> str:
    """Normalize punctuation/spacing so untranslated identity pairs are visible."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def numbers(text: str) -> set[str]:
    values = set()
    for raw in NUMBER_RE.findall(text):
        try:
            value = Decimal(raw.replace(",", "."))
            values.add(format(value.normalize(), "f"))
        except InvalidOperation:
            values.add(raw)
    return values


def marked_letters(text: str) -> int:
    return sum(character.casefold() in VI_MARK_CHARS for character in text)


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=ROOT / "data" / "reports" / "eda" / "records")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "reports" / "eda" / "medev_alignment.json")
    args = parser.parse_args()

    split_reports = {}
    for path in sorted(args.records_dir.glob("medev--*.jsonl")):
        split = path.stem.split("--", 1)[1]
        counts: Counter[str] = Counter()
        ratios: list[float] = []
        source_lengths: list[float] = []
        target_lengths: list[float] = []
        examples: dict[str, list[dict]] = {}
        rewritten = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                source = record["source_text"]
                target = record["target_text"]
                source_length = len(source)
                target_length = len(target)
                ratio = target_length / max(1, source_length)
                flags = set(record.get("quality_flags") or [])
                source_marks = marked_letters(source)
                target_ratio = vietnamese_mark_ratio(target)
                if source_marks >= 3 and vietnamese_mark_ratio(source) > 0.03:
                    flags.add("source_language_suspect")
                if target_length >= 30 and target_ratio < 0.005:
                    flags.add("target_language_suspect")
                if min(source_length, target_length) >= 20 and comparison_text(source) == comparison_text(target):
                    flags.add("untranslated_identity")
                if ratio < 0.25 or ratio > 4.0:
                    flags.add("length_ratio_outlier")
                if max(source_length, target_length) > 2000:
                    flags.add("very_long_pair")
                source_numbers = numbers(source)
                target_numbers = numbers(target)
                if source_numbers != target_numbers:
                    flags.add("number_set_mismatch")
                record["quality_flags"] = sorted(flags)
                record["alignment_features"] = {
                    "source_chars": source_length,
                    "target_chars": target_length,
                    "target_source_char_ratio": round(ratio, 5),
                    "source_vi_mark_count": source_marks,
                    "target_vi_mark_ratio": round(target_ratio, 5),
                    "source_numbers": sorted(source_numbers),
                    "target_numbers": sorted(target_numbers),
                }
                rewritten.append(json.dumps(record, ensure_ascii=False))
                ratios.append(ratio)
                source_lengths.append(float(source_length))
                target_lengths.append(float(target_length))
                counts.update(flags)
                for flag in sorted(flags):
                    bucket = examples.setdefault(flag, [])
                    if len(bucket) < 12:
                        bucket.append(
                            {
                                "id": record["id"],
                                "source": source[:300],
                                "target": target[:300],
                            }
                        )
        temp = path.with_suffix(".tmp")
        temp.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
        os.replace(temp, path)
        split_reports[split] = {
            "rows": len(ratios),
            "flag_counts": dict(sorted(counts.items())),
            "target_source_char_ratio": {
                "p01": round(quantile(ratios, 0.01), 4),
                "p50": round(quantile(ratios, 0.50), 4),
                "p99": round(quantile(ratios, 0.99), 4),
            },
            "source_chars_p95": round(quantile(source_lengths, 0.95), 2),
            "target_chars_p95": round(quantile(target_lengths, 0.95), 2),
            "examples": {flag: examples[flag] for flag in sorted(examples)},
        }

    report = {
        "status": "review_required",
        "fatal_candidates": [
            "length_ratio_outlier",
            "very_long_pair",
            "source_language_suspect",
            "target_language_suspect",
            "untranslated_identity",
        ],
        "review_slices": ["number_set_mismatch"],
        "splits": split_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({split: {"rows": value["rows"], "flag_counts": value["flag_counts"], "ratios": value["target_source_char_ratio"]} for split, value in split_reports.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

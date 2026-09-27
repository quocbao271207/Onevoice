"""High-precision PII risk scan over audited manifests before merge."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "url": re.compile(r"\b(?:https?://|www\.)\S+", re.I),
    "phone_vn": re.compile(r"(?<!\d)(?:\+?84|0)(?:[ .-]?\d){9}(?!\d)"),
    "long_identifier": re.compile(r"(?<![\d.,])\d{9,12}(?![\d.,]|\s*(?:mg|g|ml|mmhg|cm|mm)\b)", re.I),
    "address_marker": re.compile(r"\b(?:địa chỉ|số nhà|address|home address)\b", re.I),
    "name_marker": re.compile(r"\b(?:tên tôi là|tôi tên là|my name is|patient name)\b", re.I),
}


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=ROOT / "data" / "reports" / "eda" / "records")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "reports" / "eda" / "pii_scan.json")
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    scanned = 0
    candidates = []
    for path in sorted(args.records_dir.glob("*.jsonl")):
        for record in iter_jsonl(path):
            scanned += 1
            fields = {
                "text": record.get("text", ""),
                "source_text": record.get("source_text", ""),
                "target_text": record.get("target_text", ""),
            }
            hits = []
            for field, value in fields.items():
                for kind, pattern in PATTERNS.items():
                    matches = pattern.findall(str(value or ""))
                    if matches:
                        counts[kind] += len(matches)
                        hits.append({"field": field, "kind": kind, "matches": matches[:3]})
            if hits:
                candidates.append(
                    {
                        "id": record.get("id"),
                        "source": record.get("source"),
                        "split": record.get("source_split"),
                        "hits": hits,
                        "text_preview": str(record.get("text") or record.get("source_text") or "")[:300],
                    }
                )
    report = {
        "scanner": "high_precision_regex_v1",
        "scanned_records": scanned,
        "candidate_records": len(candidates),
        "hit_counts": dict(counts),
        "status": "review_required" if candidates else "pass",
        "candidates": candidates,
        "limitations": "Names without explicit markers and contextual identifiers require separate human/governance review.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "candidates"}, ensure_ascii=False, indent=2))
    return 2 if candidates else 0


if __name__ == "__main__":
    raise SystemExit(main())

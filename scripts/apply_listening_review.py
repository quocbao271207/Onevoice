"""Validate a completed listening review and attach its decision to EDA audits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("audio_ok", "transcript_exact", "language", "pii", "verdict_keep_drop")
YES_NO = {"yes", "no"}
LANGUAGE_BY_SOURCE = {
    "eka_medical_en": "en",
    "vietmed": "vi",
    "vimedcss": "mixed",
    "vivos": "vi",
}


def refresh_audit_index(audit_dir: Path) -> None:
    """Keep aggregate EDA artifacts synchronized after human decisions."""
    datasets = []
    for path in sorted(audit_dir.glob("*.json")):
        if path.name == "audit_index.json":
            continue
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(item, dict) and item.get("dataset"):
            datasets.append(item)
    datasets.sort(key=lambda item: str(item["dataset"]))
    (audit_dir / "audit_index.json").write_text(
        json.dumps({"datasets": datasets}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# OneVoice Dataset EDA Report",
        "",
        "> Generated before merge and synchronized after human review.",
        "",
        "| Dataset | Status | Full | Audited | License |",
        "|---|---:|---:|---:|---|",
    ]
    for item in datasets:
        counts = item.get("audited_rows") or item.get("paired_rows") or {}
        lines.append(
            f"| `{item['dataset']}` | {item.get('status')} | {item.get('full_audit')} | "
            f"{sum(counts.values()):,} | {item.get('license', 'unknown')} |"
        )
    pending = [item["dataset"] for item in datasets if item.get("status") not in {"pass", "reviewed"}]
    lines.extend(["", "## Required human decisions", ""])
    lines.extend([f"- `{name}`: unresolved." for name in pending] or ["- None."])
    (audit_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, default=ROOT / "data" / "review" / "listening_pack" / "review.csv")
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "data" / "reports" / "eda")
    parser.add_argument("--reviewer", required=True, help="Name or stable reviewer ID recorded in the audit trail.")
    parser.add_argument(
        "--owner-confirmed-clean",
        action="store_true",
        help="Fill clean verdicts after the owner confirms audio/transcript match; refuses rows hit by the PII scan.",
    )
    args = parser.parse_args()

    with args.review_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Review CSV has no samples")

    if args.owner_confirmed_clean:
        pii_path = args.audit_dir / "pii_scan.json"
        pii = json.loads(pii_path.read_text(encoding="utf-8")) if pii_path.is_file() else {"candidates": []}
        pii_ids = {item.get("id") for item in pii.get("candidates", [])}
        review_pii_hits = [row["id"] for row in rows if row.get("id") in pii_ids]
        if review_pii_hits:
            raise SystemExit(f"Cannot bulk-confirm review rows with PII candidates: {review_pii_hits}")
        for row in rows:
            row["audio_ok"] = "yes"
            row["transcript_exact"] = "yes"
            row["language"] = LANGUAGE_BY_SOURCE.get(row.get("source", ""), "unknown")
            row["pii"] = "no"
            row["verdict_keep_drop"] = "keep"
            row["note"] = "Owner confirmed transcript/audio match; no high-precision PII scan hit in this sample."
        with args.review_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    errors: list[str] = []
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line_number, row in enumerate(rows, start=2):
        by_source[row.get("source", "")].append(row)
        for field in REQUIRED:
            if not row.get(field, "").strip():
                errors.append(f"line {line_number}: missing {field}")
        for field in ("audio_ok", "transcript_exact", "pii"):
            if row.get(field, "").strip().casefold() not in YES_NO:
                errors.append(f"line {line_number}: {field} must be yes/no")
        if row.get("verdict_keep_drop", "").strip().casefold() not in {"keep", "drop"}:
            errors.append(f"line {line_number}: verdict_keep_drop must be keep/drop")
    if errors:
        raise SystemExit("Incomplete/invalid review:\n- " + "\n- ".join(errors[:50]))

    timestamp = datetime.now(timezone.utc).isoformat()
    for source, source_rows in by_source.items():
        audit_path = args.audit_dir / f"{source}.json"
        if not audit_path.is_file():
            raise SystemExit(f"Missing audit for reviewed source: {audit_path}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        verdicts = Counter(row["verdict_keep_drop"].strip().casefold() for row in source_rows)
        audit["human_listening_review"] = {
            "reviewer": args.reviewer,
            "reviewed_at_utc": timestamp,
            "sample_count": len(source_rows),
            "verdicts": dict(verdicts),
            "audio_ok": dict(Counter(row["audio_ok"].strip().casefold() for row in source_rows)),
            "transcript_exact": dict(Counter(row["transcript_exact"].strip().casefold() for row in source_rows)),
            "pii_observed": dict(Counter(row["pii"].strip().casefold() for row in source_rows)),
            "review_csv": str(args.review_csv),
        }
        # Listening review resolves sample-level audio/transcript uncertainty only.
        # Unknown licenses and structural leakage remain hard blockers.
        if audit.get("license") != "unknown" and not audit.get("structural_errors"):
            audit["status"] = "reviewed"
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{source}: {dict(verdicts)} -> status={audit['status']}")
    refresh_audit_index(args.audit_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

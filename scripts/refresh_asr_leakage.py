"""Re-label ASR roles from config and recompute leakage without network access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_datasets import (  # noqa: E402
    durable_asr_record,
    markdown_report,
    role_for_split,
    select_listening_records,
)
from src.data.quality import cross_split_leakage, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "data" / "reports" / "eda")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    for name, spec in config["datasets"].items():
        if not spec.get("enabled") or spec.get("task") != "asr":
            continue
        audit_path = args.audit_dir / f"{name}.json"
        if not audit_path.is_file():
            continue
        records: list[dict] = []
        for path in sorted((args.audit_dir / "records").glob(f"{name}--*.jsonl")):
            rewritten: list[str] = []
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = durable_asr_record(json.loads(line))
                    record["role"] = role_for_split(spec, record["source_split"])
                    records.append(record)
                    rewritten.append(json.dumps(record, ensure_ascii=False))
            path.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")

        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["leakage"] = cross_split_leakage(records)
        status = "pass"
        if not audit.get("full_audit") or audit.get("license") == "unknown":
            status = "review_required"
        scored_roles = {"train", "validation", "test"}
        for dimension in ("speaker", "group"):
            for pair, count in audit["leakage"][dimension]["role_pair_counts"].items():
                if count and set(pair.split("<->")) <= scored_roles:
                    status = "review_required"
        audit["automatic_status"] = status
        audit["status"] = "reviewed" if audit.get("human_listening_review") else status
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        queue_path = args.audit_dir / "listening" / f"{name}.jsonl"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue = select_listening_records(records)
        queue_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in queue) + "\n",
            encoding="utf-8",
        )
        print(f"{name}: records={len(records):,} status={audit['status']} automatic_status={status}")

    summaries = []
    for path in sorted(args.audit_dir.glob("*.json")):
        if path.name == "audit_index.json":
            continue
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("dataset"):
            summaries.append(item)
    (args.audit_dir / "REPORT.md").write_text(markdown_report(summaries), encoding="utf-8")
    write_json(args.audit_dir / "audit_index.json", {"datasets": summaries})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

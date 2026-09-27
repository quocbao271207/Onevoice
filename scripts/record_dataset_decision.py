"""Record an explicit owner decision in a dataset audit trail."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--decision", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--reviewer", default="project_owner")
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "data" / "reports" / "eda")
    args = parser.parse_args()

    path = args.audit_dir / f"{args.dataset}.json"
    if not path.is_file():
        raise SystemExit(f"Audit not found: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not audit.get("full_audit"):
        raise SystemExit("Cannot approve a sampled audit")
    if audit.get("structural_errors"):
        raise SystemExit(f"Cannot approve structural errors: {audit['structural_errors']}")
    audit["owner_decision"] = {
        "decision": args.decision,
        "scope": args.scope,
        "reviewer": args.reviewer,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    audit["status"] = "reviewed"
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.dataset}: status=reviewed; scope={args.scope}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

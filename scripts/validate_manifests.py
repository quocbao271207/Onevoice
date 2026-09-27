"""Validate merged manifests and emit reproducible counts/checksums before audio download."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROLES = ("train", "validation", "test")


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dimension(rows: list[dict], key: str) -> set[str]:
    return {str(row.get(key)).strip().casefold() for row in rows if row.get(key)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data" / "processed" / "manifests")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "reports" / "manifests" / "validation.json")
    args = parser.parse_args()

    report = {"manifest_dir": str(args.manifest_dir), "tasks": {}, "errors": []}
    for task in ("asr", "mt"):
        rows = {role: load(args.manifest_dir / f"{task}--{role}.jsonl") for role in ROLES}
        task_report = {"roles": {}, "cross_role": {}}
        for role, role_rows in rows.items():
            source_counts = Counter(row.get("source", "unknown") for row in role_rows)
            policy_counts = Counter(row.get("merge_policy", "unknown") for row in role_rows)
            hours = sum(float(row.get("duration_s") or 0) for row in role_rows) / 3600
            path = args.manifest_dir / f"{task}--{role}.jsonl"
            task_report["roles"][role] = {
                "rows": len(role_rows),
                "hours": round(hours, 4),
                "sources": dict(source_counts),
                "merge_policies": dict(policy_counts),
                "sha256": sha256(path),
            }

        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            pair = f"{left}<->{right}"
            ids = dimension(rows[left], "id") & dimension(rows[right], "id")
            if ids:
                report["errors"].append(f"{task}:{pair}:duplicate_ids={len(ids)}")
            result = {"duplicate_ids": len(ids)}
            if task == "asr":
                groups = dimension(rows[left], "group") & dimension(rows[right], "group")
                speakers = dimension(rows[left], "speaker") & dimension(rows[right], "speaker")
                transcripts = dimension(rows[left], "text_fingerprint") & dimension(rows[right], "text_fingerprint")
                result.update(
                    {
                        "recording_group_overlap": len(groups),
                        "speaker_overlap": len(speakers),
                        "transcript_overlap_observed_not_blocking": len(transcripts),
                    }
                )
                if groups:
                    report["errors"].append(f"asr:{pair}:group_overlap={len(groups)}")
                if speakers:
                    report["errors"].append(f"asr:{pair}:speaker_overlap={len(speakers)}")
            else:
                pairs = dimension(rows[left], "pair_fingerprint") & dimension(rows[right], "pair_fingerprint")
                result["exact_pair_overlap"] = len(pairs)
                if pairs:
                    report["errors"].append(f"mt:{pair}:pair_overlap={len(pairs)}")
            task_report["cross_role"][pair] = result
        report["tasks"][task] = task_report

    report["status"] = "pass" if not report["errors"] else "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Create a non-resumable forensic manifest for a legacy evidence archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.candidate_evidence import derive_legacy_evidence_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output, manifest = derive_legacy_evidence_manifest(args.archive, args.output)
    print(
        json.dumps(
            {
                "status": "verified_legacy_archive",
                "derived_manifest": str(output),
                "archive_sha256": manifest["archive_sha256"],
                "archive_bytes": manifest["archive_bytes"],
                "file_count": manifest["file_count"],
                "content_bytes": manifest["content_bytes"],
                "resume_eligible": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

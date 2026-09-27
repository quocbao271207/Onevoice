"""Summarize final signal-QC manifests, portable paths and cross-role isolation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "processed" / "manifests"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def values(rows: list[dict], key: str) -> set[str]:
    return {str(row.get(key)).strip().casefold() for row in rows if row.get(key)}


def quantile(values_: list[float], q: float) -> float | None:
    if not values_:
        return None
    ordered = sorted(values_)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 6)


def distribution(values_: list[float]) -> dict[str, float | None]:
    return {
        "min": round(min(values_), 6) if values_ else None,
        "p01": quantile(values_, 0.01),
        "p50": quantile(values_, 0.50),
        "p95": quantile(values_, 0.95),
        "p99": quantile(values_, 0.99),
        "max": round(max(values_), 6) if values_ else None,
    }


def main() -> int:
    roles = {}
    rows_by_role = {}
    errors = []
    for role in ("train", "validation", "test"):
        path = MANIFEST_DIR / f"asr--{role}-local.jsonl"
        rows = load(path)
        rows_by_role[role] = rows
        missing = 0
        nonportable = 0
        by_source = Counter()
        hours_by_source = Counter()
        quality_flags = Counter()
        repairs = Counter()
        signal_values = {
            "duration_s": [],
            "rms": [],
            "peak": [],
            "dc_offset": [],
            "clipped_fraction": [],
            "decoded_chars_per_second": [],
        }
        audio_hashes: list[str] = []
        for row in rows:
            audio_path = Path(row["audio_path"])
            nonportable += int(audio_path.is_absolute())
            resolved = audio_path if audio_path.is_absolute() else ROOT / audio_path
            missing += int(not resolved.is_file())
            source = str(row.get("source") or "unknown")
            by_source[source] += 1
            qc = row.get("audio_qc", {})
            hours_by_source[source] += float(qc.get("duration_s") or 0) / 3600
            quality_flags.update(row.get("quality_flags") or [])
            repairs.update(qc.get("repairs") or [])
            for key in signal_values:
                value = qc.get(key)
                if value is not None:
                    signal_values[key].append(float(value))
            if row.get("audio_sha256"):
                audio_hashes.append(str(row["audio_sha256"]))
        ids = values(rows, "id")
        if len(ids) != len(rows):
            errors.append(f"{role}:duplicate_ids={len(rows) - len(ids)}")
        if missing:
            errors.append(f"{role}:missing_audio={missing}")
        if nonportable:
            errors.append(f"{role}:absolute_paths={nonportable}")
        if len(audio_hashes) != len(rows):
            errors.append(f"{role}:missing_audio_sha256={len(rows) - len(audio_hashes)}")
        if len(set(audio_hashes)) != len(audio_hashes):
            errors.append(f"{role}:duplicate_audio_hashes={len(audio_hashes) - len(set(audio_hashes))}")
        roles[role] = {
            "rows": len(rows),
            "hours": round(sum(hours_by_source.values()), 4),
            "sources": dict(by_source),
            "hours_by_source": {name: round(value, 4) for name, value in sorted(hours_by_source.items())},
            "missing_audio": missing,
            "absolute_paths": nonportable,
            "unique_audio_hashes": len(set(audio_hashes)),
            "signal_distributions": {key: distribution(values_) for key, values_ in signal_values.items()},
            "quality_flags": dict(sorted(quality_flags.items())),
            "repairs": dict(sorted(repairs.items())),
            "sha256": sha256(path),
        }

    cross_role = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        result = {
            "duplicate_ids": len(values(rows_by_role[left], "id") & values(rows_by_role[right], "id")),
            "speaker_overlap": len(values(rows_by_role[left], "speaker") & values(rows_by_role[right], "speaker")),
            "recording_group_overlap": len(values(rows_by_role[left], "group") & values(rows_by_role[right], "group")),
            "exact_audio_overlap": len(values(rows_by_role[left], "audio_sha256") & values(rows_by_role[right], "audio_sha256")),
        }
        if any(result.values()):
            errors.append(f"{left}<->{right}:{result}")
        cross_role[f"{left}<->{right}"] = result

    report = {"status": "pass" if not errors else "fail", "roles": roles, "cross_role": cross_role, "errors": errors}
    output = ROOT / "data" / "reports" / "local_audio_summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Decode, signal-QC, and losslessly standardize local ASR audio to mono 16 kHz FLAC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
ROLE_PRIORITY = {"train": 0, "validation": 1, "test": 2}


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def repair_dc_offsets(manifest_dir: Path, roles: list[str], threshold: float = 0.10) -> dict:
    """Remove severe DC bias without deleting otherwise valid speech."""
    repaired_by_role: Counter[str] = Counter()
    repaired_ids: list[str] = []
    for role in roles:
        manifest = manifest_dir / f"asr--{role}-local.jsonl"
        rows = read_jsonl(manifest)
        for row in rows:
            qc = row.get("audio_qc") or {}
            if float(qc.get("dc_offset") or 0.0) <= threshold:
                continue
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = ROOT / audio_path
            audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            original_dc = float(abs(np.mean(mono)))
            repaired = mono - float(np.mean(mono))
            repaired = np.clip(repaired, -1.0, 1.0).astype(np.float32)
            temp = audio_path.with_suffix(".dc-repair.tmp.flac")
            sf.write(temp, repaired, sample_rate, format="FLAC", subtype="PCM_16")
            os.replace(temp, audio_path)

            qc["original_dc_offset"] = round(original_dc, 7)
            qc["dc_offset"] = round(float(abs(np.mean(repaired))), 7)
            qc["peak"] = round(float(np.max(np.abs(repaired))), 7)
            qc["rms"] = round(float(np.sqrt(np.mean(np.square(repaired)))), 7)
            qc["clipped_fraction"] = round(float(np.mean(np.abs(repaired) >= 0.999)), 7)
            qc["warnings"] = [value for value in qc.get("warnings", []) if value != "high_dc_offset"]
            qc["repairs"] = sorted(set(qc.get("repairs", [])) | {"dc_offset_removed"})
            row["audio_qc"] = qc
            repaired_by_role[role] += 1
            repaired_ids.append(str(row["id"]))

        with manifest.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "threshold": threshold,
        "repaired": len(repaired_ids),
        "repaired_by_role": dict(sorted(repaired_by_role.items())),
        "ids": sorted(repaired_ids),
    }


def dedupe_exact_audio(manifest_dir: Path, roles: list[str], workers: int) -> dict:
    """Remove bit-identical FLAC leakage, preserving the highest evaluation role."""
    indexed: list[tuple[str, dict, Path]] = []
    for role in roles:
        manifest = manifest_dir / f"asr--{role}-local.jsonl"
        for row in read_jsonl(manifest):
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = ROOT / audio_path
            indexed.append((role, row, audio_path))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = list(pool.map(lambda item: sha256_file(item[2]), indexed))

    by_hash: dict[str, list[tuple[str, dict]]] = {}
    for (manifest_role, row, _), audio_hash in zip(indexed, hashes):
        row["audio_sha256"] = audio_hash
        by_hash.setdefault(audio_hash, []).append((manifest_role, row))

    kept_ids: set[str] = set()
    removals: list[dict] = []
    duplicate_groups = 0
    cross_role_groups = 0
    for audio_hash, group in by_hash.items():
        if len(group) == 1:
            kept_ids.add(group[0][1]["id"])
            continue
        duplicate_groups += 1
        if len({role for role, _ in group}) > 1:
            cross_role_groups += 1
        # Test beats validation beats train. Within the same role, retain an
        # official hard example first, then the row with fewer prior flags,
        # then the stable lowest ID. Transcript differences remain logged.
        winner_role, winner = min(
            group,
            key=lambda item: (
                -ROLE_PRIORITY[item[0]],
                0 if item[1].get("source_split") == "hard" else 1,
                len(item[1].get("quality_flags") or []),
                str(item[1]["id"]),
            ),
        )
        kept_ids.add(winner["id"])
        for role, row in group:
            if row["id"] == winner["id"]:
                continue
            removals.append(
                {
                    "id": row["id"],
                    "source": row.get("source"),
                    "role": role,
                    "text": row.get("text"),
                    "audio_path": row.get("audio_path"),
                    "audio_sha256": audio_hash,
                    "reason": "exact_audio_duplicate",
                    "kept_id": winner["id"],
                    "kept_role": winner_role,
                    "kept_text": winner.get("text"),
                }
            )

    final_counts: dict[str, int] = {}
    for role in roles:
        manifest = manifest_dir / f"asr--{role}-local.jsonl"
        rows = [row for manifest_role, row, _ in indexed if manifest_role == role and row["id"] in kept_ids]
        rows.sort(key=lambda row: row["id"])
        with manifest.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        final_counts[role] = len(rows)

    rejection_path = manifest_dir / "asr--exact-audio-rejections.jsonl"
    with rejection_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(removals, key=lambda item: (item["role"], item["id"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "duplicate_groups": duplicate_groups,
        "cross_role_groups": cross_role_groups,
        "removed": len(removals),
        "removed_by_role": dict(sorted(Counter(row["role"] for row in removals).items())),
        "final_counts": final_counts,
        "rejection_file": portable_path(rejection_path),
    }


def process_one(record: dict, output_root: Path, min_rms: float) -> tuple[dict | None, dict]:
    source_path = Path(record["audio_path"])
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    reasons: list[str] = []
    warnings: list[str] = []
    try:
        audio, sample_rate = sf.read(source_path, dtype="float32", always_2d=True)
    except Exception as exc:
        return None, {"id": record.get("id"), "source": record.get("source"), "reasons": [f"decode_error:{exc}"]}
    if not len(audio) or sample_rate <= 0 or not np.isfinite(audio).all():
        return None, {"id": record.get("id"), "source": record.get("source"), "reasons": ["invalid_decoded_audio"]}

    mono = audio.mean(axis=1)
    duration_s = len(mono) / sample_rate
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(np.square(mono))))
    dc_offset = float(abs(np.mean(mono)))
    clipped_fraction = float(np.mean(np.abs(mono) >= 0.999))
    try:
        metadata_duration = float(record.get("duration_s"))
    except (TypeError, ValueError):
        metadata_duration = 0.0
    mismatch_s = abs(duration_s - metadata_duration) if metadata_duration > 0 else 0.0
    mismatch_ratio = mismatch_s / metadata_duration if metadata_duration > 0 else 0.0
    decoded_cps = len(str(record.get("text") or "")) / max(duration_s, 1e-9)

    if duration_s < 0.5:
        reasons.append("decoded_duration_too_short")
    if rms < min_rms:
        reasons.append("near_silence")
    if clipped_fraction > 0.20:
        reasons.append("severe_clipping")
    if decoded_cps > 35.0:
        reasons.append("decoded_high_text_audio_ratio")
    # ViMedCSS stores integer segment boundaries and commonly differs from the
    # encoded clip by exactly one second. Treat that systematic rounding as a
    # warning; larger mismatches remain fatal.
    if metadata_duration > 0 and mismatch_s > 1.1 and mismatch_ratio > 0.20:
        reasons.append("duration_metadata_mismatch")
    elif metadata_duration > 0 and mismatch_s > 0.5:
        warnings.append("duration_metadata_rounding")
    if dc_offset > 0.10:
        warnings.append("high_dc_offset")
    if peak < 0.01:
        warnings.append("very_low_peak")

    stats = {
        "id": record.get("id"),
        "source": record.get("source"),
        "source_split": record.get("source_split"),
        "duration_s": round(duration_s, 4),
        "metadata_duration_s": metadata_duration,
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "peak": round(peak, 7),
        "rms": round(rms, 7),
        "dc_offset": round(dc_offset, 7),
        "clipped_fraction": round(clipped_fraction, 7),
        "decoded_chars_per_second": round(decoded_cps, 4),
        "reasons": reasons,
        "warnings": warnings,
        "repairs": [],
    }
    if reasons:
        return None, stats

    if dc_offset > 0.10:
        mono = mono - float(np.mean(mono))
        mono = np.clip(mono, -1.0, 1.0).astype(np.float32)
        stats["original_dc_offset"] = round(dc_offset, 7)
        stats["dc_offset"] = round(float(abs(np.mean(mono))), 7)
        stats["peak"] = round(float(np.max(np.abs(mono))), 7)
        stats["rms"] = round(float(np.sqrt(np.mean(np.square(mono)))), 7)
        stats["clipped_fraction"] = round(float(np.mean(np.abs(mono) >= 0.999)), 7)
        stats["warnings"] = [value for value in warnings if value != "high_dc_offset"]
        stats["repairs"] = ["dc_offset_removed"]

    if sample_rate != 16000:
        common = math.gcd(int(sample_rate), 16000)
        mono = resample_poly(mono, 16000 // common, int(sample_rate) // common).astype(np.float32)
    destination = output_root / record["source"] / record["source_split"] / f"{record['id']}.flac"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp.flac")
    sf.write(temp, np.clip(mono, -1.0, 1.0), 16000, format="FLAC", subtype="PCM_16")
    os.replace(temp, destination)
    item = dict(record)
    item["original_audio_path"] = portable_path(source_path)
    item["audio_path"] = portable_path(destination)
    item["audio_qc"] = {key: value for key, value in stats.items() if key not in {"id", "source", "source_split", "reasons"}}
    return item, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data" / "processed" / "manifests")
    parser.add_argument("--output-audio-dir", type=Path, default=ROOT / "data" / "processed" / "audio_16k")
    parser.add_argument("--roles", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-rms", type=float, default=0.003)
    parser.add_argument("--max-drop-rate", type=float, default=0.02)
    parser.add_argument("--report", type=Path, default=ROOT / "data" / "reports" / "audio_qc.json")
    parser.add_argument(
        "--dedupe-only",
        action="store_true",
        help="Apply exact-FLAC dedup to already signal-QC'd local manifests without re-decoding audio.",
    )
    parser.add_argument(
        "--repair-dc-only",
        action="store_true",
        help="Repair retained high-DC files, refresh exact-audio hashes, and preserve prior rejection evidence.",
    )
    args = parser.parse_args()

    if args.repair_dc_only:
        if not args.report.is_file():
            raise FileNotFoundError(f"Existing signal-QC report required: {args.report}")
        report = json.loads(args.report.read_text(encoding="utf-8"))
        repair = repair_dc_offsets(args.manifest_dir, args.roles)
        previous_dedup = report.get("exact_audio_dedup")
        dedup = dedupe_exact_audio(args.manifest_dir, args.roles, args.workers)
        if dedup["removed"] == 0 and previous_dedup and previous_dedup.get("removed", 0) > 0:
            dedup = dict(previous_dedup)
            dedup["final_counts"] = {
                role: len(read_jsonl(args.manifest_dir / f"asr--{role}-local.jsonl"))
                for role in args.roles
            }
        report["dc_offset_repair"] = repair
        report["exact_audio_dedup"] = dedup
        for role, count in repair["repaired_by_role"].items():
            warnings = report["roles"][role].get("warnings", {})
            remaining = max(0, int(warnings.get("high_dc_offset", 0)) - count)
            if remaining:
                warnings["high_dc_offset"] = remaining
            else:
                warnings.pop("high_dc_offset", None)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"dc_offset_repair": repair, "exact_audio_dedup": dedup}, ensure_ascii=False, indent=2))
        return 0

    if args.dedupe_only:
        if not args.report.is_file():
            raise FileNotFoundError(f"Existing signal-QC report required: {args.report}")
        report = json.loads(args.report.read_text(encoding="utf-8"))
        previous_dedup = report.get("exact_audio_dedup")
        dedup = dedupe_exact_audio(args.manifest_dir, args.roles, args.workers)
        # Do not erase the original leakage evidence on an idempotent rerun
        # over manifests that have already been deduplicated.
        if dedup["removed"] == 0 and previous_dedup and previous_dedup.get("removed", 0) > 0:
            dedup = dict(previous_dedup)
            dedup["final_counts"] = {
                role: len(read_jsonl(args.manifest_dir / f"asr--{role}-local.jsonl"))
                for role in args.roles
            }
        report["exact_audio_dedup"] = dedup
        report["status"] = "pass"
        for role in args.roles:
            removed = dedup["removed_by_role"].get(role, 0)
            role_report = report["roles"][role]
            # A repeated --dedupe-only run is idempotent: recompute the total
            # from the original input and current final kept count.
            role_report["kept"] = dedup["final_counts"][role]
            role_report["rejected"] = role_report["input"] - role_report["kept"]
            if removed:
                role_report["reasons"]["exact_audio_duplicate"] = removed
            role_report["drop_rate"] = round(
                role_report["rejected"] / max(1, role_report["input"]), 6
            )
            if role_report["drop_rate"] > args.max_drop_rate:
                report["status"] = "review_required"
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "pass" else 2

    report = {"thresholds": {"min_rms": args.min_rms, "max_drop_rate": args.max_drop_rate}, "roles": {}, "status": "pass"}
    for role in args.roles:
        input_path = args.manifest_dir / f"asr--{role}-local.jsonl"
        records = read_jsonl(input_path)
        kept: list[dict] = []
        rejected: list[dict] = []
        warning_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, record, args.output_audio_dir, args.min_rms): record for record in records}
            for index, future in enumerate(as_completed(futures), start=1):
                item, stats = future.result()
                warning_counts.update(stats.get("warnings", []))
                if item is None:
                    rejected.append(stats)
                    reason_counts.update(stats.get("reasons", []))
                else:
                    kept.append(item)
                if index % 500 == 0:
                    print(f"[{role}] {index}/{len(records)} decoded; rejected={len(rejected)}", flush=True)
        kept.sort(key=lambda row: row["id"])
        with input_path.open("w", encoding="utf-8") as handle:
            for item in kept:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        rejected_path = args.manifest_dir / f"asr--{role}-audio-rejections.jsonl"
        rejected_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in rejected) + ("\n" if rejected else ""),
            encoding="utf-8",
        )
        drop_rate = len(rejected) / max(1, len(records))
        report["roles"][role] = {
            "input": len(records),
            "kept": len(kept),
            "rejected": len(rejected),
            "drop_rate": round(drop_rate, 6),
            "reasons": dict(reason_counts),
            "warnings": dict(warning_counts),
            "rejection_file": str(rejected_path),
        }
        if drop_rate > args.max_drop_rate:
            report["status"] = "review_required"
    dedup = dedupe_exact_audio(args.manifest_dir, args.roles, args.workers)
    report["exact_audio_dedup"] = dedup
    for role in args.roles:
        removed = dedup["removed_by_role"].get(role, 0)
        report["roles"][role]["kept"] = dedup["final_counts"][role]
        report["roles"][role]["rejected"] += removed
        if removed:
            report["roles"][role]["reasons"]["exact_audio_duplicate"] = removed
        report["roles"][role]["drop_rate"] = round(
            report["roles"][role]["rejected"] / max(1, report["roles"][role]["input"]), 6
        )
        if report["roles"][role]["drop_rate"] > args.max_drop_rate:
            report["status"] = "review_required"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

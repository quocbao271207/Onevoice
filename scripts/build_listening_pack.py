"""Download the EDA listening queue and create a fillable review CSV."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly

try:
    from scripts.materialize_audio import destination_path, download_one, refresh_audio_urls
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from materialize_audio import destination_path, download_one, refresh_audio_urls


ROOT = Path(__file__).resolve().parents[1]


def make_review_copy(original: Path, target: Path) -> dict[str, float]:
    """Create a broadly playable mono/16 kHz PCM16 copy with bounded peak gain."""
    audio, sample_rate = sf.read(original, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    duration_s = len(mono) / sample_rate
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
    if sample_rate != 16000:
        common = math.gcd(int(sample_rate), 16000)
        mono = resample_poly(mono, 16000 // common, int(sample_rate) // common).astype(np.float32)
    gain = min(20.0, 0.9 / peak) if peak > 1e-5 else 1.0
    mono = np.clip(mono * gain, -1.0, 1.0)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, mono, 16000, subtype="PCM_16")
    return {
        "actual_duration_s": round(duration_s, 3),
        "original_peak": round(peak, 6),
        "original_rms": round(rms, 6),
        "playback_gain_db": round(20 * math.log10(gain), 2) if gain > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eda-dir", type=Path, default=ROOT / "data" / "reports" / "eda")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "review" / "listening_pack")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "datasets.yaml")
    parser.add_argument("--per-source", type=int, default=12)
    parser.add_argument("--source", action="append", help="Only build queues for this source key (repeatable).")
    parser.add_argument("--min-duration-s", type=float, default=2.0)
    parser.add_argument("--min-rms", type=float, default=0.003)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset_specs = config.get("datasets")
    if not isinstance(dataset_specs, dict):
        raise ValueError("Dataset config must contain a datasets mapping")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_rows = []
    exclusions = []
    selected_sources = set(args.source or [])
    for queue_path in sorted((args.eda_dir / "listening").glob("*.jsonl")):
        if selected_sources and queue_path.stem not in selected_sources:
            continue
        records = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines() if line]
        records = refresh_audio_urls(records, dataset_specs)
        accepted = 0
        for record in records:
            if accepted >= args.per_source:
                break
            local = download_one(record, args.output_dir / "audio")
            original_path = Path(local["audio_path"])
            review_path = destination_path(record, args.output_dir / "playback")
            audio_stats = make_review_copy(original_path, review_path)
            rejection_reasons = []
            if audio_stats["actual_duration_s"] < args.min_duration_s:
                rejection_reasons.append(f"duration<{args.min_duration_s}s")
            if audio_stats["original_rms"] < args.min_rms:
                rejection_reasons.append(f"rms<{args.min_rms}")
            if rejection_reasons:
                exclusions.append(
                    {
                        "id": record["id"],
                        "source": record["source"],
                        "split": record["source_split"],
                        "reasons": rejection_reasons,
                        **audio_stats,
                    }
                )
                review_path.unlink(missing_ok=True)
                original_path.unlink(missing_ok=True)
                continue
            review_rows.append(
                {
                    "id": record["id"],
                    "source": record["source"],
                    "split": record["source_split"],
                    "review_reason": record.get("review_reason", ""),
                    "audio_path": str(review_path.resolve()),
                    "original_audio_path": str(original_path.resolve()),
                    "metadata_duration_s": record.get("duration_s", ""),
                    **audio_stats,
                    "transcript": record["text"],
                    "audio_ok": "",
                    "transcript_exact": "",
                    "language": "",
                    "accent_if_confident": "",
                    "pii": "",
                    "verdict_keep_drop": "",
                    "note": "",
                }
            )
            accepted += 1
        if accepted < args.per_source:
            raise RuntimeError(f"{queue_path.stem}: only {accepted}/{args.per_source} clips passed signal QC")
    csv_path = args.output_dir / "review.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]) if review_rows else ["id"])
        writer.writeheader()
        writer.writerows(review_rows)

    cards = []
    output_root = args.output_dir.resolve()
    for index, row in enumerate(review_rows, start=1):
        relative_audio = Path(row["audio_path"]).resolve().relative_to(output_root).as_posix()
        duration = float(row["actual_duration_s"])
        cards.append(
            f"""<article>
              <h2>{index}. {html.escape(row['source'])} / {html.escape(row['split'])}</h2>
              <audio controls preload="metadata" src="{html.escape(relative_audio)}"></audio>
              <p><b>{duration:.2f}s</b> · {html.escape(row['review_reason'])}</p>
              <p class="transcript">{html.escape(row['transcript'])}</p>
              <p class="id">ID: {html.escape(row['id'])}</p>
            </article>"""
        )
    page = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OneVoice listening review</title>
<style>
body{{font:16px system-ui;max-width:1000px;margin:32px auto;padding:0 18px;background:#f5f7fb;color:#172033}}
header,article{{background:white;border:1px solid #dce2ec;border-radius:12px;padding:18px;margin:14px 0}}
audio{{width:100%}} .transcript{{font-size:1.08rem;line-height:1.55}} .id{{color:#667085;font-size:.85rem}}
.warn{{color:#b42318}} code{{background:#eef2f6;padding:2px 5px;border-radius:5px}}
</style></head><body>
<header><h1>OneVoice — {len(review_rows)} mẫu nghe</h1>
<p>Audio trên trang là bản mono 16 kHz PCM16 đã chuẩn hóa âm lượng để dễ phát. File gốc vẫn được giữ trong <code>audio/</code>.</p>
<p>Nghe theo thứ tự rồi ghi verdict vào <a href="review.csv">review.csv</a>. Mẫu dưới {args.min_duration_s:g} giây hoặc gần im lặng đã bị loại tự động.</p></header>
{''.join(cards)}
</body></html>"""
    html_path = args.output_dir / "review.html"
    html_path.write_text(page, encoding="utf-8")
    exclusion_path = args.output_dir / "signal_exclusions.jsonl"
    exclusion_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in exclusions) + ("\n" if exclusions else ""),
        encoding="utf-8",
    )
    print(f"Listening pack: {len(review_rows)} clips -> {csv_path}")
    print(f"Signal exclusions: {len(exclusions)} -> {exclusion_path}")
    print(f"Browser player: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

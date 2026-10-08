"""Run the complete base-model cascade on one real VI and one real EN clip.

This is an operational integration test:
audio -> frontend -> ASR -> MT -> safety guard -> TTS audio.
ASR has a transcript reference, but the ASR datasets do not contain a target
translation reference. Translation quality therefore remains covered by the
separate locked MedEV benchmark rather than being invented here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from jiwer import cer, wer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.asr_engine import BASE_MODEL_REVISIONS  # noqa: E402
from src.pipeline.mt_engine import NLLB_BASE_REVISION  # noqa: E402
from src.pipeline.orchestrator import MediVoicePipeline  # noqa: E402
from src.utils.text_normalization import normalize_for_wer  # noqa: E402


DEFAULT_IDS = {
    "vi": "a0ceb4047f2bc7ed73d6ce9b",
    "en": "71dc5445c25c808449b702f1",
}


def load_cases(manifest: Path) -> dict[str, dict]:
    wanted = set(DEFAULT_IDS.values())
    cases: dict[str, dict] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("id") not in wanted:
                continue
            language = "en" if row.get("language") == "en" else "vi"
            cases[language] = row
    missing = set(DEFAULT_IDS) - set(cases)
    if missing:
        raise RuntimeError(f"Missing locked E2E cases: {sorted(missing)}")
    return cases


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "processed" / "manifests" / "asr--test-local.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "reports" / "smoke" / "base_e2e",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.manifest)
    pipeline = MediVoicePipeline(config_path=str(ROOT / "configs" / "pipeline_config.yaml"))

    # Force the exact pinned base models so this artifact cannot accidentally
    # mix a later fine-tuned checkpoint into the baseline.
    pipeline.asr_engine.vi_model_path = "vinai/PhoWhisper-small"
    pipeline.asr_engine.en_model_path = "distil-whisper/distil-small.en"
    pipeline.asr_engine.allow_base_fallback = False
    pipeline.mt_engine.model_path = "facebook/nllb-200-distilled-600M"
    pipeline.mt_engine.allow_base_fallback = False
    pipeline.load()

    expected_models = {
        "asr_vi": {
            "model": "vinai/PhoWhisper-small",
            "revision": BASE_MODEL_REVISIONS["vinai/PhoWhisper-small"],
        },
        "asr_en": {
            "model": "distil-whisper/distil-small.en",
            "revision": BASE_MODEL_REVISIONS["distil-whisper/distil-small.en"],
        },
        "mt": {
            "model": "facebook/nllb-200-distilled-600M",
            "revision": NLLB_BASE_REVISION,
        },
    }
    if pipeline.asr_engine.loaded_model_paths != {
        "vi": expected_models["asr_vi"]["model"],
        "en": expected_models["asr_en"]["model"],
    }:
        raise RuntimeError(f"Unexpected ASR models: {pipeline.asr_engine.loaded_model_paths}")
    if pipeline.mt_engine.loaded_model_path != expected_models["mt"]["model"]:
        raise RuntimeError(f"Unexpected MT model: {pipeline.mt_engine.loaded_model_path}")

    results = []
    for language in ("vi", "en"):
        row = cases[language]
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = ROOT / audio_path
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        result = pipeline.translate_speech(
            np.asarray(audio, dtype=np.float32),
            source_lang=language,
            sample_rate=sample_rate,
            skip_tts=False,
        )
        output_path = args.output_dir / f"{language}_to_{result.target_language}.wav"
        if result.output_audio is not None:
            sf.write(output_path, result.output_audio, result.output_sample_rate)
        reference = normalize_for_wer(row["text"])
        hypothesis = normalize_for_wer(result.asr_text)
        output_duration = (
            len(result.output_audio) / result.output_sample_rate if result.output_audio is not None else 0.0
        )
        results.append(
            {
                "id": row["id"],
                "source": row["source"],
                "direction": f"{language}_to_{result.target_language}",
                "input_audio": str(audio_path.relative_to(ROOT).as_posix()),
                "input_duration_s": round(len(audio) / sample_rate, 3),
                "reference_transcript": row["text"],
                "asr_transcript": result.asr_text,
                "asr_wer": round(float(wer(reference, hypothesis)), 6),
                "asr_cer": round(float(cer(reference, hypothesis)), 6),
                "translation": result.translated_text,
                "translation_reference_available": False,
                "safety_passed": result.safety_passed,
                "safety_issues": result.safety_issues,
                "requires_confirmation": result.requires_confirmation,
                "degraded_mode": result.degraded_mode,
                "degradation_code": result.degradation_code,
                "from_cache": result.from_cache,
                "output_audio": str(output_path.relative_to(ROOT).as_posix()),
                "output_audio_exists": output_path.is_file(),
                "output_audio_duration_s": round(output_duration, 3),
                "latency_ms": {
                    **{key: round(float(value), 2) for key, value in result.latency_breakdown.items()},
                    "total": round(result.total_latency_ms, 2),
                },
                "overall_rtf": round(result.overall_rtf, 4),
            }
        )

    totals = np.asarray([row["latency_ms"]["total"] for row in results], dtype=np.float64)
    operational_pass = all(
        row["asr_transcript"].strip()
        and row["translation"].strip()
        and row["safety_passed"]
        and row["output_audio_exists"]
        and row["output_audio_duration_s"] > 0
        for row in results
    )
    report = {
        "test_type": "complete_base_cascade_operational_smoke",
        "flow": "presegmented_real_audio -> denoise_stage -> base_ASR -> base_MT -> safety -> Piper_TTS",
        "scope_limitations": [
            "The input is a presegmented locked audio file, so microphone capture and VAD segmentation are not exercised.",
            "Only two samples are used; latency quantiles are descriptive smoke values, not a benchmark.",
            "The ASR datasets have no paired target translation, so end-to-end translation quality is not scored here.",
        ],
        "frontend": {
            "vad_loaded": bool(pipeline.audio_frontend.vad._is_loaded),
            "noise_suppression_loaded": bool(pipeline.audio_frontend.denoiser._is_loaded),
            "note": "translate_speech receives presegmented clips; VAD is outside this call.",
        },
        "models": expected_models,
        "translation_quality_note": (
            "These ASR clips have no paired translation reference; use mt_base.json for BLEU/chrF."
        ),
        "samples": len(results),
        "latency_ms": {
            "p50": round(float(np.quantile(totals, 0.50)), 2),
            "p95": round(float(np.quantile(totals, 0.95)), 2),
        },
        "results": results,
        "status": "pass" if operational_pass else "fail",
    }
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if operational_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

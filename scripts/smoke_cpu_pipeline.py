"""Run deterministic CPU smoke checks for base ASR, MT, TTS and safety guard."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.asr_engine import ASREngine, BASE_MODEL_REVISIONS  # noqa: E402
from src.pipeline.mt_engine import MTEngine, NLLB_BASE_REVISION  # noqa: E402
from src.pipeline.safety_guard import validate_translation  # noqa: E402
from src.pipeline.tts_engine import TTSEngine  # noqa: E402


def build_base_asr() -> ASREngine:
    return ASREngine(
        vi_model_path="vinai/PhoWhisper-small",
        en_model_path="distil-whisper/distil-small.en",
        device="cpu",
        allow_base_fallback=False,
    )


def build_base_mt() -> MTEngine:
    return MTEngine(
        model_path="facebook/nllb-200-distilled-600M",
        device="cpu",
        allow_base_fallback=False,
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "reports" / "smoke")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"device": "cpu", "warning": "Base-model smoke only; not a quality benchmark."}

    started = time.perf_counter()
    tts = TTSEngine()
    tts.load()
    report["tts_load_ms"] = round((time.perf_counter() - started) * 1000, 2)
    prompts = {
        "vi": "Bệnh nhân không được tiêm quá không phẩy năm miligam epinephrine.",
        "en": "The patient must not receive more than zero point five milligrams of epinephrine.",
    }
    tts_audio = {}
    report["tts"] = {}
    for language, prompt in prompts.items():
        result = tts.synthesize(prompt, language)
        path = args.output_dir / f"tts_{language}.wav"
        sf.write(path, result.audio, result.sample_rate)
        tts_audio[language] = (result.audio, result.sample_rate)
        report["tts"][language] = {
            "prompt": prompt,
            "audio_path": str(path),
            "duration_s": round(result.duration_s, 3),
            "latency_ms": round(result.latency_ms, 2),
            "rtf": round(result.rtf, 4),
        }

    started = time.perf_counter()
    asr = build_base_asr()
    asr.load()
    expected_asr_models = {
        "vi": "vinai/PhoWhisper-small",
        "en": "distil-whisper/distil-small.en",
    }
    if asr.loaded_model_paths != expected_asr_models:
        raise RuntimeError(f"Unexpected ASR models: {asr.loaded_model_paths}")
    report["asr_load_ms"] = round((time.perf_counter() - started) * 1000, 2)
    report["asr_models"] = {
        language: {
            "model": model,
            "revision": BASE_MODEL_REVISIONS[model],
        }
        for language, model in expected_asr_models.items()
    }
    report["asr"] = {}
    for language, (audio, sample_rate) in tts_audio.items():
        result = asr.transcribe(audio, language=language, sample_rate=sample_rate)
        report["asr"][language] = {
            "reference": prompts[language],
            "hypothesis": result.text,
            "confidence_uncalibrated": round(result.confidence, 4),
            "latency_ms": round(result.latency_ms, 2),
        }

    started = time.perf_counter()
    mt = build_base_mt()
    mt.load()
    if mt.loaded_model_path != "facebook/nllb-200-distilled-600M":
        raise RuntimeError(f"Unexpected MT model: {mt.loaded_model_path}")
    report["mt_load_ms"] = round((time.perf_counter() - started) * 1000, 2)
    report["mt_model"] = {
        "model": mt.loaded_model_path,
        "revision": NLLB_BASE_REVISION,
    }
    mt_inputs = {
        "vi": "Bệnh nhân không được tiêm quá 0,5 mg epinephrine.",
        "en": "The patient must not receive more than 0.5 mg of epinephrine.",
    }
    report["mt"] = {}
    for source_language, text in mt_inputs.items():
        result = mt.translate(text, source_language)
        safety = validate_translation(text, result.translated_text, result.source_lang, result.target_lang)
        report["mt"][f"{result.source_lang}_to_{result.target_lang}"] = {
            "source": text,
            "translation": result.translated_text,
            "latency_ms": round(result.latency_ms, 2),
            "first_token_ms": result.first_token_ms,
            "safety_pass": safety.safe,
            "safety_issues": safety.issues,
        }

    output_path = args.output_dir / "cpu_smoke.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

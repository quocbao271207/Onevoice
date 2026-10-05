"""Run reproducible ASR or MT benchmarks on locked manifests.

The default remains a CPU baseline.  Candidate LoRA adapters can be evaluated
on CUDA with the same scoring path so a fine-tuned checkpoint is never judged
with a different metric implementation from its locked baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.candidate_evidence import adapter_identity, canonical_sha256, sha256  # noqa: E402
from scripts.evaluate_benchmarks import score_asr, score_mt  # noqa: E402
from src.training.mt_model_adapter import (  # noqa: E402
    SUPPORTED_MT_FAMILIES,
    configure_tokenizer,
    direction_fields,
    forced_bos_token_id,
    language_codes,
    requested_directions,
)


MODEL_REVISIONS = {
    "vinai/PhoWhisper-small": "a86b604c346caf7148c37512eafe783a16420adb",
    "distil-whisper/distil-small.en": "9e4a67ca4569c30be43a3fe7fba1621e504f0093",
    "facebook/nllb-200-distilled-600M": "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
    "facebook/m2m100_418M": "55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636",
    "vinai/vinai-translate-en2vi-v2": "82f8c91bd22e82085186b45a8a76373f5a79f667",
    "vinai/vinai-translate-vi2en-v2": "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138",
    "openai/whisper-small": "973afd24965f72e36ca33b3055d56a652f456b4d",
    "vinai/PhoWhisper-base": "7ebdb9e88f5cc5271fb88f4d642c82ff9388650e",
}

PREDICTION_CHECKPOINT_SCHEMA_VERSION = 1


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_rank(seed: int, row: dict[str, Any]) -> str:
    value = f"{seed}\x1f{row.get('id', '')}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def source_balanced_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """Round-robin sources so a large corpus cannot consume a small baseline."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("source") or "unknown")].append(row)
    for values in groups.values():
        values.sort(key=lambda row: stable_rank(seed, row))
    selected = []
    offset = 0
    names = sorted(groups)
    while len(selected) < min(size, len(rows)):
        added = False
        for name in names:
            if offset < len(groups[name]) and len(selected) < size:
                selected.append(groups[name][offset])
                added = True
        if not added:
            break
        offset += 1
    return selected


def batches(rows: list[Any], size: int):
    for offset in range(0, len(rows), size):
        yield rows[offset : offset + size]


def write_predictions_checkpoint(path: Path, predictions: list[dict[str, Any]]) -> None:
    """Atomically preserve inference output before potentially long scoring."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def prediction_provenance_path(prediction_path: Path) -> Path:
    return prediction_path.with_suffix(prediction_path.suffix + ".provenance.json")


def prediction_checkpoint_specification(args: argparse.Namespace) -> dict[str, Any]:
    """Bind resumable predictions to every input that can affect inference."""
    manifest = args.manifest.resolve()
    return {
        "task": args.task,
        "model": args.model,
        "model_revision": args.model_revision,
        "adapter": adapter_identity(args.adapter),
        "manifest": {
            "path": str(manifest),
            "bytes": manifest.stat().st_size,
            "sha256": sha256(manifest),
        },
        "samples": args.samples,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_beams": args.num_beams,
        "requested_device": args.device,
        "precision": args.precision,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "language": args.language if args.task == "asr" else None,
        "asr_prompt": args.asr_prompt if args.task == "asr" else None,
        "mt_model_family": args.mt_model_family if args.task == "mt" else None,
        "mt_direction": args.mt_direction if args.task == "mt" else None,
    }


def write_prediction_checkpoint(
    prediction_path: Path,
    predictions: list[dict[str, Any]],
    specification: dict[str, Any],
    decoding: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Persist predictions plus immutable provenance before CPU scoring starts."""
    write_predictions_checkpoint(prediction_path, predictions)
    provenance = {
        "schema_version": PREDICTION_CHECKPOINT_SCHEMA_VERSION,
        "specification": specification,
        "specification_sha256": canonical_sha256(specification),
        "predictions": {
            "path": str(prediction_path.resolve()),
            "bytes": prediction_path.stat().st_size,
            "sha256": sha256(prediction_path),
            "rows": len(predictions),
        },
        "decoding": decoding,
        "runtime": runtime,
    }
    provenance_path = prediction_provenance_path(prediction_path)
    _atomic_json(provenance_path, provenance)
    verified = load_verified_prediction_checkpoint(prediction_path, specification)
    if verified is None:  # pragma: no cover - the files were just written above
        raise RuntimeError(f"Prediction checkpoint disappeared: {prediction_path}")
    return provenance


def load_verified_prediction_checkpoint(
    prediction_path: Path,
    expected_specification: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Return an exact checkpoint or fail closed on incomplete/tampered evidence."""
    provenance_path = prediction_provenance_path(prediction_path)
    prediction_exists = prediction_path.is_file()
    provenance_exists = provenance_path.is_file()
    if not prediction_exists and not provenance_exists:
        return None
    if not prediction_exists or not provenance_exists:
        raise ValueError(f"Incomplete prediction checkpoint: {prediction_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_sha256 = canonical_sha256(expected_specification)
    if (
        provenance.get("schema_version") != PREDICTION_CHECKPOINT_SCHEMA_VERSION
        or provenance.get("specification") != expected_specification
        or provenance.get("specification_sha256") != expected_sha256
    ):
        raise ValueError(f"Prediction checkpoint specification mismatch: {prediction_path}")
    prediction_record = provenance.get("predictions")
    if not isinstance(prediction_record, dict):
        raise ValueError(f"Invalid prediction checkpoint provenance: {provenance_path}")
    if (
        prediction_record.get("path") != str(prediction_path.resolve())
        or int(prediction_record.get("bytes", -1)) != prediction_path.stat().st_size
        or prediction_record.get("sha256") != sha256(prediction_path)
    ):
        raise ValueError(f"Prediction checkpoint checksum mismatch: {prediction_path}")
    predictions = read_jsonl(prediction_path)
    if (
        any(not isinstance(row, dict) for row in predictions)
        or int(prediction_record.get("rows", -1)) != len(predictions)
        or not isinstance(provenance.get("decoding"), dict)
        or not isinstance(provenance.get("runtime"), dict)
    ):
        raise ValueError(f"Prediction checkpoint structure mismatch: {prediction_path}")
    return predictions, provenance


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` lazily so importing this module stays CPU-test friendly."""
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def model_load_kwargs(device: str, precision: str) -> dict[str, Any]:
    if device == "cpu":
        if precision != "fp32":
            raise ValueError("CPU benchmarking supports only fp32 precision")
        return {}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmarking requested but CUDA is unavailable")
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]
    return {"torch_dtype": dtype}


def attach_adapter(model: Any, adapter: Path | None) -> Any:
    if adapter is None:
        return model
    if not adapter.is_dir():
        raise FileNotFoundError(f"Missing adapter directory: {adapter}")
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter))


def asr_prediction_slices(row: dict[str, Any]) -> dict[str, Any]:
    """Preserve selection/blind dimensions in saved ASR predictions and reports."""
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    dimensions = row.get("blind_dimensions") or row.get("selection_dimensions") or {}
    if not isinstance(dimensions, dict):
        dimensions = {}
    return {
        "accent": row.get("accent") or dimensions.get("accent_region"),
        "role": metadata.get("role") or dimensions.get("role"),
        "recording_condition": metadata.get("rec_condition")
        or dimensions.get("recording_condition"),
        "noise": dimensions.get("noise") if isinstance(dimensions.get("noise"), bool) else None,
        "noise_snr": row.get("noise_snr")
        if row.get("noise_snr") is not None
        else metadata.get("noise_snr"),
    }


def prepare_runtime(args: argparse.Namespace) -> str:
    device = resolve_device(args.device)
    if not 0.0 < args.gpu_memory_fraction <= 0.40:
        raise ValueError("gpu_memory_fraction must be in (0, 0.40]")
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    return device


def run_asr(
    args: argparse.Namespace,
    prediction_path: Path | None = None,
    checkpoint_specification: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    eligible = [row for row in read_jsonl(args.manifest) if str(row.get("language") or "").startswith(args.language)]
    rows = eligible if args.samples == 0 else source_balanced_sample(eligible, args.samples, args.seed)
    processor_source = str(args.adapter) if args.adapter else args.model
    processor_revision = None if args.adapter else args.model_revision
    if args.language == "vi":
        processor = WhisperProcessor.from_pretrained(
            processor_source, revision=processor_revision, language="vi", task="transcribe"
        )
    else:
        # English-only Distil-Whisper has no multilingual lang_to_id mapping.
        processor = WhisperProcessor.from_pretrained(processor_source, revision=processor_revision)
    device = prepare_runtime(args)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.model_revision,
        **model_load_kwargs(device, args.precision),
    )
    model = attach_adapter(model, args.adapter).to(device).eval()
    if args.language == "vi":
        model.generation_config.language = "vi"
        model.generation_config.task = "transcribe"
        model.generation_config.forced_decoder_ids = None
    predictions = []
    prompt_ids = None
    if args.asr_prompt:
        prompt_ids = processor.get_prompt_ids(args.asr_prompt, return_tensors="pt").to(device)
    generation_started = time.perf_counter()
    with torch.inference_mode():
        for chunk in batches(rows, args.batch_size):
            arrays = []
            for row in chunk:
                audio_path = Path(row["audio_path"])
                if not audio_path.is_absolute():
                    audio_path = ROOT / audio_path
                audio, rate = sf.read(audio_path, dtype="float32", always_2d=False)
                if rate != 16000:
                    raise ValueError(f"Expected 16 kHz QC audio, got {rate}: {audio_path}")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                arrays.append(np.asarray(audio, dtype=np.float32))
            features = processor.feature_extractor(
                arrays,
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
            )
            generation_kwargs = {
                "attention_mask": features.attention_mask.to(device),
                "max_new_tokens": 225,
                "num_beams": args.num_beams,
                "do_sample": False,
            }
            if prompt_ids is not None:
                generation_kwargs["prompt_ids"] = prompt_ids
            generated = model.generate(features.input_features.to(device), **generation_kwargs)
            hypotheses = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, hypothesis in zip(chunk, hypotheses):
                predictions.append(
                    {
                        "id": row["id"],
                        "reference": row["text"],
                        "hypothesis": hypothesis,
                        "source": row.get("source"),
                        **asr_prediction_slices(row),
                        "code_switch": row.get("language") == "vi-code-switch"
                        or "code_switch" in row.get("categories", []),
                        "categories": row.get("categories", []),
                        "safety_expectations": row.get("safety_expectations", {}),
                    }
                )
            print(f"[ASR] {len(predictions)}/{len(rows)}", flush=True)
    generation_seconds = time.perf_counter() - generation_started
    decoding = {
        "num_beams": args.num_beams,
        "prompt": args.asr_prompt,
        "generation_seconds": round(generation_seconds, 3),
        "samples_per_second": round(len(predictions) / generation_seconds, 6),
    }
    if prediction_path is not None:
        write_prediction_checkpoint(
            prediction_path,
            predictions,
            checkpoint_specification or prediction_checkpoint_specification(args),
            decoding,
            {"resolved_device": device},
        )
        print(f"[artifact] wrote predictions checkpoint: {prediction_path}", flush=True)
    report = score_asr(predictions)
    report["decoding"] = decoding
    return predictions, report


def run_mt(
    args: argparse.Namespace,
    prediction_path: Path | None = None,
    checkpoint_specification: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    manifest_rows = read_jsonl(args.manifest)
    pair_count = len(manifest_rows) if args.samples == 0 else max(1, args.samples // 2)
    rows = source_balanced_sample(manifest_rows, pair_count, args.seed)
    device = prepare_runtime(args)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        **model_load_kwargs(device, args.precision),
    )
    model = attach_adapter(model, args.adapter).to(device).eval()
    predictions = []
    directions = requested_directions(args.mt_direction, args.mt_model_family)
    generation_started = time.perf_counter()
    with torch.inference_mode():
        for direction in directions:
            source_field, target_field = direction_fields(direction)
            source_lang, target_lang = language_codes(args.mt_model_family, direction)
            tokenizer_source = str(args.adapter) if args.adapter else args.model
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                revision=None if args.adapter else args.model_revision,
            )
            configure_tokenizer(tokenizer, args.mt_model_family, direction)
            for chunk in batches(rows, args.batch_size):
                encoded = tokenizer(
                    [row[source_field] for row in chunk],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                generation_kwargs = {
                    "max_new_tokens": 256,
                    "num_beams": args.num_beams,
                    "do_sample": False,
                }
                bos_token_id = forced_bos_token_id(
                    tokenizer, args.mt_model_family, direction
                )
                if bos_token_id is not None:
                    generation_kwargs["forced_bos_token_id"] = bos_token_id
                generated = model.generate(**encoded, **generation_kwargs)
                hypotheses = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for row, hypothesis in zip(chunk, hypotheses):
                    predictions.append(
                        {
                            "id": row["id"],
                            "direction": direction,
                            "source": row[source_field],
                            "reference": row[target_field],
                            "hypothesis": hypothesis,
                            "quality_flags": row.get("quality_flags", []),
                            "categories": row.get("categories", []),
                            "terminology": row.get("terminology", {}),
                        }
                    )
                print(
                    f"[MT:{direction}] {len(predictions)}/{len(rows) * len(directions)}",
                    flush=True,
                )
    generation_seconds = time.perf_counter() - generation_started
    decoding = {
        "num_beams": args.num_beams,
        "generation_seconds": round(generation_seconds, 3),
        "samples_per_second": round(len(predictions) / generation_seconds, 6),
    }
    if prediction_path is not None:
        write_prediction_checkpoint(
            prediction_path,
            predictions,
            checkpoint_specification or prediction_checkpoint_specification(args),
            decoding,
            {"resolved_device": device},
        )
        print(f"[artifact] wrote predictions checkpoint: {prediction_path}", flush=True)
    report = score_mt(predictions)
    report["decoding"] = decoding
    return predictions, report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["asr", "mt"], required=True)
    parser.add_argument("--model")
    parser.add_argument("--model-revision")
    parser.add_argument("--mt-model-family", choices=SUPPORTED_MT_FAMILIES, default="nllb")
    parser.add_argument(
        "--mt-direction",
        choices=["joint", "en_to_vi", "vi_to_en"],
        default="joint",
    )
    parser.add_argument("--adapter", type=Path, help="Optional local PEFT/LoRA adapter directory.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.35)
    parser.add_argument("--language", choices=["vi", "en"], default="vi", help="ASR benchmark language.")
    parser.add_argument("--name", help="Artifact stem; defaults to task/language-specific base name.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-beams", type=int, help="Beam width; defaults to 1 for ASR and 4 for MT.")
    parser.add_argument("--asr-prompt", help="Optional Whisper prompt for domain vocabulary experiments.")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--resume-scoring",
        action="store_true",
        help="Reuse an exact checksum-verified prediction checkpoint and rerun CPU scoring only.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "reports" / "baselines")
    args = parser.parse_args()
    if args.num_beams is None:
        args.num_beams = 1 if args.task == "asr" else 4
    if args.num_beams < 1:
        parser.error("--num-beams must be at least 1")
    if args.samples < 0:
        parser.error("--samples must be non-negative; use 0 for the full manifest")
    if not 0.0 < args.gpu_memory_fraction <= 0.40:
        parser.error("--gpu-memory-fraction must be in (0, 0.40]")
    if args.device == "cpu" and args.precision != "fp32":
        parser.error("--device cpu requires --precision fp32")
    if args.task != "asr" and args.asr_prompt:
        parser.error("--asr-prompt is valid only with --task asr")
    if args.task == "mt":
        requested_directions(args.mt_direction, args.mt_model_family)
    if args.model is None:
        if args.task == "asr":
            args.model = "vinai/PhoWhisper-small" if args.language == "vi" else "distil-whisper/distil-small.en"
        else:
            args.model = "facebook/nllb-200-distilled-600M"
    if args.manifest is None:
        filename = "asr--test-local.jsonl" if args.task == "asr" else "mt--test.jsonl"
        args.manifest = ROOT / "data" / "processed" / "manifests" / filename
    if args.model_revision is None:
        args.model_revision = MODEL_REVISIONS.get(args.model)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    stem = args.name or (f"asr_{args.language}_base" if args.task == "asr" else "mt_base")
    prediction_path = args.output_dir / f"{stem}_predictions.jsonl"
    checkpoint_specification = prediction_checkpoint_specification(args)
    checkpoint = (
        load_verified_prediction_checkpoint(prediction_path, checkpoint_specification)
        if args.resume_scoring
        else None
    )
    if checkpoint is None:
        predictions, report = (
            run_asr(
                args,
                prediction_path=prediction_path,
                checkpoint_specification=checkpoint_specification,
            )
            if args.task == "asr"
            else run_mt(
                args,
                prediction_path=prediction_path,
                checkpoint_specification=checkpoint_specification,
            )
        )
        report_device = resolve_device(args.device)
        scoring_resumed = False
    else:
        predictions, provenance = checkpoint
        report = score_asr(predictions) if args.task == "asr" else score_mt(predictions)
        report["decoding"] = dict(provenance["decoding"])
        report["decoding"]["scoring_resumed_from_checkpoint"] = True
        report_device = str(provenance["runtime"]["resolved_device"])
        scoring_resumed = True
        print(
            f"[resume] verified predictions; skipped model loading and inference: {prediction_path}",
            flush=True,
        )
    report.update(
        {
            "model": args.model,
            "model_revision": args.model_revision,
            "mt_model_family": args.mt_model_family if args.task == "mt" else None,
            "mt_direction": args.mt_direction if args.task == "mt" else None,
            "adapter": str(args.adapter) if args.adapter else None,
            "device": report_device,
            "precision": args.precision,
            "gpu_memory_fraction": args.gpu_memory_fraction if report_device == "cuda" else None,
            "manifest": str(args.manifest),
            "seed": args.seed,
            "predictions": str(prediction_path),
            "prediction_provenance": str(prediction_provenance_path(prediction_path)),
            "scoring_resumed": scoring_resumed,
        }
    )
    report_path = args.output_dir / f"{stem}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

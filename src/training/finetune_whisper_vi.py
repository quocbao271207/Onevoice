"""Fine-tune multilingual PhoWhisper on audited Vietnamese medical speech."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.training.clinical_sampling import oversample_clinical_rows, prioritize_rows


logger = logging.getLogger(__name__)


@dataclass
class ASRTrainingConfig:
    base_model: str = "vinai/PhoWhisper-small"
    base_model_revision: str = "a86b604c346caf7148c37512eafe783a16420adb"
    train_manifest: str = "data/processed/manifests/asr--train-local.jsonl"
    validation_manifest: str = "data/processed/manifests/asr--validation-local.jsonl"
    output_dir: str = "models/asr/phowhisper-small-medical"
    learning_rate: float = 1e-5
    num_train_epochs: float = 5.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    generation_max_length: int = 225
    seed: int = 20260922
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    augmentation: bool = True
    speed_probability: float = 0.35
    gain_probability: float = 0.80
    noise_probability: float = 0.30
    noise_snr_db_min: float = 12.0
    noise_snr_db_max: float = 35.0
    reverb_probability: float = 0.12
    telephone_probability: float = 0.10
    codec_probability: float = 0.10
    early_stopping_patience: int = 2
    max_steps: int = -1
    resume_from_checkpoint: str | bool | None = None
    limit_train_examples: int = 0
    limit_validation_examples: int = 0
    method: str = "full"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    clinical_oversample_factor: int = 1
    cpu_smoke: bool = False
    local_files_only: bool = False
    gpu_memory_fraction: float = 0.35
    eval_steps: int = 0
    save_steps: int = 0
    logging_steps: int = 25


def augment_waveform(audio: np.ndarray, rng: np.random.Generator, config: ASRTrainingConfig) -> np.ndarray:
    """Apply conservative, label-preserving acoustic perturbations to train only."""
    value = np.asarray(audio, dtype=np.float32)
    if rng.random() < config.speed_probability and len(value) > 1600:
        speed = float(rng.uniform(0.90, 1.10))
        output_length = max(1, int(round(len(value) / speed)))
        positions = np.linspace(0, len(value) - 1, output_length)
        value = np.interp(positions, np.arange(len(value)), value).astype(np.float32)

    if rng.random() < config.gain_probability:
        gain_db = float(rng.uniform(-4.0, 4.0))
        value = value * (10.0 ** (gain_db / 20.0))

    if rng.random() < config.reverb_probability and len(value) > 3200:
        delay = int(rng.uniform(0.035, 0.12) * 16000)
        wet = float(rng.uniform(0.05, 0.18))
        reverbed = value.copy()
        reverbed[delay:] += wet * value[:-delay]
        if 2 * delay < len(value):
            reverbed[2 * delay :] += (wet * wet * 0.5) * value[: -2 * delay]
        value = reverbed

    if rng.random() < config.telephone_probability and len(value) > 1:
        spectrum = np.fft.rfft(value)
        frequencies = np.fft.rfftfreq(len(value), d=1.0 / 16000)
        spectrum[(frequencies < 300.0) | (frequencies > 3400.0)] = 0
        value = np.fft.irfft(spectrum, n=len(value)).astype(np.float32)

    if rng.random() < config.codec_probability:
        mu = 255.0
        bounded = np.clip(value, -1.0, 1.0)
        encoded = np.sign(bounded) * np.log1p(mu * np.abs(bounded)) / np.log1p(mu)
        quantized = np.round((encoded + 1.0) * 127.5) / 127.5 - 1.0
        value = np.sign(quantized) * np.expm1(np.abs(quantized) * np.log1p(mu)) / mu

    if rng.random() < config.noise_probability and len(value):
        signal_rms = float(np.sqrt(np.mean(np.square(value), dtype=np.float64)))
        if signal_rms > 1e-5:
            noise_kind = int(rng.integers(0, 3))
            if noise_kind == 0:
                noise = rng.normal(0.0, 1.0, len(value))
            elif noise_kind == 1:
                white = rng.normal(0.0, 1.0, len(value))
                spectrum = np.fft.rfft(white)
                frequencies = np.fft.rfftfreq(len(value), d=1.0 / 16000)
                spectrum /= np.sqrt(np.maximum(frequencies, 20.0))
                noise = np.fft.irfft(spectrum, n=len(value))
            else:
                time_axis = np.arange(len(value), dtype=np.float64) / 16000.0
                base = float(rng.choice([50.0, 60.0]))
                phase = float(rng.uniform(0.0, 2.0 * np.pi))
                noise = np.sin(2.0 * np.pi * base * time_axis + phase)
                noise += 0.35 * np.sin(2.0 * np.pi * 2.0 * base * time_axis + phase / 2.0)
            noise = np.asarray(noise, dtype=np.float32)
            noise_rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
            if noise_rms > 1e-8:
                snr_db = float(rng.uniform(config.noise_snr_db_min, config.noise_snr_db_max))
                target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
                value = value + noise * (target_noise_rms / noise_rms)

    return np.clip(value, -1.0, 1.0).astype(np.float32)


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing local-audio manifest: {file_path}. Materialize audio after EDA/merge.")
    rows: list[dict[str, Any]] = []
    with file_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            audio_path = Path(row.get("audio_path", ""))
            if not audio_path.is_absolute():
                audio_path = Path(__file__).resolve().parents[2] / audio_path
            if not row.get("text") or not audio_path.is_file():
                raise ValueError(f"{file_path}:{line_number}: invalid text/audio_path")
            row["audio_path"] = str(audio_path.resolve())
            rows.append(row)
    if not rows:
        raise ValueError(f"No examples in {file_path}")
    return rows


def preflight(config: ASRTrainingConfig) -> dict[str, Any]:
    train_rows = load_manifest(config.train_manifest)
    validation_rows = load_manifest(config.validation_manifest)
    train_fp = {row["text_fingerprint"] for row in train_rows}
    validation_fp = {row["text_fingerprint"] for row in validation_rows}
    overlap = train_fp & validation_fp
    train_speakers = {row.get("speaker") for row in train_rows if row.get("speaker")}
    validation_speakers = {row.get("speaker") for row in validation_rows if row.get("speaker")}
    speaker_overlap = train_speakers & validation_speakers
    if speaker_overlap:
        raise ValueError(f"Train/validation speaker leakage: {len(speaker_overlap)} speakers")
    train_groups = {row.get("group") for row in train_rows if row.get("group")}
    validation_groups = {row.get("group") for row in validation_rows if row.get("group")}
    group_overlap = train_groups & validation_groups
    if group_overlap:
        raise ValueError(f"Train/validation recording-group leakage: {len(group_overlap)} groups")
    train_audio_hashes = [row.get("audio_sha256") for row in train_rows]
    validation_audio_hashes = [row.get("audio_sha256") for row in validation_rows]
    if any(not value for value in train_audio_hashes + validation_audio_hashes):
        raise ValueError("Missing audio_sha256; run qc_local_audio.py exact-audio dedup first")
    if len(set(train_audio_hashes)) != len(train_audio_hashes):
        raise ValueError("Exact audio duplicates remain within train")
    if len(set(validation_audio_hashes)) != len(validation_audio_hashes):
        raise ValueError("Exact audio duplicates remain within validation")
    audio_overlap = set(train_audio_hashes) & set(validation_audio_hashes)
    if audio_overlap:
        raise ValueError(f"Train/validation exact-audio leakage: {len(audio_overlap)} hashes")
    return {
        "base_model": config.base_model,
        "base_model_revision": config.base_model_revision,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "transcript_overlap_observed_not_blocking": len(overlap),
        "speaker_overlap": 0,
        "recording_group_overlap": 0,
        "exact_audio_overlap": 0,
        "on_the_fly_augmentation": config.augmentation,
        "augmentation_profile": {
            "speed_probability": config.speed_probability,
            "gain_probability": config.gain_probability,
            "noise_probability": config.noise_probability,
            "noise_snr_db": [config.noise_snr_db_min, config.noise_snr_db_max],
            "reverb_probability": config.reverb_probability,
            "telephone_probability": config.telephone_probability,
            "codec_probability": config.codec_probability,
        },
        "precision": "bf16" if config.bf16 else ("fp16" if config.fp16 else "fp32"),
        "batching": {
            "train_per_device": config.per_device_train_batch_size,
            "validation_per_device": config.per_device_eval_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_train_batch": config.per_device_train_batch_size * config.gradient_accumulation_steps,
        },
        "max_steps": config.max_steps,
        "method": config.method,
        "clinical_oversample_factor": config.clinical_oversample_factor,
        "cpu_smoke": config.cpu_smoke,
        "gpu_memory_fraction": config.gpu_memory_fraction,
        "eval_steps": config.eval_steps,
        "save_steps": config.save_steps,
        "dry_run_limits": {
            "train": config.limit_train_examples,
            "validation": config.limit_validation_examples,
        },
    }


def train(config: ASRTrainingConfig) -> None:
    import evaluate
    import soundfile as sf
    import torch
    from datasets import Dataset
    from transformers import (
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        set_seed,
    )

    if not torch.cuda.is_available() and not config.cpu_smoke:
        raise RuntimeError("CUDA GPU is required for production ASR fine-tuning. Use --cpu-smoke only to verify the local path.")
    if not 0.0 < config.gpu_memory_fraction <= 0.40:
        raise ValueError("gpu_memory_fraction must be in (0, 0.40]")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction)
    if config.cpu_smoke and (
        config.max_steps < 1
        or config.max_steps > 5
        or config.limit_train_examples < 1
        or config.limit_validation_examples < 1
        or Path(config.output_dir) == Path(ASRTrainingConfig.output_dir)
    ):
        raise ValueError("CPU smoke requires 1-5 max steps, bounded train/validation data, and a non-production output path")
    report = preflight(config)
    logger.info("Preflight: %s", json.dumps(report, ensure_ascii=False))
    set_seed(config.seed)
    processor = WhisperProcessor.from_pretrained(
        config.base_model,
        revision=config.base_model_revision,
        language="vi",
        task="transcribe",
        local_files_only=config.local_files_only,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        config.base_model,
        revision=config.base_model_revision,
        local_files_only=config.local_files_only,
    )
    if config.method == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "v_proj"],
            ),
        )
        if config.gradient_checkpointing:
            model.enable_input_require_grads()
        model.print_trainable_parameters()
    model.generation_config.language = "vi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = not config.gradient_checkpointing

    def make_dataset(rows: list[dict[str, Any]], augment: bool) -> Dataset:
        return Dataset.from_list(
            [
                {"audio_path": row["audio_path"], "text": row["text"], "augment": augment}
                for row in rows
            ]
        )

    train_rows = load_manifest(config.train_manifest)
    validation_rows = load_manifest(config.validation_manifest)
    train_rows = prioritize_rows(train_rows, "asr", config.seed)
    if config.limit_train_examples:
        train_rows = train_rows[: config.limit_train_examples]
    if config.limit_validation_examples:
        validation_rows = validation_rows[: config.limit_validation_examples]
    train_rows = oversample_clinical_rows(
        train_rows,
        "asr",
        config.clinical_oversample_factor,
        config.seed,
    )
    train_dataset = make_dataset(train_rows, config.augmentation)
    validation_dataset = make_dataset(validation_rows, False)

    class SpeechCollator:
        """Decode and augment per batch so every epoch sees a new, deterministic stream."""

        def __init__(self) -> None:
            self.rng = np.random.default_rng(config.seed)

        def load_audio(self, path: str) -> np.ndarray:
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
            if sample_rate != 16000:
                raise ValueError(f"Expected signal-QC 16 kHz audio, got {sample_rate}: {path}")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return np.asarray(audio, dtype=np.float32)

        def augment_audio(self, audio: np.ndarray) -> np.ndarray:
            return augment_waveform(audio, self.rng, config)

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
            arrays = []
            for feature in features:
                audio = self.load_audio(feature["audio_path"])
                if feature.get("augment"):
                    audio = self.augment_audio(audio)
                arrays.append(audio)
            batch = processor.feature_extractor(
                arrays,
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
            )
            label_batch = processor.tokenizer(
                [feature["text"] for feature in features],
                padding=True,
                return_tensors="pt",
            )
            label_ids = label_batch["input_ids"].masked_fill(label_batch.attention_mask.ne(1), -100)
            if (label_ids[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
                label_ids = label_ids[:, 1:]
            batch["labels"] = label_ids
            return batch

    wer_metric = evaluate.load("wer")

    def metrics(prediction: Any) -> dict[str, float]:
        labels = prediction.label_ids
        labels[labels == -100] = processor.tokenizer.pad_token_id
        hypotheses = processor.tokenizer.batch_decode(prediction.predictions, skip_special_tokens=True)
        references = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {"wer": 100.0 * wer_metric.compute(predictions=hypotheses, references=references)}

    step_evaluation = config.cpu_smoke or config.eval_steps > 0
    evaluation_steps = 1 if config.cpu_smoke else (config.eval_steps or None)
    checkpoint_steps = 1 if config.cpu_smoke else (config.save_steps or config.eval_steps or 500)
    if step_evaluation and checkpoint_steps % int(evaluation_steps) != 0:
        raise ValueError("save_steps must be a multiple of eval_steps when step evaluation is enabled")
    args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16 and torch.cuda.is_available(),
        bf16=config.bf16 and torch.cuda.is_available(),
        max_steps=config.max_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        eval_strategy="steps" if step_evaluation else "epoch",
        save_strategy="steps" if step_evaluation else "epoch",
        eval_steps=evaluation_steps,
        save_steps=checkpoint_steps,
        logging_steps=1 if config.cpu_smoke else config.logging_steps,
        predict_with_generate=True,
        generation_max_length=config.generation_max_length,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        seed=config.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=SpeechCollator(),
        compute_metrics=metrics,
        processing_class=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
    run_report = {
        **report,
        "status": "cpu_smoke_complete" if config.cpu_smoke else "training_complete",
        "promotion_allowed": not config.cpu_smoke,
        "train_rows_after_oversampling": len(train_rows),
        "validation_rows": len(validation_rows),
        "global_step": trainer.state.global_step,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "train_metrics": train_result.metrics,
        "log_history": trainer.state.log_history,
        "gpu_peak_memory_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
        "gpu_peak_memory_reserved_bytes": (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        ),
    }
    Path(config.output_dir, "training_run.json").write_text(
        json.dumps(run_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=ASRTrainingConfig.base_model)
    parser.add_argument("--base-model-revision", default=ASRTrainingConfig.base_model_revision)
    parser.add_argument("--train-manifest", default=ASRTrainingConfig.train_manifest)
    parser.add_argument("--validation-manifest", default=ASRTrainingConfig.validation_manifest)
    parser.add_argument("--output-dir", default=ASRTrainingConfig.output_dir)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=ASRTrainingConfig.learning_rate)
    parser.add_argument("--warmup-ratio", type=float, default=ASRTrainingConfig.warmup_ratio)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Per-device validation batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Accumulate gradients to preserve the effective batch on lower-VRAM GPUs.",
    )
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1, help="Use 100 for a billed GPU dry-run; -1 trains by epochs.")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--resume-from-checkpoint", nargs="?", const="latest")
    parser.add_argument("--limit-train", type=int, default=0, help="Dry-run only; 0 uses the locked full manifest.")
    parser.add_argument("--limit-validation", type=int, default=0, help="Dry-run only; 0 uses the locked full manifest.")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--method", choices=["full", "lora"], default="full")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--clinical-oversample-factor", type=int, default=1)
    parser.add_argument("--cpu-smoke", action="store_true", help="Run a bounded real LoRA training step on CPU; never a release checkpoint.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.35)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=25)
    args = parser.parse_args()
    if min(args.batch_size, args.eval_batch_size, args.gradient_accumulation_steps) < 1:
        parser.error("batch sizes and gradient accumulation must be at least 1")
    if args.clinical_oversample_factor < 1 or args.lora_rank < 1 or args.lora_alpha < 1:
        parser.error("LoRA dimensions and clinical oversampling must be at least 1")
    if args.learning_rate <= 0 or not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("learning rate must be positive and warmup ratio must be in [0, 1)")
    if not 0.0 <= args.lora_dropout < 1.0:
        parser.error("--lora-dropout must be in [0, 1)")
    if args.cpu_smoke and args.method != "lora":
        parser.error("--cpu-smoke requires --method lora")
    if not 0.0 < args.gpu_memory_fraction <= 0.40:
        parser.error("--gpu-memory-fraction must be in (0, 0.40]")
    if min(args.eval_steps, args.save_steps, args.logging_steps) < 0 or args.logging_steps == 0:
        parser.error("step intervals must be non-negative and logging steps must be positive")
    config = ASRTrainingConfig(
        base_model=args.base_model,
        base_model_revision=args.base_model_revision,
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        augmentation=not args.no_augmentation,
        max_steps=args.max_steps,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        resume_from_checkpoint=(True if args.resume_from_checkpoint == "latest" else args.resume_from_checkpoint),
        limit_train_examples=args.limit_train,
        limit_validation_examples=args.limit_validation,
        method=args.method,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        clinical_oversample_factor=args.clinical_oversample_factor,
        cpu_smoke=args.cpu_smoke,
        local_files_only=args.local_files_only,
        gradient_checkpointing=not args.cpu_smoke,
        gpu_memory_fraction=args.gpu_memory_fraction,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )
    if args.preflight:
        print(json.dumps(preflight(config), ensure_ascii=False, indent=2))
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

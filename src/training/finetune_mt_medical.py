"""Fine-tune a sequence-to-sequence medical translator on locked manifests.

Translation is treated as a constrained encoder-decoder task. This is smaller,
easier to evaluate, and more realistic for an edge target than a chat LLM.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.training.clinical_sampling import oversample_clinical_rows, prioritize_rows


logger = logging.getLogger(__name__)


@dataclass
class MTTrainingConfig:
    base_model: str = "facebook/nllb-200-distilled-600M"
    base_model_revision: str = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
    train_manifest: str = "data/processed/manifests/mt--train.jsonl"
    validation_manifest: str = "data/processed/manifests/mt--validation.jsonl"
    output_dir: str = "models/mt/nllb-medical"
    direction: str = "joint"
    learning_rate: float = 2e-5
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_source_length: int = 256
    max_target_length: int = 256
    warmup_ratio: float = 0.05
    seed: int = 20260922
    fp16: bool = True
    bf16: bool = False
    max_steps: int = -1
    resume_from_checkpoint: str | bool | None = None
    initial_adapter: str | None = None
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
    logging_steps: int = 50
    early_stopping_patience: int = 4


def direction_fields(direction: str) -> tuple[str, str, str, str]:
    if direction == "en_to_vi":
        return "source_text", "target_text", "eng_Latn", "vie_Latn"
    if direction == "vi_to_en":
        return "target_text", "source_text", "vie_Latn", "eng_Latn"
    raise ValueError(f"Unsupported direction: {direction}")


def requested_directions(direction: str) -> tuple[str, ...]:
    if direction == "joint":
        return ("en_to_vi", "vi_to_en")
    direction_fields(direction)
    return (direction,)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing locked manifest: {file_path}. Run EDA and merge first.")
    rows: list[dict[str, Any]] = []
    with file_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("source_text") or not row.get("target_text"):
                raise ValueError(f"{file_path}:{line_number}: empty parallel text")
            rows.append(row)
    if not rows:
        raise ValueError(f"No examples in {file_path}")
    return rows


def preflight(config: MTTrainingConfig) -> dict[str, Any]:
    train_rows = load_jsonl(config.train_manifest)
    validation_rows = load_jsonl(config.validation_manifest)
    train_fp = {row["pair_fingerprint"] for row in train_rows}
    validation_fp = {row["pair_fingerprint"] for row in validation_rows}
    overlap = train_fp & validation_fp
    if overlap:
        raise ValueError(f"Train/validation leakage: {len(overlap)} exact pairs")
    return {
        "base_model": config.base_model,
        "base_model_revision": config.base_model_revision,
        "direction": config.direction,
        "directions": list(requested_directions(config.direction)),
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "effective_train_examples": len(train_rows) * len(requested_directions(config.direction)),
        "effective_validation_examples": len(validation_rows) * len(requested_directions(config.direction)),
        "exact_overlap": 0,
        "max_source_length": config.max_source_length,
        "max_target_length": config.max_target_length,
        "precision": "bf16" if config.bf16 else ("fp16" if config.fp16 else "fp32"),
        "batching": {
            "train_per_device": config.per_device_train_batch_size,
            "validation_per_device": config.per_device_eval_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_train_batch": config.per_device_train_batch_size * config.gradient_accumulation_steps,
        },
        "max_steps": config.max_steps,
        "initial_adapter": config.initial_adapter,
        "method": config.method,
        "clinical_oversample_factor": config.clinical_oversample_factor,
        "cpu_smoke": config.cpu_smoke,
        "gpu_memory_fraction": config.gpu_memory_fraction,
        "eval_steps": config.eval_steps,
        "save_steps": config.save_steps,
        "early_stopping_patience": config.early_stopping_patience,
        "dry_run_limits": {
            "train": config.limit_train_examples,
            "validation": config.limit_validation_examples,
        },
    }


def train(config: MTTrainingConfig) -> None:
    import torch
    from datasets import Dataset, concatenate_datasets
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available() and not config.cpu_smoke:
        raise RuntimeError("CUDA GPU is required for production MT fine-tuning. Use --cpu-smoke only to verify the local path.")
    if not 0.0 < config.gpu_memory_fraction <= 0.40:
        raise ValueError("gpu_memory_fraction must be in (0, 0.40]")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction)
    if config.cpu_smoke and (
        config.max_steps < 1
        or config.max_steps > 5
        or config.limit_train_examples < 1
        or config.limit_validation_examples < 1
        or Path(config.output_dir) == Path(MTTrainingConfig.output_dir)
    ):
        raise ValueError("CPU smoke requires 1-5 max steps, bounded train/validation data, and a non-production output path")
    report = preflight(config)
    logger.info("Preflight: %s", json.dumps(report, ensure_ascii=False))
    set_seed(config.seed)

    directions = requested_directions(config.direction)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        revision=config.base_model_revision,
        local_files_only=config.local_files_only,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model,
        revision=config.base_model_revision,
        local_files_only=config.local_files_only,
    )
    if config.method == "lora":
        if config.initial_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                config.initial_adapter,
                is_trainable=True,
            )
        else:
            from peft import LoraConfig, TaskType, get_peft_model

            model = get_peft_model(
                model,
                LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    r=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=["q_proj", "v_proj"],
                ),
            )
        model.print_trainable_parameters()

    def tokenized_direction(rows: list[dict[str, Any]], direction: str) -> Dataset:
        src_field, tgt_field, src_lang, tgt_lang = direction_fields(direction)
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
        directional = Dataset.from_list(
            [
                {
                    "model_source": row[src_field],
                    "model_target": row[tgt_field],
                    "direction": direction,
                }
                for row in rows
            ]
        )

        def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
            inputs = tokenizer(batch["model_source"], max_length=config.max_source_length, truncation=True)
            labels = tokenizer(
                text_target=batch["model_target"],
                max_length=config.max_target_length,
                truncation=True,
            )
            inputs["labels"] = labels["input_ids"]
            return inputs

        return directional.map(tokenize, batched=True, remove_columns=directional.column_names)

    train_rows = load_jsonl(config.train_manifest)
    validation_rows = load_jsonl(config.validation_manifest)
    train_rows = prioritize_rows(train_rows, "mt", config.seed)
    if config.limit_train_examples:
        train_rows = train_rows[: config.limit_train_examples]
    if config.limit_validation_examples:
        validation_rows = validation_rows[: config.limit_validation_examples]
    train_rows = oversample_clinical_rows(
        train_rows,
        "mt",
        config.clinical_oversample_factor,
        config.seed,
    )
    train_dataset = concatenate_datasets(
        [tokenized_direction(train_rows, direction) for direction in directions]
    ).shuffle(seed=config.seed)
    validation_dataset = concatenate_datasets(
        [tokenized_direction(validation_rows, direction) for direction in directions]
    )

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
        eval_strategy="steps" if step_evaluation else "epoch",
        save_strategy="steps" if step_evaluation else "epoch",
        eval_steps=evaluation_steps,
        save_steps=checkpoint_steps,
        logging_steps=1 if config.cpu_smoke else config.logging_steps,
        # Joint EN↔VI validation uses target-language tokens embedded in labels.
        # Generation is benchmarked separately because forced BOS differs by row.
        predict_with_generate=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        seed=config.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
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
    parser.add_argument("--base-model", default=MTTrainingConfig.base_model)
    parser.add_argument("--base-model-revision", default=MTTrainingConfig.base_model_revision)
    parser.add_argument("--train-manifest", default=MTTrainingConfig.train_manifest)
    parser.add_argument("--validation-manifest", default=MTTrainingConfig.validation_manifest)
    parser.add_argument("--output-dir", default=MTTrainingConfig.output_dir)
    parser.add_argument("--direction", choices=["joint", "en_to_vi", "vi_to_en"], default="joint")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=MTTrainingConfig.learning_rate)
    parser.add_argument("--warmup-ratio", type=float, default=MTTrainingConfig.warmup_ratio)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Per-device validation batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Accumulate gradients to preserve the effective batch on lower-VRAM GPUs.",
    )
    parser.add_argument("--max-steps", type=int, default=-1, help="Use 100 for a billed GPU dry-run; -1 trains by epochs.")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--resume-from-checkpoint", nargs="?", const="latest")
    parser.add_argument("--initial-adapter")
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
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
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
    if args.initial_adapter and args.method != "lora":
        parser.error("--initial-adapter requires --method lora")
    if args.initial_adapter and not (
        Path(args.initial_adapter) / "adapter_config.json"
    ).is_file():
        parser.error("--initial-adapter must contain adapter_config.json")
    if not 0.0 < args.gpu_memory_fraction <= 0.40:
        parser.error("--gpu-memory-fraction must be in (0, 0.40]")
    if min(args.eval_steps, args.save_steps, args.logging_steps) < 0 or args.logging_steps == 0:
        parser.error("step intervals must be non-negative and logging steps must be positive")
    if args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience must be positive")
    config = MTTrainingConfig(
        base_model=args.base_model,
        base_model_revision=args.base_model_revision,
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        output_dir=args.output_dir,
        direction=args.direction,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        resume_from_checkpoint=(True if args.resume_from_checkpoint == "latest" else args.resume_from_checkpoint),
        initial_adapter=args.initial_adapter,
        limit_train_examples=args.limit_train,
        limit_validation_examples=args.limit_validation,
        method=args.method,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        clinical_oversample_factor=args.clinical_oversample_factor,
        cpu_smoke=args.cpu_smoke,
        local_files_only=args.local_files_only,
        gpu_memory_fraction=args.gpu_memory_fraction,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        early_stopping_patience=args.early_stopping_patience,
    )
    if args.preflight:
        print(json.dumps(preflight(config), ensure_ascii=False, indent=2))
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

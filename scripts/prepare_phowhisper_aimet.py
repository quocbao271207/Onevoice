"""Calibrate optimized PhoWhisper components with Qualcomm's AIMET recipe.

This entrypoint is intentionally Linux-only. It never edits ``site-packages``
and never reuses the official OpenAI Whisper encodings for PhoWhisper weights.
Decoder calibration inputs are streamed from real train audio and transcripts.
Raw audio, transcript text, token IDs, and KV-cache tensors are never persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

MEAN_DECODE_LEN = 200
AUDIO_EMB_LEN = 1500

if __package__:
    from scripts.export_phowhisper_aihub import (
        DEFAULT_MODEL,
        DEFAULT_REVISION,
        resolve_checkpoint,
        validate_whisper_small_config,
    )
else:
    from export_phowhisper_aihub import (  # type: ignore[no-redef]
        DEFAULT_MODEL,
        DEFAULT_REVISION,
        resolve_checkpoint,
        validate_whisper_small_config,
    )


def require_linux(system_name: str | None = None) -> None:
    detected = system_name or platform.system()
    if detected.lower() != "linux":
        raise RuntimeError(
            "AIMET-ONNX custom calibration requires Linux. "
            "Do not patch the Windows package or reuse official encodings."
        )


def resolve_model_checkpoint(
    checkpoint_path: str | Path | None,
    hf_model: str,
    revision: str,
) -> Path:
    if checkpoint_path is None:
        return resolve_checkpoint(hf_model, revision)
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    from transformers import WhisperConfig

    validate_whisper_small_config(WhisperConfig.from_pretrained(checkpoint))
    return checkpoint


def select_calibration_rows(
    manifest_path: str | Path,
    workspace: str | Path,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("calibration limit must be at least 1")
    root = Path(workspace).resolve()
    candidates: list[tuple[str, dict[str, Any]]] = []
    with Path(manifest_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("role") != "train":
                raise ValueError(f"non-train row at line {line_number}")
            row_id = str(row.get("id", ""))
            audio_path = row.get("audio_path")
            if not row_id or not isinstance(audio_path, str):
                raise ValueError(f"missing id/audio_path at line {line_number}")
            resolved_audio = (root / audio_path).resolve()
            try:
                resolved_audio.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"audio path escapes workspace at line {line_number}") from exc
            if not resolved_audio.is_file():
                raise FileNotFoundError(resolved_audio)
            selected = dict(row)
            selected["resolved_audio_path"] = str(resolved_audio)
            rank = hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()
            candidates.append((rank, selected))
    if not candidates:
        raise ValueError("calibration manifest contains no rows")
    return [row for _, row in sorted(candidates)[:limit]]


def encoder_feature_batches(
    rows: Iterable[dict[str, Any]],
    checkpoint: Path,
) -> Iterable[dict[str, Any]]:
    import numpy as np
    import soundfile as sf
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor.from_pretrained(checkpoint)
    for row in rows:
        audio, sample_rate = sf.read(
            row["resolved_audio_path"],
            dtype="float32",
            always_2d=False,
        )
        if sample_rate != 16000 or np.asarray(audio).ndim != 1:
            raise ValueError(
                f"expected mono 16 kHz audio: {row['resolved_audio_path']}"
            )
        features = extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="np",
        ).input_features.astype("float32", copy=False)
        yield {"input_features": features}


def decoder_input_spec(
    config: Any,
    mean_decode_len: int = MEAN_DECODE_LEN,
    audio_emb_len: int = AUDIO_EMB_LEN,
) -> dict[str, tuple[tuple[int, ...], str]]:
    """Return the exact ordered input contract used by Qualcomm's decoder."""
    layers = int(config.decoder_layers)
    heads = int(config.decoder_attention_heads)
    head_dim = int(config.d_model) // heads
    specs: dict[str, tuple[tuple[int, ...], str]] = {
        "input_ids": ((1, 1), "int32"),
        "attention_mask": ((1, 1, 1, mean_decode_len), "float32"),
    }
    for index in range(layers):
        specs[f"k_cache_self_{index}_in"] = (
            (heads, 1, head_dim, mean_decode_len - 1),
            "float32",
        )
        specs[f"v_cache_self_{index}_in"] = (
            (heads, 1, mean_decode_len - 1, head_dim),
            "float32",
        )
    for index in range(layers):
        specs[f"k_cache_cross_{index}"] = (
            (heads, 1, head_dim, audio_emb_len),
            "float32",
        )
        specs[f"v_cache_cross_{index}"] = (
            (heads, 1, audio_emb_len, head_dim),
            "float32",
        )
    specs["position_ids"] = ((1,), "int32")
    return specs


def validate_decoder_calibration_batch(
    batch: dict[str, Any],
    specs: dict[str, tuple[tuple[int, ...], str]],
) -> None:
    """Fail before AIMET sees a decoder batch with an incompatible contract."""
    import numpy as np

    if list(batch) != list(specs):
        raise ValueError("decoder calibration input names/order do not match recipe")
    for name, (expected_shape, expected_dtype) in specs.items():
        value = np.asarray(batch[name])
        if value.shape != expected_shape:
            raise ValueError(
                f"decoder input {name} shape {value.shape} != {expected_shape}"
            )
        if value.dtype.name != expected_dtype:
            raise ValueError(
                f"decoder input {name} dtype {value.dtype.name} != {expected_dtype}"
            )


def _flatten_cache(cache: Any) -> tuple[Any, ...]:
    return tuple(value for pair in cache for value in pair)


def teacher_forced_decoder_batches(
    rows: Iterable[dict[str, Any]],
    checkpoint: Path,
    encoder: Any,
    decoder: Any,
    config: Any,
    steps_per_row: int,
    stats: dict[str, int] | None = None,
) -> Iterable[dict[str, Any]]:
    """Stream representative decoder states without writing sensitive tensors."""
    if not 1 <= steps_per_row < MEAN_DECODE_LEN:
        raise ValueError(f"decoder steps must be between 1 and {MEAN_DECODE_LEN - 1}")

    import numpy as np
    import soundfile as sf
    import torch
    from transformers import WhisperFeatureExtractor, WhisperTokenizer

    extractor = WhisperFeatureExtractor.from_pretrained(checkpoint)
    tokenizer = WhisperTokenizer.from_pretrained(
        checkpoint,
        language="vi",
        task="transcribe",
    )
    specs = decoder_input_spec(config)
    layers = int(config.decoder_layers)
    heads = int(config.decoder_attention_heads)
    head_dim = int(config.d_model) // heads
    mask_neg = float(config.mask_neg)
    if stats is not None:
        stats["batches"] = 0

    for row in rows:
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"missing transcript for calibration row {row['id']}")
        audio, sample_rate = sf.read(
            row["resolved_audio_path"],
            dtype="float32",
            always_2d=False,
        )
        if sample_rate != 16000 or np.asarray(audio).ndim != 1:
            raise ValueError(
                f"expected mono 16 kHz audio: {row['resolved_audio_path']}"
            )
        features = extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).input_features
        with torch.no_grad():
            cross_cache = encoder(features)
        if not isinstance(cross_cache, (tuple, list)):
            raise ValueError("encoder did not return decoder cross-attention caches")
        if cross_cache and not isinstance(cross_cache[0], (tuple, list)):
            cross_cache = tuple(
                (cross_cache[index], cross_cache[index + 1])
                for index in range(0, len(cross_cache), 2)
            )
        if len(cross_cache) != layers:
            raise ValueError("encoder cross-cache layer count does not match decoder")

        token_ids = tokenizer(text.strip(), return_tensors="pt").input_ids[0]
        if token_ids.numel() < 1:
            raise ValueError(f"tokenizer returned no tokens for calibration row {row['id']}")
        self_cache = tuple(
            (
                torch.zeros((heads, 1, head_dim, MEAN_DECODE_LEN - 1)),
                torch.zeros((heads, 1, MEAN_DECODE_LEN - 1, head_dim)),
            )
            for _ in range(layers)
        )
        attention_mask = torch.full(
            (1, 1, 1, MEAN_DECODE_LEN),
            mask_neg,
            dtype=torch.float32,
        )

        for step in range(min(steps_per_row, int(token_ids.numel()))):
            attention_mask[:, :, :, MEAN_DECODE_LEN - step - 1] = 0.0
            decoder_args = (
                token_ids[step : step + 1].reshape(1, 1).to(torch.int32),
                attention_mask,
                *_flatten_cache(self_cache),
                *_flatten_cache(cross_cache),
                torch.tensor([step], dtype=torch.int32),
            )
            batch = {
                name: value.detach().cpu().numpy().copy()
                for name, value in zip(specs, decoder_args, strict=True)
            }
            validate_decoder_calibration_batch(batch, specs)
            with torch.no_grad():
                decoder_output = decoder(*decoder_args)
            if len(decoder_output) == 2 and isinstance(
                decoder_output[1], (tuple, list)
            ):
                self_cache = tuple(decoder_output[1])
            else:
                self_cache = tuple(
                    decoder_output[index : index + 2]
                    for index in range(1, len(decoder_output), 2)
                )
            if len(self_cache) != layers:
                raise ValueError("decoder self-cache layer count changed unexpectedly")
            if stats is not None:
                stats["batches"] += 1
            yield batch


def calibrate_encoder(
    checkpoint: Path,
    rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    # Set the recipe's in-memory checkpoint pointer before constructing the
    # quantizable class. This preserves the installed package byte-for-byte.
    import qai_hub_models.models.whisper_small.model as float_recipe

    float_recipe.WHISPER_VERSION = str(checkpoint)
    from qai_hub_models.models.whisper_small_quantized.model import (
        WhisperSmallEncoderQuantizable,
    )

    encoder = WhisperSmallEncoderQuantizable.from_pretrained(aimet_encodings=None)
    if encoder.quant_sim is None:
        raise RuntimeError("failed to construct AIMET QuantSim")
    encoder.quant_sim.compute_encodings(encoder_feature_batches(rows, checkpoint))
    encoder.save_calibrated_checkpoint(str(output_dir))


def calibrate_decoder(
    checkpoint: Path,
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    steps_per_row: int,
) -> int:
    """Calibrate the decoder with real teacher-forced token and cache states."""
    import qai_hub_models.models.whisper_small.model as float_recipe

    float_recipe.WHISPER_VERSION = str(checkpoint)
    from qai_hub_models.models.whisper_small.model import WhisperSmall
    from qai_hub_models.models.whisper_small_quantized.model import (
        WhisperSmallDecoderQuantizable,
    )

    float_model = WhisperSmall.from_pretrained()
    decoder = WhisperSmallDecoderQuantizable.from_pretrained(aimet_encodings=None)
    if decoder.quant_sim is None:
        raise RuntimeError("failed to construct decoder AIMET QuantSim")
    stats: dict[str, int] = {}
    batches = teacher_forced_decoder_batches(
        rows,
        checkpoint,
        float_model.encoder,
        float_model.decoder,
        float_model.config,
        steps_per_row,
        stats,
    )
    decoder.quant_sim.compute_encodings(batches)
    batch_count = stats.get("batches", 0)
    if batch_count < 1:
        raise RuntimeError("decoder calibration produced no batches")
    decoder.save_calibrated_checkpoint(str(output_dir))
    return batch_count


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--checkpoint",
        help="Local base or fine-tuned Whisper-small checkpoint; overrides --hf-model.",
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/manifests/asr--train-local.jsonl",
    )
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--component",
        choices=("encoder", "decoder"),
        default="encoder",
    )
    parser.add_argument("--decoder-steps-per-row", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    require_linux()
    workspace = Path.cwd()
    checkpoint = resolve_model_checkpoint(
        args.checkpoint,
        args.hf_model,
        args.revision,
    )
    rows = select_calibration_rows(args.manifest, workspace, args.limit, args.seed)
    decoder_batches = None
    if args.component == "encoder":
        calibrate_encoder(checkpoint, rows, args.output_dir)
    else:
        decoder_batches = calibrate_decoder(
            checkpoint,
            rows,
            args.output_dir,
            args.decoder_steps_per_row,
        )
    evidence = {
        "component": args.component,
        "checkpoint": str(checkpoint),
        "revision": args.revision,
        "calibration_kind": (
            "train_audio_features"
            if args.component == "encoder"
            else "train_audio_transcript_teacher_forced_in_memory"
        ),
        "calibration_rows": len(rows),
        "calibration_ids_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in rows).encode()
        ).hexdigest(),
        "decoder_steps_per_row": (
            args.decoder_steps_per_row if args.component == "decoder" else None
        ),
        "decoder_calibration_batches": decoder_batches,
        "sensitive_calibration_tensors_persisted": False,
    }
    destination = Path(args.output_dir) / "onevoice_calibration_evidence.json"
    destination.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

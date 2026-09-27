"""
MediVoice Edge — post-checkpoint export and quantization scaffold.

This module creates an ONNX Runtime dynamic-QInt8 *prototype* for desktop
parity testing.  It does not claim that this artifact is calibrated, accepted
by QNN, or deployable on Hexagon.  Device deployment remains a separate,
fail-closed step after a real fine-tuned checkpoint exists.

Pipeline:
    PyTorch Model → ONNX → dynamic-QInt8 ONNX (desktop parity candidate)
                                    ↘ optional SNPE .dlc conversion

Quantization targets:
- ASR (PhoWhisper): INT8 candidate; final quality/latency requires calibration and device measurement
- MT (NLLB seq2seq): INT8 candidate; final quality/latency requires calibration and device measurement
- TTS (Piper): preserve upstream ONNX first, then validate a device-specific quantization path

Usage:
    python -m src.training.quantize_qnn \
        --model-path outputs/asr-phowhisper-small \
        --model-type asr \
        --output-dir models/dlc/asr-vi

    python -m src.training.quantize_qnn \
        --model-path outputs/mt-nllb-joint \
        --model-type mt \
        --output-dir models/qnn/mt

This module fails closed for INT4, missing quantization dependencies, missing
SDK tools, and converter failures. ONNX Runtime QInt8 is not calibrated QNN
INT8 and must never be reported as such. QNN context-binary compilation for
QCS6490 remains a post-checkpoint device-SDK gate documented in
project_analysis.md.
"""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_to_onnx(
    model_path: str,
    model_type: str,
    output_path: str,
    opset_version: int = 17,
):
    """
    Export a PyTorch model to ONNX format.

    Args:
        model_path: Path to fine-tuned model
        model_type: "asr", "mt", or "tts"
        output_path: Output ONNX file path
        opset_version: ONNX opset version
    """
    logger.info(f"Exporting {model_type} model to ONNX...")
    logger.info(f"  Input: {model_path}")
    logger.info(f"  Output: {output_path}")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "asr":
        _export_whisper_onnx(model_path, str(output_dir), opset_version)
    elif model_type == "mt":
        _export_mt_onnx(model_path, str(output_dir), opset_version)
    elif model_type == "tts":
        _export_tts_onnx(model_path, str(output_dir), opset_version)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _export_whisper_onnx(model_path: str, output_dir: str, opset_version: int):
    """Export Whisper model to ONNX using optimum."""
    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

        logger.info("  Using optimum for Whisper ONNX export...")
        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            export=True,
        )
        model.save_pretrained(output_dir)
        logger.info(f"  ✅ Whisper ONNX exported to: {output_dir}")

    except ImportError:
        logger.warning("  optimum not available. Using torch.onnx.export fallback.")
        from transformers import WhisperForConditionalGeneration

        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        model.eval()

        # Export encoder
        dummy_input = torch.randn(1, 80, 3000)  # Mel spectrogram
        onnx_path = Path(output_dir) / "encoder.onnx"

        torch.onnx.export(
            model.get_encoder(),
            dummy_input,
            str(onnx_path),
            opset_version=opset_version,
            input_names=["input_features"],
            output_names=["encoder_output"],
            dynamic_axes={"input_features": {0: "batch", 2: "time"}},
        )
        logger.info(f"  ✅ Whisper encoder exported to: {onnx_path}")


def _export_mt_onnx(model_path: str, output_dir: str, opset_version: int):
    """Export the joint NLLB encoder-decoder checkpoint to ONNX."""
    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM

        logger.info("  Using optimum for NLLB seq2seq ONNX export...")
        model = ORTModelForSeq2SeqLM.from_pretrained(
            model_path,
            export=True,
        )
        model.save_pretrained(output_dir)
        logger.info(f"  ✅ NLLB ONNX exported to: {output_dir}")

    except ImportError as exc:
        raise RuntimeError(
            "NLLB ONNX export requires optimum[onnxruntime]; no silent fallback is allowed."
        ) from exc


def _export_tts_onnx(model_path: str, output_dir: str, opset_version: int):
    """Export TTS model to ONNX."""
    logger.info("  Piper-TTS models are already in ONNX format.")
    # Piper models come as .onnx files, just copy them
    src = Path(model_path)
    dst = Path(output_dir)
    if src.exists():
        for f in src.glob("*.onnx"):
            shutil.copy2(f, dst / f.name)
            logger.info(f"  Copied: {f.name}")
    logger.info(f"  ✅ TTS ONNX files in: {output_dir}")


def quantize_onnx(
    onnx_path: str,
    output_path: str,
    precision: str = "int8",
    calibration_data: list | None = None,
) -> list[Path]:
    """
    Produce a dynamic-QInt8 ONNX Runtime prototype.

    Args:
        onnx_path: Path to input ONNX model
        output_path: Path to save quantized model
        precision: only "int8" is implemented here
        calibration_data: reserved for a future calibrated path; rejected now
    """
    logger.info(f"Quantizing ONNX model to {precision.upper()}...")
    logger.info(f"  Input: {onnx_path}")
    logger.info(f"  Output: {output_path}")

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find ONNX files
        onnx_dir = Path(onnx_path)
        onnx_files = list(onnx_dir.glob("*.onnx")) if onnx_dir.is_dir() else [onnx_dir]

        if not onnx_files or not all(path.is_file() for path in onnx_files):
            raise FileNotFoundError(f"No ONNX model files found at: {onnx_path}")
        if calibration_data is not None:
            raise NotImplementedError(
                "Calibrated PTQ is not implemented in this desktop scaffold; "
                "use a verified QNN/AIMET calibration workflow after fine-tuning."
            )

        outputs: list[Path] = []
        for onnx_file in onnx_files:
            out_file = output_dir / f"{onnx_file.stem}_quantized.onnx"

            if precision == "int8":
                # Dynamic quantization (no calibration data needed)
                quantize_dynamic(
                    str(onnx_file),
                    str(out_file),
                    weight_type=QuantType.QInt8,
                )
                outputs.append(out_file)
            else:
                raise ValueError(
                    "INT4 is not implemented. Use a verified QNN/AIMET path after a real checkpoint exists."
                )

            # Report size reduction
            original_size = onnx_file.stat().st_size / (1024 * 1024)
            quantized_size = out_file.stat().st_size / (1024 * 1024)
            reduction = (1 - quantized_size / original_size) * 100

            logger.info(
                f"  ✅ desktop parity candidate {onnx_file.name}: "
                f"{original_size:.1f}MB → {quantized_size:.1f}MB "
                f"({reduction:.0f}% reduction)"
            )

        return outputs

    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime quantization is required; install the locked local dependencies."
        ) from exc


def compile_to_dlc(onnx_path: str, output_path: str) -> Path:
    """
    Compile ONNX to SNPE DLC, failing closed when the SDK is unavailable.

    This requires the Qualcomm Neural Processing SDK to be installed.

    Args:
        onnx_path: Path to quantized ONNX model
        output_path: Path to save .dlc file
    """
    logger.info(f"Compiling to Qualcomm DLC format...")
    logger.info(f"  Input: {onnx_path}")
    logger.info(f"  Output: {output_path}")

    # A QNN converter does not produce an SNPE DLC. Keep the paths explicit so
    # a QNN model library/context binary is never mislabeled with a .dlc suffix.
    snpe_convert = shutil.which("snpe-onnx-to-dlc")
    if not snpe_convert:
        raise RuntimeError(
            "SNPE converter 'snpe-onnx-to-dlc' is not on PATH; DLC conversion was requested "
            "and cannot be skipped. QNN output requires its own model-library/context-binary workflow."
        )

    source = Path(onnx_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        snpe_convert,
        "--input_network", str(source),
        "--output_path", str(destination),
    ]
    logger.info("  Running SNPE converter")
    subprocess.run(cmd, check=True)
    if not destination.is_file():
        raise RuntimeError(f"SNPE converter exited successfully but produced no DLC: {destination}")
    logger.info(f"  ✅ SNPE DLC compiled: {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(
        description="Quantize and compile models for Qualcomm NPU"
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--model-type", required=True,
        choices=["asr", "mt", "tts"],
        help="Type of model"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for quantized/compiled model"
    )
    parser.add_argument(
        "--precision", default="int8",
        choices=["int8"],
        help="Quantization precision"
    )
    parser.add_argument(
        "--compile-dlc", action="store_true",
        help="Also compile to Qualcomm DLC format"
    )

    args = parser.parse_args()

    # Step 1: Export to ONNX
    onnx_dir = Path(args.output_dir) / "onnx"
    export_to_onnx(args.model_path, args.model_type, str(onnx_dir))

    # Step 2: Quantize
    quant_dir = Path(args.output_dir) / f"quantized_{args.precision}"
    quantized_files = quantize_onnx(str(onnx_dir), str(quant_dir), args.precision)

    # Step 3: Compile to DLC (optional)
    if args.compile_dlc:
        dlc_dir = Path(args.output_dir) / "dlc"
        dlc_dir.mkdir(parents=True, exist_ok=True)
        for onnx_file in quantized_files:
            dlc_path = dlc_dir / f"{onnx_file.stem}.dlc"
            compile_to_dlc(str(onnx_file), str(dlc_path))

    logger.info("\n✅ Quantization pipeline complete!")
    logger.info(f"   ONNX: {onnx_dir}")
    logger.info(f"   Quantized: {quant_dir}")


if __name__ == "__main__":
    main()

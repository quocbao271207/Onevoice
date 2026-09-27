"""Export the pinned PhoWhisper-small checkpoint with Qualcomm's Whisper recipe.

Run this script with the isolated ``.venv-aihub-models`` environment. The
floating-point recipe works on Windows; calibrated w8a16 export requires
AIMET-ONNX on Linux/WSL and intentionally remains a separate gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "vinai/PhoWhisper-small"
DEFAULT_REVISION = "a86b604c346caf7148c37512eafe783a16420adb"


def validate_whisper_small_config(config: Any) -> None:
    """Reject checkpoints that do not match the Qualcomm Whisper-small graph."""
    expected = {
        "model_type": "whisper",
        "d_model": 768,
        "encoder_layers": 12,
        "decoder_layers": 12,
        "encoder_attention_heads": 12,
        "decoder_attention_heads": 12,
        "num_mel_bins": 80,
    }
    mismatches = {
        name: {"expected": value, "actual": getattr(config, name, None)}
        for name, value in expected.items()
        if getattr(config, name, None) != value
    }
    if mismatches:
        raise RuntimeError(f"checkpoint is not Whisper-small compatible: {mismatches}")


def resolve_checkpoint(model_id: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download
    from transformers import WhisperConfig

    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
        )
    )
    validate_whisper_small_config(WhisperConfig.from_pretrained(snapshot))
    return snapshot


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    custom = argparse.ArgumentParser(add_help=False)
    custom.add_argument("--hf-model", default=DEFAULT_MODEL)
    custom.add_argument("--revision", default=DEFAULT_REVISION)
    custom_args, recipe_args = custom.parse_known_args(argv)

    checkpoint = resolve_checkpoint(custom_args.hf_model, custom_args.revision)

    # The official float recipe reads this module global when its Model class
    # is instantiated. Pointing it at the immutable local snapshot preserves
    # Qualcomm's graph adaptations while using the PhoWhisper weights.
    import qai_hub_models.models.whisper_small.model as recipe_model
    from qai_hub_models.models.whisper_small import export as recipe_export

    recipe_model.WHISPER_VERSION = str(checkpoint)
    parsed_recipe_args = recipe_export.build_parser().parse_args(recipe_args)
    recipe_export.main(parsed_recipe_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared fail-closed checks for autoregressive sequence generation."""

from __future__ import annotations

from typing import Any


class GenerationContractError(RuntimeError):
    """A generated sequence violated the fail-closed completion contract."""


def require_completed_generation(
    generated_tokens: Any,
    eos_token_id: Any,
    *,
    pad_token_id: Any = None,
    context: str = "Generation",
) -> None:
    """Require a generated EOS after the decoder start token."""
    if eos_token_id is None:
        raise GenerationContractError(
            f"{context} cannot verify completion without an EOS token ID"
        )
    values = (
        generated_tokens.tolist()
        if hasattr(generated_tokens, "tolist")
        else generated_tokens
    )
    if not isinstance(values, (list, tuple)) or not values:
        raise GenerationContractError(
            f"{context} returned an invalid token sequence"
        )
    if isinstance(values[0], (list, tuple)):
        raise GenerationContractError(f"{context} expected one token sequence")
    eos_ids = (
        {int(token_id) for token_id in eos_token_id}
        if isinstance(eos_token_id, (list, tuple, set, frozenset))
        else {int(eos_token_id)}
    )
    terminator = next(
        (
            index
            for index, token_id in enumerate(values[1:], start=1)
            if int(token_id) in eos_ids
        ),
        None,
    )
    if terminator is None:
        raise GenerationContractError(
            f"{context} did not produce EOS; refusing potentially truncated output"
        )
    if pad_token_id is not None:
        allowed_trailing_ids = eos_ids | {int(pad_token_id)}
        if any(
            int(token_id) not in allowed_trailing_ids
            for token_id in values[terminator + 1 :]
        ):
            raise GenerationContractError(f"{context} produced tokens after EOS")


# Version 2 replaces the token-only Whisper EOS check, which cannot work with
# transformers' Whisper post-processing (see ``require_completed_whisper_batch``).
GENERATION_CONTRACT_VERSION = 2


class GenerationStepCounter:
    """Stopping criterion that never stops; it only counts decoding steps.

    Hugging Face calls every stopping criterion exactly once per generated
    token (greedy and beam search alike), so ``steps`` equals the number of
    tokens the decoding loop produced, independent of any post-processing
    that ``generate()`` later applies to the returned sequences.
    """

    def __init__(self) -> None:
        self.steps = 0

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
        import torch

        self.steps += 1
        return torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )


def require_completed_whisper_batch(
    generated: Any,
    eos_token_id: Any,
    *,
    pad_token_id: Any = None,
    generation_steps: int | None,
    max_new_tokens: int,
    max_target_positions: int | None = None,
    contexts: list[str] | None = None,
) -> None:
    """Fail closed unless every Whisper row provably terminated with EOS.

    Whisper's ``generate()`` removes trailing ``pad_token_id`` tokens from each
    row and re-pads the batch. Whisper uses ``pad_token_id == eos_token_id``,
    so the longest row always loses its genuine EOS and the token-only check
    in ``require_completed_generation`` rejects every batch.

    The only stopping conditions active here are "all rows emitted EOS" and the
    length cap. If the decoding loop ran strictly fewer steps than
    ``max_new_tokens`` and the output stayed below the decoder's position
    limit (Whisper may lower the cap to fit ``max_target_positions``), then
    the loop ended because every row emitted EOS. Otherwise, fall back to the
    strict per-row token check, which rejects any row without a visible EOS.
    """
    rows = generated.tolist() if hasattr(generated, "tolist") else generated
    if not isinstance(rows, (list, tuple)) or not rows:
        raise GenerationContractError("Whisper generation returned no sequences")
    labels = contexts or [f"Generation row {index}" for index in range(len(rows))]
    if len(labels) != len(rows):
        raise GenerationContractError("Whisper generation context count mismatch")
    for label, row in zip(labels, rows):
        if not isinstance(row, (list, tuple)) or not row:
            raise GenerationContractError(f"{label} returned an invalid token sequence")
    if isinstance(generation_steps, bool) or not isinstance(generation_steps, int):
        generation_steps = None
    width = max(len(row) for row in rows)
    below_position_limit = max_target_positions is None or width < int(
        max_target_positions
    )
    if (
        generation_steps is not None
        and 0 < generation_steps < int(max_new_tokens)
        and below_position_limit
    ):
        return
    for label, row in zip(labels, rows):
        require_completed_generation(
            row, eos_token_id, pad_token_id=pad_token_id, context=label
        )

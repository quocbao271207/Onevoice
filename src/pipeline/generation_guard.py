"""Shared fail-closed checks for autoregressive sequence generation."""

from __future__ import annotations

from typing import Any


def require_completed_generation(
    generated_tokens: Any,
    eos_token_id: Any,
    *,
    pad_token_id: Any = None,
    context: str = "Generation",
) -> None:
    """Require a generated EOS after the decoder start token."""
    if eos_token_id is None:
        raise RuntimeError(
            f"{context} cannot verify completion without an EOS token ID"
        )
    values = (
        generated_tokens.tolist()
        if hasattr(generated_tokens, "tolist")
        else generated_tokens
    )
    if not isinstance(values, (list, tuple)) or not values:
        raise RuntimeError(f"{context} returned an invalid token sequence")
    if isinstance(values[0], (list, tuple)):
        raise RuntimeError(f"{context} expected one token sequence")
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
        raise RuntimeError(
            f"{context} did not produce EOS; refusing potentially truncated output"
        )
    if pad_token_id is not None:
        allowed_trailing_ids = eos_ids | {int(pad_token_id)}
        if any(
            int(token_id) not in allowed_trailing_ids
            for token_id in values[terminator + 1 :]
        ):
            raise RuntimeError(f"{context} produced tokens after EOS")

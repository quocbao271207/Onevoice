"""Model-family differences shared by MT training and evaluation.

The bake-off mixes NLLB, M2M100 and one-direction VinAI mBART models.  Their
language tokens are not interchangeable, so keeping the mapping in one module
prevents a candidate from being scored with another model's decoding recipe.
"""

from __future__ import annotations

from typing import Any


SUPPORTED_MT_FAMILIES = ("nllb", "m2m100", "mbart_directional")
SUPPORTED_MT_DIRECTIONS = ("en_to_vi", "vi_to_en")


def direction_fields(direction: str) -> tuple[str, str]:
    if direction == "en_to_vi":
        return "source_text", "target_text"
    if direction == "vi_to_en":
        return "target_text", "source_text"
    raise ValueError(f"Unsupported MT direction: {direction}")


def language_codes(family: str, direction: str) -> tuple[str | None, str | None]:
    if family not in SUPPORTED_MT_FAMILIES:
        raise ValueError(f"Unsupported MT model family: {family}")
    if direction not in SUPPORTED_MT_DIRECTIONS:
        raise ValueError(f"Unsupported MT direction: {direction}")
    if family == "nllb":
        return ("eng_Latn", "vie_Latn") if direction == "en_to_vi" else (
            "vie_Latn",
            "eng_Latn",
        )
    if family == "m2m100":
        return ("en", "vi") if direction == "en_to_vi" else ("vi", "en")
    return None, None


def configure_tokenizer(tokenizer: Any, family: str, direction: str) -> None:
    source_lang, target_lang = language_codes(family, direction)
    if source_lang is not None:
        tokenizer.src_lang = source_lang
    if target_lang is not None:
        tokenizer.tgt_lang = target_lang


def forced_bos_token_id(tokenizer: Any, family: str, direction: str) -> int | None:
    _, target_lang = language_codes(family, direction)
    if target_lang is None:
        return None
    if family == "m2m100":
        return int(tokenizer.get_lang_id(target_lang))
    token_id = int(tokenizer.convert_tokens_to_ids(target_lang))
    if token_id < 0:
        raise ValueError(f"Tokenizer does not contain target language token {target_lang!r}")
    return token_id


def requested_directions(direction: str, family: str) -> tuple[str, ...]:
    if direction == "joint":
        if family == "mbart_directional":
            raise ValueError("A one-direction mBART candidate cannot use direction=joint")
        return SUPPORTED_MT_DIRECTIONS
    if direction not in SUPPORTED_MT_DIRECTIONS:
        raise ValueError(f"Unsupported MT direction: {direction}")
    return (direction,)

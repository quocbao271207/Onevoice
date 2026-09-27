"""Versioned normalization for reproducible Vietnamese/English ASR scoring."""

from __future__ import annotations

import re
import unicodedata


NORMALIZER_VERSION = "onevoice-wer-v1"
_PUNCT = re.compile(r"[^\w\s%°.,]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_for_wer(text: str) -> str:
    value = unicodedata.normalize("NFC", text or "").casefold()
    value = _PUNCT.sub(" ", value)
    value = value.replace(",", " ").replace(".", " ")
    return _SPACE.sub(" ", value).strip()

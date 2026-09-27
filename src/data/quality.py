"""Pure, dependency-light quality primitives used by data preparation scripts."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence


_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_VI_MARKS = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")


def normalize_text(text: Any) -> str:
    """Unicode-normalize text without destroying clinically meaningful punctuation."""
    value = unicodedata.normalize("NFC", str(text or ""))
    return _SPACE_RE.sub(" ", value).strip()


def fingerprint_text(text: Any) -> str:
    """Cross-source content hash; deliberately excludes source name."""
    value = normalize_text(text).casefold()
    value = _PUNCT_RE.sub(" ", value)
    value = _SPACE_RE.sub(" ", value).strip()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_record_id(source: str, split: str, native_id: Any, text: Any) -> str:
    payload = "\x1f".join((source, split, normalize_text(native_id), fingerprint_text(text)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def vietnamese_mark_ratio(text: Any) -> float:
    letters = [c.casefold() for c in normalize_text(text) if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in _VI_MARKS for c in letters) / len(letters)


def audio_url(value: Any) -> str | None:
    """Extract the dataset-server audio URL without downloading/decoding audio."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return value.get("url") or value.get("path") or value.get("src")
    if isinstance(value, Sequence) and value and isinstance(value[0], Mapping):
        return value[0].get("src") or value[0].get("url")
    return None


@dataclass
class RunningProfile:
    rows: int = 0
    empty_text: int = 0
    invalid_duration: int = 0
    total_duration_s: float = 0.0
    text_lengths: list[int] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    chars_per_second: list[float] = field(default_factory=list)
    fingerprints: Counter[str] = field(default_factory=Counter)
    categorical: dict[str, Counter[str]] = field(default_factory=dict)

    def add(
        self,
        text: Any,
        duration: Any = None,
        categories: Mapping[str, Any] | None = None,
    ) -> list[str]:
        flags: list[str] = []
        normalized = normalize_text(text)
        self.rows += 1
        self.text_lengths.append(len(normalized))
        if not normalized:
            self.empty_text += 1
            flags.append("empty_text")
        else:
            self.fingerprints[fingerprint_text(normalized)] += 1

        if duration is not None:
            try:
                seconds = float(duration)
            except (TypeError, ValueError):
                seconds = -1.0
            if seconds <= 0:
                self.invalid_duration += 1
                flags.append("invalid_duration")
            else:
                self.durations.append(seconds)
                self.total_duration_s += seconds
                cps = len(normalized) / seconds
                self.chars_per_second.append(cps)
                if seconds < 0.5:
                    flags.append("too_short")
                if seconds > 30.0:
                    flags.append("too_long")
                if cps > 35.0:
                    flags.append("high_text_audio_ratio")

        for key, value in (categories or {}).items():
            bucket = self.categorical.setdefault(key, Counter())
            bucket[normalize_text(value) or "<missing>"] += 1
        return flags

    @staticmethod
    def _quantile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        fraction = pos - lo
        return ordered[lo] * (1 - fraction) + ordered[hi] * fraction

    def to_dict(self) -> dict[str, Any]:
        duplicate_rows = sum(count - 1 for count in self.fingerprints.values() if count > 1)
        return {
            "rows": self.rows,
            "empty_text": self.empty_text,
            "invalid_duration": self.invalid_duration,
            "total_hours": round(self.total_duration_s / 3600, 4),
            "exact_duplicate_rows": duplicate_rows,
            "exact_duplicate_rate": round(duplicate_rows / max(1, self.rows), 6),
            "text_length_chars": {
                "min": min(self.text_lengths) if self.text_lengths else None,
                "p50": self._quantile([float(x) for x in self.text_lengths], 0.5),
                "p95": self._quantile([float(x) for x in self.text_lengths], 0.95),
                "max": max(self.text_lengths) if self.text_lengths else None,
                "mean": round(mean(self.text_lengths), 3) if self.text_lengths else None,
            },
            "duration_seconds": {
                "min": min(self.durations) if self.durations else None,
                "p50": self._quantile(self.durations, 0.5),
                "p95": self._quantile(self.durations, 0.95),
                "max": max(self.durations) if self.durations else None,
                "mean": round(mean(self.durations), 3) if self.durations else None,
                "median": round(median(self.durations), 3) if self.durations else None,
            },
            "chars_per_second": {
                "p50": self._quantile(self.chars_per_second, 0.5),
                "p95": self._quantile(self.chars_per_second, 0.95),
                "max": max(self.chars_per_second) if self.chars_per_second else None,
            },
            "categorical": {
                key: dict(counter.most_common()) for key, counter in sorted(self.categorical.items())
            },
        }


def cross_split_leakage(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Detect exact-text, speaker, and recording-group overlap across roles."""
    seen: dict[str, dict[str, set[str]]] = {
        "text": {},
        "speaker": {},
        "group": {},
    }
    for record in records:
        role = str(record.get("role", record.get("split", "unknown")))
        values = {
            "text": str(record.get("text_fingerprint", "")),
            "speaker": normalize_text(record.get("speaker")),
            "group": normalize_text(record.get("group")),
        }
        for dimension, value in values.items():
            if value:
                seen[dimension].setdefault(value, set()).add(role)

    result: dict[str, Any] = {}
    for dimension, index in seen.items():
        overlaps = {key: sorted(roles) for key, roles in index.items() if len(roles) > 1}
        role_pairs: Counter[str] = Counter()
        for roles in overlaps.values():
            for left_index, left in enumerate(roles):
                for right in roles[left_index + 1 :]:
                    role_pairs[f"{left}<->{right}"] += 1
        result[dimension] = {
            "overlap_count": len(overlaps),
            "role_pair_counts": dict(sorted(role_pairs.items())),
            "examples": dict(list(overlaps.items())[:20]),
        }
    return result


def write_json(path: Any, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

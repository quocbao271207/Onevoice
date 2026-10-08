"""Conservative text-level guards for safety-critical medical translation."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Mapping


NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\s*\d+)?(?!\w)")
COMPACT_IDENTIFIER_RE = re.compile(r"(?i)\b([a-z]+)(\d+)\b")
SPACED_IDENTIFIER_RE = re.compile(r"(?i)\b([BT])\s+(\d+)\b")
HYPHENATED_IDENTIFIER_RE = re.compile(r"\b([A-Z]{2,})-(\d+)\b")
QUANTITY_RE = re.compile(
    r"(?i)(?<!\w)(\d+(?:[.,]\s*\d+)?)\s*"
    r"(mcg/kg/min|µg/kg/min|mg/kg/hour|mg/kg/giờ|mg/kg/h|units/kg/hour|units/kg/h|đơn vị/kg/giờ|đơn vị/kg/h|"
    r"mmol/l|mg/kg|mcg/kg|µg/kg|ml/hour|ml/giờ|ml/h|l/min|"
    r"milligrams?|micrograms?|grams?|kilograms?|milliliters?|liters?|"
    r"mg|mcg|µg|g|kg|ml|l|mmhg|bpm|iu|units?|đơn vị|%|°c|độ c|"
    r"tablets?|viên|hours?|hrs?|giờ|days?|ngày)(?!\w)"
)
VI_NEGATION_RE = re.compile(r"(?i)\b(?:không được|không có|không phải|không|chưa|chẳng|đừng)\b")
EN_NEGATION_RE = re.compile(
    r"(?i)\b(?:does not|do not|did not|is not|are not|was not|were not|"
    r"doesn't|don't|didn't|isn't|aren't|wasn't|weren't|no|not|never|without)\b"
)
VI_IMPLICIT_NEGATION_RE = re.compile(r"(?i)\b(?:bất|vô|thiếu|âm tính)\w*\b")
EN_IMPLICIT_NEGATION_RE = re.compile(
    r"(?i)\b(?:unequal|uneven|absent|absence|negative|impossible|unable|lack|lacking)\b"
)
NUMBER_WORDS = {
    "vi": {
        "một": "1", "hai": "2", "ba": "3", "bốn": "4", "tư": "4",
        "năm": "5", "sáu": "6", "bảy": "7", "tám": "8", "chín": "9", "mười": "10",
    },
    "en": {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    },
}

UNIT_ALIASES = {
    "µg": "mcg",
    "µg/kg": "mcg/kg",
    "µg/kg/min": "mcg/kg/min",
    "units": "unit",
    "đơn vị": "unit",
    "units/kg/hour": "unit/kg/h",
    "đơn vị/kg/giờ": "unit/kg/h",
    "units/kg/h": "unit/kg/h",
    "đơn vị/kg/h": "unit/kg/h",
    "ml/hour": "ml/h",
    "ml/giờ": "ml/h",
    "mg/kg/hour": "mg/kg/h",
    "mg/kg/giờ": "mg/kg/h",
    "milligram": "mg",
    "milligrams": "mg",
    "microgram": "mcg",
    "micrograms": "mcg",
    "gram": "g",
    "grams": "g",
    "kilogram": "kg",
    "kilograms": "kg",
    "milliliter": "ml",
    "milliliters": "ml",
    "liter": "l",
    "liters": "l",
    "hour": "h",
    "hours": "h",
    "hr": "h",
    "hrs": "h",
    "giờ": "h",
    "day": "day",
    "days": "day",
    "ngày": "day",
    "tablet": "tablet",
    "tablets": "tablet",
    "viên": "tablet",
}


def _canonical_number(value: str) -> str:
    return unicodedata.normalize("NFKC", value).replace(" ", "").replace(",", ".")


def _normalized_identifiers(text: str) -> str:
    """Collapse medical identifiers such as T 3, T3, B12, and COVID-19."""
    value = HYPHENATED_IDENTIFIER_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}", text or "")
    value = SPACED_IDENTIFIER_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}", value)
    return value


def extract_identifiers(text: str) -> list[str]:
    normalized = _normalized_identifiers(text)
    return [
        f"{prefix.casefold()}{digits}"
        for prefix, digits in COMPACT_IDENTIFIER_RE.findall(normalized)
    ]


def extract_numbers(text: str, language: str | None = None) -> list[str]:
    normalized = _normalized_identifiers(text)
    return [_canonical_number(value) for value in NUMBER_RE.findall(normalized)]


def number_word_counts(text: str, language: str) -> Counter[str]:
    mapping = NUMBER_WORDS.get(language, {})
    return Counter(
        mapping[word]
        for word in re.findall(r"\b\w+\b", (text or "").casefold(), flags=re.UNICODE)
        if word in mapping
    )


def reconciled_number_counts(source: str, target: str, source_lang: str, target_lang: str) -> tuple[Counter[str], Counter[str]]:
    """Accept digit↔word rendering only when it explains an actual missing digit."""
    source_counts = Counter(extract_numbers(source))
    target_counts = Counter(extract_numbers(target))
    source_words = number_word_counts(source, source_lang)
    target_words = number_word_counts(target, target_lang)
    for value in set(source_counts) | set(target_counts):
        if source_counts[value] > target_counts[value]:
            target_counts[value] += min(source_counts[value] - target_counts[value], target_words[value])
        elif target_counts[value] > source_counts[value]:
            source_counts[value] += min(target_counts[value] - source_counts[value], source_words[value])
    return source_counts, target_counts


def _canonical_unit(value: str) -> str:
    folded = value.casefold().strip().replace(" ", " ")
    return UNIT_ALIASES.get(folded, folded.replace("µg", "mcg"))


def _quantity_scan_text(text: str) -> str:
    normalized = re.sub(r"\s*/\s*", "/", text or "")
    normalized = re.sub(r"(?i)\s+per\s+", "/", normalized)
    word_numbers = {**NUMBER_WORDS["en"], **NUMBER_WORDS["vi"]}
    return re.sub(
        r"\b(?:" + "|".join(map(re.escape, sorted(word_numbers, key=len, reverse=True))) + r")\b",
        lambda match: word_numbers[match.group(0).casefold()],
        normalized,
        flags=re.IGNORECASE,
    )


def _quantity_spans(text: str) -> tuple[str, list[tuple[tuple[str, str], int, int]]]:
    normalized = _quantity_scan_text(text)
    return normalized, [
        (
            (_canonical_number(match.group(1)), _canonical_unit(match.group(2))),
            match.start(),
            match.end(),
        )
        for match in QUANTITY_RE.finditer(normalized)
    ]


def extract_quantities(text: str) -> list[tuple[str, str]]:
    """Return number-unit pairs so swapped dosages cannot pass as equal multisets."""
    _, spans = _quantity_spans(text)
    return [quantity for quantity, _, _ in spans]


def extract_units(text: str) -> list[str]:
    return [unit for _, unit in extract_quantities(text)]


def has_negation(text: str, language: str) -> bool:
    return negation_count(text, language) > 0


def negation_count(text: str, language: str) -> int:
    """Count explicit and implicit negations so one dropped clause cannot hide another."""
    if language == "vi":
        return len(VI_NEGATION_RE.findall(text or "")) + len(VI_IMPLICIT_NEGATION_RE.findall(text or ""))
    if language == "en":
        return len(EN_NEGATION_RE.findall(text or "")) + len(EN_IMPLICIT_NEGATION_RE.findall(text or ""))
    raise ValueError(f"Unsupported language for negation check: {language}")


def _phrase_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    words = [word for word in re.split(r"\s+", phrase.strip()) if word]
    if not words:
        return []
    pattern = re.compile(
        r"(?<!\w)" + r"\s+".join(re.escape(word) for word in words) + r"(?!\w)",
        flags=re.IGNORECASE,
    )
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def _terminology_anchors(
    text: str,
    terminology: Mapping[str, str | Sequence[str]],
    *,
    source_side: bool,
) -> list[tuple[str, int, int]]:
    anchors: list[tuple[str, int, int]] = []
    for term, expected in terminology.items():
        alternatives = [expected] if isinstance(expected, str) else list(expected)
        phrases = [term] if source_side else alternatives
        key = term.casefold()
        for phrase in phrases:
            if not isinstance(phrase, str):
                continue
            anchors.extend(
                (key, start, end) for start, end in _phrase_spans(text, phrase)
            )
    return anchors


def _span_distance(start: int, end: int, anchor_start: int, anchor_end: int) -> int:
    if end <= anchor_start:
        return anchor_start - end
    if anchor_end <= start:
        return start - anchor_end
    return 0


def _quantity_bindings(
    spans: list[tuple[tuple[str, str], int, int]],
    anchors: list[tuple[str, int, int]],
) -> tuple[dict[str, Counter[tuple[str, str]]], bool]:
    bindings: defaultdict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    ambiguous = False
    for quantity, start, end in spans:
        if not anchors:
            break
        distances = [
            (_span_distance(start, end, anchor_start, anchor_end), key)
            for key, anchor_start, anchor_end in anchors
        ]
        minimum = min(distance for distance, _ in distances)
        nearest_keys = {key for distance, key in distances if distance == minimum}
        if len(nearest_keys) != 1:
            ambiguous = True
            continue
        bindings[next(iter(nearest_keys))][quantity] += 1
    return dict(bindings), ambiguous


def _negation_spans(text: str, language: str) -> list[tuple[int, int]]:
    if language == "vi":
        patterns = (VI_NEGATION_RE, VI_IMPLICIT_NEGATION_RE)
    elif language == "en":
        patterns = (EN_NEGATION_RE, EN_IMPLICIT_NEGATION_RE)
    else:
        raise ValueError(f"Unsupported language for negation check: {language}")
    spans = {
        (match.start(), match.end())
        for pattern in patterns
        for match in pattern.finditer(text or "")
    }
    return sorted(spans)


def _clause_spans(text: str) -> list[tuple[int, int]]:
    """Return strong scope boundaries without splitting ordinary drug lists."""
    boundaries = re.compile(r"[.;!?]+|\b(?:but|nhưng)\b", flags=re.IGNORECASE)
    spans: list[tuple[int, int]] = []
    start = 0
    for match in boundaries.finditer(text):
        if start < match.start():
            spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def _negation_bindings(
    text: str,
    language: str,
    anchors: list[tuple[str, int, int]],
) -> Counter[str]:
    clauses = _clause_spans(text)
    negations = _negation_spans(text, language)
    clause_negations = [
        sum(start <= negation_start < end for negation_start, _ in negations)
        for start, end in clauses
    ]
    clauses_by_term: defaultdict[str, set[int]] = defaultdict(set)
    for key, anchor_start, _ in anchors:
        for index, (start, end) in enumerate(clauses):
            if start <= anchor_start < end:
                clauses_by_term[key].add(index)
                break
    return Counter(
        {
            key: sum(clause_negations[index] for index in indices)
            for key, indices in clauses_by_term.items()
        }
    )


def _terminology_binding_issues(
    source: str,
    target: str,
    source_lang: str,
    target_lang: str,
    terminology: Mapping[str, str | Sequence[str]],
) -> list[str]:
    source_quantity_text, source_quantity_spans = _quantity_spans(source)
    target_quantity_text, target_quantity_spans = _quantity_spans(target)
    source_quantity_anchors = _terminology_anchors(
        source_quantity_text, terminology, source_side=True
    )
    target_quantity_anchors = _terminology_anchors(
        target_quantity_text, terminology, source_side=False
    )
    source_quantities, source_ambiguous = _quantity_bindings(
        source_quantity_spans, source_quantity_anchors
    )
    target_quantities, target_ambiguous = _quantity_bindings(
        target_quantity_spans, target_quantity_anchors
    )

    issues: list[str] = []
    if source_ambiguous or target_ambiguous:
        issues.append("quantity_binding_ambiguous")
    quantity_keys = {key for key, _, _ in source_quantity_anchors} & {
        key for key, _, _ in target_quantity_anchors
    }
    for key in sorted(quantity_keys):
        source_values = source_quantities.get(key, Counter())
        target_values = target_quantities.get(key, Counter())
        if source_values != target_values:
            issues.append(
                f"quantity_binding_mismatch:{key}:"
                f"{list(source_values.elements())}->{list(target_values.elements())}"
            )

    source_anchors = _terminology_anchors(source, terminology, source_side=True)
    target_anchors = _terminology_anchors(target, terminology, source_side=False)
    source_negations = _negation_bindings(source, source_lang, source_anchors)
    target_negations = _negation_bindings(target, target_lang, target_anchors)
    negation_keys = {key for key, _, _ in source_anchors} & {
        key for key, _, _ in target_anchors
    }
    for key in sorted(negation_keys):
        if source_negations[key] != target_negations[key]:
            issues.append(
                f"negation_binding_mismatch:{key}:"
                f"{source_negations[key]}->{target_negations[key]}"
            )
    return issues


@dataclass
class SafetyCheck:
    safe: bool
    issues: list[str] = field(default_factory=list)


def validate_translation(
    source: str,
    target: str,
    source_lang: str,
    target_lang: str,
    terminology: Mapping[str, str | Sequence[str]] | None = None,
) -> SafetyCheck:
    issues: list[str] = []
    if not target.strip():
        issues.append("empty_translation")
    source_numbers, target_numbers = reconciled_number_counts(source, target, source_lang, target_lang)
    if source_numbers != target_numbers:
        issues.append(f"number_mismatch:{list(source_numbers.elements())}->{list(target_numbers.elements())}")

    source_units = Counter(extract_units(source))
    target_units = Counter(extract_units(target))
    if source_units != target_units:
        issues.append(f"unit_mismatch:{list(source_units.elements())}->{list(target_units.elements())}")

    source_quantities = Counter(extract_quantities(source))
    target_quantities = Counter(extract_quantities(target))
    if source_quantities != target_quantities:
        issues.append(
            f"quantity_mismatch:{list(source_quantities.elements())}->{list(target_quantities.elements())}"
        )

    source_identifiers = Counter(extract_identifiers(source))
    target_identifiers = Counter(extract_identifiers(target))
    if source_identifiers != target_identifiers:
        issues.append(
            f"identifier_mismatch:{list(source_identifiers.elements())}->{list(target_identifiers.elements())}"
        )

    source_negations = negation_count(source, source_lang)
    target_negations = negation_count(target, target_lang)
    if source_negations != target_negations:
        issues.append("negation_mismatch")

    if terminology:
        issues.extend(
            _terminology_binding_issues(
                source,
                target,
                source_lang,
                target_lang,
                terminology,
            )
        )

    source_folded = source.casefold()
    target_folded = target.casefold()
    for term, expected in (terminology or {}).items():
        alternatives = [expected] if isinstance(expected, str) else list(expected)
        if term.casefold() in source_folded and not any(
            alternative.casefold() in target_folded for alternative in alternatives
        ):
            issues.append(f"terminology_missing:{term}->{alternatives}")
    return SafetyCheck(safe=not issues, issues=issues)

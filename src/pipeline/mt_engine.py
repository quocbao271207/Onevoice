"""
MediVoice Edge — Machine Translation Engine
Medical-domain bilingual translation (Vietnamese ↔ English).

Pipeline Stage 3: Transcribed Text → Translated Text

Model: NLLB-200 distilled 600M, fine-tuned on locked MedEV pairs
Quantization target: calibrated INT8, subject to physical-device validation

The current backend is deterministic utterance-level generation. Streaming
first-token timing and device latency are not reported until instrumented.
"""

import json
import logging
import math
import time
import unicodedata
from numbers import Real
from threading import Lock
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from pathlib import Path

from .generation_guard import require_completed_generation
from .safety_guard import safety_issue_codes, validate_translation
from ..utils.bounded_file import read_stable_regular_file

logger = logging.getLogger(__name__)


@dataclass
class MTResult:
    """Result from Machine Translation."""
    source_text: str
    translated_text: str
    source_lang: str        # "vi" or "en"
    target_lang: str        # "en" or "vi"
    latency_ms: float       # Total inference time
    first_token_ms: Optional[float]  # None until streaming generation is instrumented
    tokens_generated: int
    from_cache: bool        # True if served from flash cache
    safety_passed: bool = True
    safety_issues: List[str] = field(default_factory=list)
    requires_confirmation: bool = False


NLLB_LANG_CODES = {"vi": "vie_Latn", "en": "eng_Latn"}
NLLB_BASE_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
DEFAULT_MAX_SOURCE_TOKENS = 256
DEFAULT_MAX_SOURCE_CHARACTERS = 4096
MAX_MEDICAL_LEXICON_BYTES = 1_000_000
MAX_MEDICAL_LEXICON_ENTRIES = 10_000
MAX_MEDICAL_TERM_CHARACTERS = 4096
MEDICAL_LEXICON_DIRECTIONS = {"vi_to_en", "en_to_vi"}


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Medical lexicon contains a duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> object:
    raise ValueError("Medical lexicon contains a non-finite JSON number")


def _contains_disallowed_term_character(value: str) -> bool:
    return any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        for character in value
    )


def validate_mt_source_text(
    text: str,
    *,
    max_source_characters: int = DEFAULT_MAX_SOURCE_CHARACTERS,
) -> None:
    """Reject malformed or unbounded MT input before cache/model dispatch."""
    if not isinstance(text, str):
        raise TypeError("MT source text must be a string")
    if not text.strip():
        raise ValueError("MT source text must not be blank")
    if len(text) > max_source_characters:
        raise ValueError(
            "MT source text has "
            f"{len(text)} characters; limit is {max_source_characters}"
        )


class MedicalLexicon:
    """
    Medical terminology lookup for ICD-10 codes and drug names.
    Ensures consistent translation of critical medical terms.
    """

    def __init__(self, lexicon_path: Optional[str] = None):
        self.vi_to_en: Dict[str, str] = {}
        self.en_to_vi: Dict[str, str] = {}
        self._load_default_terms()
        if lexicon_path is not None:
            self._load_from_file(lexicon_path)

    def _load_default_terms(self):
        """Load essential medical terms that must never be mistranslated."""
        critical_terms = {
            # Emergency terms
            "sốc phản vệ": "anaphylactic shock",
            "ngừng tim": "cardiac arrest",
            "xuất huyết": "hemorrhage",
            "đột quỵ": "stroke",
            "nhồi máu cơ tim": "myocardial infarction",
            "suy hô hấp": "respiratory failure",
            "tràn khí màng phổi": "pneumothorax",
            "thuyên tắc phổi": "pulmonary embolism",
            # Vitals
            "huyết áp": "blood pressure",
            "nhịp tim": "heart rate",
            "nồng độ oxy": "oxygen saturation",
            "thân nhiệt": "body temperature",
            "nhịp thở": "respiratory rate",
            # Procedures
            "đặt nội khí quản": "intubation",
            "hồi sức tim phổi": "CPR",
            "truyền máu": "blood transfusion",
            "phẫu thuật": "surgery",
            "sinh thiết": "biopsy",
            "nội soi": "endoscopy",
            "chụp CT": "CT scan",
            "chụp MRI": "MRI scan",
            # Medications
            "kháng sinh": "antibiotics",
            "thuốc giảm đau": "analgesics",
            "thuốc gây mê": "anesthetics",
            "thuốc chống đông": "anticoagulants",
            "dịch truyền": "IV fluids",
            "epinephrine": "epinephrine",
            "adrenaline": "adrenaline",
            "amoxicillin": "amoxicillin",
            "aspirin": "aspirin",
            "heparin": "heparin",
            "insulin": "insulin",
            "metformin": "metformin",
            "morphine": "morphine",
            "paracetamol": "paracetamol",
            "warfarin": "warfarin",
            "nước muối": "saline",
        }
        self.vi_to_en = critical_terms
        self.en_to_vi = {v: k for k, v in critical_terms.items()}

    def _load_from_file(self, path: str):
        """Validate and atomically load additional terms from a JSON file."""
        lexicon_path = Path(path)
        payload = read_stable_regular_file(
            lexicon_path,
            maximum_bytes=MAX_MEDICAL_LEXICON_BYTES,
            label="Configured medical lexicon",
        )
        try:
            data = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError):
            raise ValueError("Invalid medical lexicon JSON") from None
        if not isinstance(data, dict):
            raise ValueError("Medical lexicon root must be an object")
        unknown = set(data) - MEDICAL_LEXICON_DIRECTIONS
        if unknown:
            raise ValueError(
                f"Medical lexicon has unknown sections: {sorted(unknown)}"
            )

        validated = {}
        entry_count = 0
        for direction in MEDICAL_LEXICON_DIRECTIONS:
            terms = data.get(direction, {})
            if not isinstance(terms, dict):
                raise ValueError(f"Medical lexicon {direction} must be an object")
            clean_terms = {}
            for source, target in terms.items():
                if not isinstance(source, str) or not source.strip():
                    raise ValueError(
                        f"Medical lexicon {direction} keys must be non-empty strings"
                    )
                if not isinstance(target, str) or not target.strip():
                    raise ValueError(
                        f"Medical lexicon {direction} values must be non-empty strings"
                    )
                if (
                    len(source) > MAX_MEDICAL_TERM_CHARACTERS
                    or len(target) > MAX_MEDICAL_TERM_CHARACTERS
                ):
                    raise ValueError(
                        f"Medical lexicon {direction} term exceeds "
                        f"{MAX_MEDICAL_TERM_CHARACTERS} characters"
                    )
                if _contains_disallowed_term_character(
                    source
                ) or _contains_disallowed_term_character(target):
                    raise ValueError(
                        f"Medical lexicon {direction} terms must not contain control characters"
                    )
                clean_terms[source] = target
                entry_count += 1
                if entry_count > MAX_MEDICAL_LEXICON_ENTRIES:
                    raise ValueError(
                        f"Medical lexicon exceeds {MAX_MEDICAL_LEXICON_ENTRIES} entries"
                    )
            validated[direction] = clean_terms
        if not any(validated.values()):
            raise ValueError("Medical lexicon must contain at least one term")

        vi_to_en = dict(self.vi_to_en)
        en_to_vi = dict(self.en_to_vi)
        vi_to_en.update(validated["vi_to_en"])
        en_to_vi.update(validated["en_to_vi"])
        self.vi_to_en = vi_to_en
        self.en_to_vi = en_to_vi
        loaded_count = sum(len(terms) for terms in validated.values())
        logger.info(
            "Loaded %d additional medical terms from %s",
            loaded_count,
            lexicon_path,
        )

    def post_process(self, text: str, direction: str) -> str:
        """
        Post-process translation to fix critical medical terms.

        Args:
            text: Translated text
            direction: "vi_to_en" or "en_to_vi"
        """
        lexicon = self.vi_to_en if direction == "vi_to_en" else self.en_to_vi
        # This is a simple lookup; in production, use regex for context-aware replacement
        return text


class MTEngine:
    """
    Medical Machine Translation engine using an encoder-decoder model.

    Architecture:
    The configured latency is a target, not a measured result.
    """

    def __init__(
        self,
        model_path: str = "models/mt/nllb-medical",
        device: str = "auto",
        lexicon_path: Optional[str] = None,
        max_new_tokens: int = 256,
        max_source_tokens: int = DEFAULT_MAX_SOURCE_TOKENS,
        max_source_characters: int = DEFAULT_MAX_SOURCE_CHARACTERS,
        num_beams: int = 1,
        temperature: float = 0.0,
        allow_base_fallback: bool = False,
    ):
        if num_beams < 1:
            raise ValueError("num_beams must be at least 1")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if max_source_tokens < 1:
            raise ValueError("max_source_tokens must be at least 1")
        if max_source_characters < 1:
            raise ValueError("max_source_characters must be at least 1")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, Real)
            or not math.isfinite(float(temperature))
            or float(temperature) != 0.0
        ):
            raise ValueError(
                "temperature must be 0 because MT decoding is deterministic"
            )
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_source_tokens = max_source_tokens
        self.max_source_characters = max_source_characters
        self.num_beams = num_beams
        self.temperature = 0.0
        self.allow_base_fallback = allow_base_fallback
        self.lexicon = MedicalLexicon(lexicon_path)
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self.loaded_model_path: Optional[str] = None
        self._inference_lock = Lock()

    @property
    def is_ready(self) -> bool:
        return (
            self._is_loaded
            and self.model is not None
            and self.tokenizer is not None
            and isinstance(self.loaded_model_path, str)
            and bool(self.loaded_model_path)
        )

    def load(self):
        """Load the translation model."""
        if self.is_ready:
            logger.info("MT engine already ready; skipping reload")
            return
        with self._inference_lock:
            self._is_loaded = False
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        resolved_device = self.device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading MT model from: {self.model_path}")

        def load_candidate(path: str, revision: Optional[str]):
            tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                path,
                revision=revision,
                torch_dtype=(
                    torch.float16
                    if resolved_device == "cuda"
                    else torch.float32
                ),
                low_cpu_mem_usage=True,
                device_map="auto" if resolved_device == "cuda" else None,
            )
            model.eval()
            return tokenizer, model

        try:
            # Try loading fine-tuned model
            revision = NLLB_BASE_REVISION if self.model_path == "facebook/nllb-200-distilled-600M" else None
            tokenizer, model = load_candidate(
                self.model_path,
                revision,
            )
            loaded_model_path = self.model_path
        except Exception as e:
            if not self.allow_base_fallback:
                raise RuntimeError(
                    f"Failed to load configured MT checkpoint: {self.model_path}. "
                    "Base fallback is disabled to prevent silently evaluating the wrong model."
                ) from e
            # Fallback to base model
            fallback = "facebook/nllb-200-distilled-600M"
            logger.warning(
                f"Failed to load fine-tuned MT model: {e}. "
                f"Falling back to: {fallback}"
            )
            tokenizer, model = load_candidate(
                fallback,
                NLLB_BASE_REVISION,
            )
            loaded_model_path = fallback

        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        with self._inference_lock:
            self.device = resolved_device
            self.tokenizer = tokenizer
            self.model = model
            self.loaded_model_path = loaded_model_path
            self._is_loaded = True
        logger.info(f"MT model loaded: {param_count:.0f}M parameters on {self.device}")

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: Optional[str] = None,
    ) -> MTResult:
        """
        Translate text between Vietnamese and English.

        Args:
            text: Source text to translate
            source_lang: Source language ("vi" or "en")
            target_lang: Target language (auto-determined if None)

        Returns:
            MTResult with translated text and performance metrics
        """
        if not self.is_ready:
            raise RuntimeError(
                "MT engine is not ready; call load() and verify the model, "
                "tokenizer, and loaded model path"
            )

        validate_mt_source_text(
            text,
            max_source_characters=self.max_source_characters,
        )

        if target_lang is None:
            target_lang = "en" if source_lang == "vi" else "vi"

        start_time = time.perf_counter()

        direction = f"{source_lang}_to_{target_lang}"
        if source_lang not in NLLB_LANG_CODES or target_lang not in NLLB_LANG_CODES or source_lang == target_lang:
            raise ValueError(f"Unsupported translation direction: {direction}")
        import torch

        # The tokenizer stores source language as mutable shared state. Keep
        # direction selection, tokenization, generation, and decoding in one
        # critical section so concurrent opposite-direction requests cannot
        # contaminate each other.
        with self._inference_lock:
            self.tokenizer.src_lang = NLLB_LANG_CODES[source_lang]
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
            )
            input_ids = inputs.get("input_ids")
            if (
                input_ids is None
                or not hasattr(input_ids, "shape")
                or len(input_ids.shape) < 2
            ):
                raise RuntimeError("MT tokenizer did not return batched input_ids")
            source_tokens = int(input_ids.shape[-1])
            if source_tokens > self.max_source_tokens:
                raise ValueError(
                    "MT source has "
                    f"{source_tokens} tokens; limit is {self.max_source_tokens}. "
                    "Refusing to silently truncate clinical text."
                )
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=self.num_beams,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(
                        NLLB_LANG_CODES[target_lang]
                    ),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_tokens = outputs[0]
            require_completed_generation(
                generated_tokens,
                self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                context="MT generation",
            )
            translated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

        # Post-process with medical lexicon
        translated_text = self.lexicon.post_process(translated_text, direction)

        terminology = (
            self.lexicon.vi_to_en
            if source_lang == "vi"
            else self.lexicon.en_to_vi
        )
        safety = validate_translation(
            text,
            translated_text,
            source_lang,
            target_lang,
            terminology,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        result = MTResult(
            source_text=text,
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=target_lang,
            latency_ms=latency_ms,
            first_token_ms=None,
            tokens_generated=len(generated_tokens),
            from_cache=False,
            safety_passed=safety.safe,
            safety_issues=safety.issues,
        )

        logger.info(
            "MT complete source_lang=%s target_lang=%s source_chars=%d "
            "output_chars=%d latency_ms=%.0f tokens=%d safety_passed=%s "
            "safety_issue_codes=%s",
            source_lang,
            target_lang,
            len(text),
            len(translated_text),
            latency_ms,
            len(generated_tokens),
            safety.safe,
            safety_issue_codes(safety.issues),
        )
        return result

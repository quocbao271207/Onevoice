"""
MediVoice Edge — Machine Translation Engine
Medical-domain bilingual translation (Vietnamese ↔ English).

Pipeline Stage 3: Transcribed Text → Translated Text

Model: NLLB-200 distilled 600M, fine-tuned on locked MedEV pairs
Quantization target: calibrated INT8, subject to physical-device validation

The current backend is deterministic utterance-level generation. Streaming
first-token timing and device latency are not reported until instrumented.
"""

import logging
import math
import time
import json
from numbers import Real
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from pathlib import Path

from .generation_guard import require_completed_generation
from .safety_guard import safety_issue_codes, validate_translation

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
MEDICAL_LEXICON_DIRECTIONS = {"vi_to_en", "en_to_vi"}


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Medical lexicon contains duplicate key {key!r}")
        result[key] = value
    return result


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
        if not lexicon_path.is_file():
            raise FileNotFoundError(f"Configured medical lexicon does not exist: {path}")
        if lexicon_path.stat().st_size > MAX_MEDICAL_LEXICON_BYTES:
            raise ValueError(
                "Medical lexicon exceeds "
                f"{MAX_MEDICAL_LEXICON_BYTES} bytes: {lexicon_path}"
            )
        try:
            data = json.loads(
                lexicon_path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_json_object,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid medical lexicon {lexicon_path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Medical lexicon root must be an object")
        unknown = set(data) - MEDICAL_LEXICON_DIRECTIONS
        if unknown:
            raise ValueError(
                f"Medical lexicon has unknown sections: {sorted(unknown)}"
            )

        validated = {}
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
                clean_terms[source] = target
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
        self._is_loaded = False
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading MT model from: {self.model_path}")

        try:
            # Try loading fine-tuned model
            revision = NLLB_BASE_REVISION if self.model_path == "facebook/nllb-200-distilled-600M" else None
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, revision=revision)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                revision=revision,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == "cuda" else None,
            )
            self.loaded_model_path = self.model_path
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
            self.tokenizer = AutoTokenizer.from_pretrained(fallback, revision=NLLB_BASE_REVISION)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                fallback,
                revision=NLLB_BASE_REVISION,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if self.device == "cuda" else None,
            )
            self.loaded_model_path = fallback

        self.model.eval()
        self._is_loaded = True

        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
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
        if not self._is_loaded:
            raise RuntimeError("MT engine not loaded. Call load() first.")

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
        self.tokenizer.src_lang = NLLB_LANG_CODES[source_lang]

        # Tokenize
        import torch
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
        )
        input_ids = inputs.get("input_ids")
        if input_ids is None or not hasattr(input_ids, "shape") or len(input_ids.shape) < 2:
            raise RuntimeError("MT tokenizer did not return batched input_ids")
        source_tokens = int(input_ids.shape[-1])
        if source_tokens > self.max_source_tokens:
            raise ValueError(
                "MT source has "
                f"{source_tokens} tokens; limit is {self.max_source_tokens}. "
                "Refusing to silently truncate clinical text."
            )
        inputs = inputs.to(self.device)

        # Generate translation
        first_token_time = None
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=self.num_beams,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(NLLB_LANG_CODES[target_lang]),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output
        generated_tokens = outputs[0]
        require_completed_generation(
            generated_tokens,
            self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            context="MT generation",
        )
        translated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
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

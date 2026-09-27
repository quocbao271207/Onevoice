"""
MediVoice Edge — Evaluation Metrics
Benchmark tools for measuring ASR, MT, TTS, and pipeline performance.

Metrics:
- WER (Word Error Rate) for ASR — target < 15%
- BLEU for MT — target > 35
- COMET for MT — target > 0.80
- MOS (Mean Opinion Score) for TTS — target > 4.25
- RTF (Real-Time Factor) — target < 0.55
- End-to-end Latency — target < 1400ms
"""

import time
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Complete benchmark result across all pipeline stages."""
    # ASR metrics
    wer_vi: float = 0.0
    wer_en: float = 0.0
    # MT metrics
    bleu_vi_to_en: float = 0.0
    bleu_en_to_vi: float = 0.0
    comet_vi_to_en: float = 0.0
    comet_en_to_vi: float = 0.0
    # TTS metrics
    mos_vi: float = 0.0
    mos_en: float = 0.0
    # Latency metrics
    avg_asr_latency_ms: float = 0.0
    avg_mt_latency_ms: float = 0.0
    avg_tts_latency_ms: float = 0.0
    avg_total_latency_ms: float = 0.0
    avg_rtf: float = 0.0
    p95_latency_ms: float = 0.0
    # Sample counts
    num_samples: int = 0


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute Word Error Rate (WER) between reference and hypothesis texts.

    WER = (Substitutions + Insertions + Deletions) / Reference Length

    Args:
        references: List of ground truth transcriptions
        hypotheses: List of ASR output transcriptions

    Returns:
        WER score (0.0 = perfect, 1.0 = 100% error)
    """
    try:
        from jiwer import wer
        return wer(references, hypotheses)
    except ImportError:
        logger.warning("jiwer not installed. Computing WER manually.")
        return _manual_wer(references, hypotheses)


def _manual_wer(references: List[str], hypotheses: List[str]) -> float:
    """Manual WER computation using edit distance."""
    total_errors = 0
    total_words = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.lower().split()
        hyp_words = hyp.lower().split()
        total_words += len(ref_words)

        # Levenshtein distance at word level
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(
                        d[i - 1][j] + 1,      # Deletion
                        d[i][j - 1] + 1,      # Insertion
                        d[i - 1][j - 1] + 1,  # Substitution
                    )
        total_errors += d[len(ref_words)][len(hyp_words)]

    return total_errors / max(1, total_words)


def compute_bleu(
    references: List[str],
    hypotheses: List[str],
    tokenize: str = "intl",
) -> float:
    """
    Compute BLEU score for machine translation evaluation.

    Args:
        references: List of reference translations
        hypotheses: List of MT output translations
        tokenize: Tokenization method

    Returns:
        BLEU score (0-100 scale)
    """
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenize)
        return bleu.score
    except ImportError:
        logger.warning("sacrebleu not installed. BLEU computation unavailable.")
        return 0.0


def compute_comet(
    sources: List[str],
    references: List[str],
    hypotheses: List[str],
    model_name: str = "Unbabel/wmt22-comet-da",
) -> float:
    """
    Compute COMET score for MT quality evaluation.
    COMET correlates better with human judgments than BLEU.

    Args:
        sources: List of source texts
        references: List of reference translations
        hypotheses: List of MT output translations
        model_name: COMET model to use

    Returns:
        COMET score (0.0 - 1.0, higher is better)
    """
    try:
        from comet import download_model, load_from_checkpoint
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)

        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        scores = model.predict(data, batch_size=8)
        return float(np.mean(scores.scores))
    except ImportError:
        logger.warning("COMET not installed. COMET computation unavailable.")
        return 0.0


def compute_rtf(
    audio_duration_s: float,
    processing_time_s: float,
) -> float:
    """
    Compute Real-Time Factor (RTF).

    RTF = Processing Time / Audio Duration
    RTF < 1.0 means faster than real-time (required by competition)

    Args:
        audio_duration_s: Duration of input audio in seconds
        processing_time_s: Total pipeline processing time in seconds

    Returns:
        RTF value (target < 0.55)
    """
    if audio_duration_s <= 0:
        return float('inf')
    return processing_time_s / audio_duration_s


def format_benchmark_report(result: BenchmarkResult) -> str:
    """
    Format a benchmark result into a readable report.

    Returns:
        Formatted string report
    """
    def status(value, target, higher_is_better=True):
        if higher_is_better:
            return "✅" if value >= target else "❌"
        else:
            return "✅" if value <= target else "❌"

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║            MEDIVOICE EDGE — BENCHMARK REPORT                ║
║            Samples: {result.num_samples}                                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📊 ASR (Automatic Speech Recognition)                       ║
║  ────────────────────────────────────────                     ║
║  WER Vietnamese:  {result.wer_vi:>6.1%}   (target < 15%)  {status(result.wer_vi, 0.15, False)}         ║
║  WER English:     {result.wer_en:>6.1%}   (target < 12%)  {status(result.wer_en, 0.12, False)}         ║
║                                                              ║
║  🌐 MT (Machine Translation)                                 ║
║  ────────────────────────────────────────                     ║
║  BLEU VI→EN:      {result.bleu_vi_to_en:>6.1f}    (target > 35)   {status(result.bleu_vi_to_en, 35)}         ║
║  BLEU EN→VI:      {result.bleu_en_to_vi:>6.1f}    (target > 35)   {status(result.bleu_en_to_vi, 35)}         ║
║  COMET VI→EN:     {result.comet_vi_to_en:>6.2f}   (target > 0.80) {status(result.comet_vi_to_en, 0.80)}         ║
║  COMET EN→VI:     {result.comet_en_to_vi:>6.2f}   (target > 0.80) {status(result.comet_en_to_vi, 0.80)}         ║
║                                                              ║
║  🔊 TTS (Text-to-Speech)                                     ║
║  ────────────────────────────────────────                     ║
║  MOS Vietnamese:  {result.mos_vi:>6.2f}   (target > 4.25) {status(result.mos_vi, 4.25)}         ║
║  MOS English:     {result.mos_en:>6.2f}   (target > 4.25) {status(result.mos_en, 4.25)}         ║
║                                                              ║
║  ⏱️  Latency                                                  ║
║  ────────────────────────────────────────                     ║
║  Avg ASR:         {result.avg_asr_latency_ms:>6.0f}ms  (target < 380ms)  {status(result.avg_asr_latency_ms, 380, False)}  ║
║  Avg MT:          {result.avg_mt_latency_ms:>6.0f}ms  (target < 520ms)  {status(result.avg_mt_latency_ms, 520, False)}  ║
║  Avg TTS:         {result.avg_tts_latency_ms:>6.0f}ms  (target < 280ms)  {status(result.avg_tts_latency_ms, 280, False)}  ║
║  Avg Total:       {result.avg_total_latency_ms:>6.0f}ms  (target < 1400ms) {status(result.avg_total_latency_ms, 1400, False)}  ║
║  P95 Total:       {result.p95_latency_ms:>6.0f}ms                         ║
║  Avg RTF:         {result.avg_rtf:>6.2f}   (target < 0.55) {status(result.avg_rtf, 0.55, False)}         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    return report

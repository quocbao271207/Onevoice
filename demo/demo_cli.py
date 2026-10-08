"""
MediVoice Edge — CLI Demo
Command-line demonstration of the complete speech-to-speech translation pipeline.

Usage:
    # Interactive microphone mode
    python demo/demo_cli.py --mode interactive

    # Translate an audio file
    python demo/demo_cli.py --mode file --input test_audio.wav

    # Text-only translation (no ASR/TTS)
    python demo/demo_cli.py --mode text --text "Bệnh nhân sốc phản vệ" --lang vi

    # Benchmark mode
    python demo/demo_cli.py --mode benchmark
"""

import argparse
import logging
import math
import sys
import time
from numbers import Integral, Real
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.audio_frontend import (
    MAX_AUDIO_CHANNELS,
    MAX_SAMPLE_RATE,
    MIN_SAMPLE_RATE,
)
from src.pipeline.safety_guard import safety_issue_codes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MediVoice")

MAX_CLI_AUDIO_FILE_BYTES = 64 * 1024 * 1024
MAX_CLI_DECODED_AUDIO_BYTES = 64 * 1024 * 1024
MAX_BENCHMARK_SAMPLES = 10


BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    __  __          _ _ __     __    _            _____     _  ║
║   |  \/  | ___  __| (_) \   / /__ (_) ___ ___  | ____|__| | ║
║   | |\/| |/ _ \/ _` | |\ \ / / _ \| |/ __/ _ \ |  _| / _` | ║
║   | |  | |  __/ (_| | | \ V / (_) | | (_|  __/ | |__| (_| | ║
║   |_|  |_|\___|\__,_|_|  \_/ \___/|_|\___\___| |_____\__,_| ║
║                                                              ║
║   Medical Voice Translation — Offline-First Prototype        ║
║   OneVoice AI Challenge 2026                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


def format_translation_for_display(result) -> str:
    """Never display a candidate that failed the clinical safety gate."""
    if not result.safety_passed:
        codes = ", ".join(safety_issue_codes(result.safety_issues)) or "unknown"
        return f"[BLOCKED BY CLINICAL SAFETY GATE: {codes}]"
    return result.translated_text


def print_confirmation_warning(result) -> None:
    if result.requires_confirmation:
        print("  ⚠️  Clinical action requires explicit confirmation before playback")


def print_speech_result(result) -> None:
    """Display a speech result before any interactive audio playback."""
    print(f"\n{'─'*60}")
    print(f"  🎤 Input ({result.asr_language.upper()}): {result.asr_text}")
    print(
        f"  🌐 Output ({result.target_language.upper()}): "
        f"{format_translation_for_display(result)}"
    )
    print(
        f"  ⏱️  Latency: {result.total_latency_ms:.0f}ms "
        f"(RTF={result.overall_rtf:.2f})"
    )
    print(
        "  📊 ASR token-probability proxy (uncalibrated): "
        f"{result.asr_confidence:.2%}"
    )
    print(f"  💾 From Cache: {'Yes ⚡' if result.from_cache else 'No'}")
    print_confirmation_warning(result)
    if getattr(result, "degraded_mode", None) == "text_only":
        code = getattr(result, "degradation_code", None) or "tts_unavailable"
        print(f"  🔇 Text-only fallback active ({code}); no audio was produced")
    print(f"{'─'*60}\n")


def load_audio_file(
    input_path: str | Path,
    *,
    max_duration_seconds: float,
) -> tuple[np.ndarray, int]:
    """Preflight bounded audio metadata before allocating decoded samples."""
    import soundfile as sf

    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio input does not exist: {path}")
    file_bytes = path.stat().st_size
    if file_bytes > MAX_CLI_AUDIO_FILE_BYTES:
        raise ValueError(
            f"Audio file has {file_bytes} bytes; limit is "
            f"{MAX_CLI_AUDIO_FILE_BYTES}"
        )
    if (
        isinstance(max_duration_seconds, bool)
        or not isinstance(max_duration_seconds, Real)
        or not math.isfinite(float(max_duration_seconds))
        or float(max_duration_seconds) <= 0.0
    ):
        raise ValueError("max_duration_seconds must be finite and positive")

    try:
        metadata = sf.info(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Unable to inspect audio input: {type(exc).__name__}"
        ) from exc

    frames = metadata.frames
    sample_rate = metadata.samplerate
    channels = metadata.channels
    if isinstance(frames, bool) or not isinstance(frames, Integral) or frames <= 0:
        raise ValueError("Audio input has no audio frames")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, Integral)
        or not MIN_SAMPLE_RATE <= int(sample_rate) <= MAX_SAMPLE_RATE
    ):
        raise ValueError(
            "Audio sample rate must be an integer in "
            f"[{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}]"
        )
    if (
        isinstance(channels, bool)
        or not isinstance(channels, Integral)
        or not 1 <= int(channels) <= MAX_AUDIO_CHANNELS
    ):
        raise ValueError(
            "Audio channels must be an integer in "
            f"[1, {MAX_AUDIO_CHANNELS}]"
        )

    duration_seconds = int(frames) / int(sample_rate)
    if duration_seconds > float(max_duration_seconds):
        raise ValueError(
            f"Audio duration is {duration_seconds:.3f}s; limit is "
            f"{float(max_duration_seconds):.3f}s"
        )
    decoded_bytes = int(frames) * int(channels) * np.dtype(np.float32).itemsize
    if decoded_bytes > MAX_CLI_DECODED_AUDIO_BYTES:
        raise ValueError(
            f"Decoded audio requires {decoded_bytes} bytes; limit is "
            f"{MAX_CLI_DECODED_AUDIO_BYTES}"
        )

    try:
        audio, decoded_sample_rate = sf.read(
            path,
            dtype="float32",
            always_2d=False,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Unable to decode audio input: {type(exc).__name__}"
        ) from exc
    audio = np.asarray(audio, dtype=np.float32)
    expected_shape = (
        (int(frames),)
        if int(channels) == 1
        else (int(frames), int(channels))
    )
    if audio.shape != expected_shape:
        raise ValueError(
            f"Decoded audio shape {audio.shape} does not match metadata "
            f"{expected_shape}"
        )
    if decoded_sample_rate != int(sample_rate):
        raise ValueError(
            "Decoded audio sample rate does not match inspected metadata"
        )
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def run_interactive(pipeline):
    """Run interactive microphone-based translation."""
    print("\n🎤 Interactive Mode — Speak into the microphone")
    print("   Supported: Vietnamese ↔ English")
    print("   Press Ctrl+C to stop\n")

    try:
        pipeline.run_interactive(on_result=print_speech_result)
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


def run_file_translation(
    pipeline,
    input_path: str,
    source_lang: str = None,
    *,
    prepared_audio: tuple[np.ndarray, int] | None = None,
):
    """Translate an audio file."""
    print(f"\n📂 Translating file: {input_path}")

    if prepared_audio is None:
        audio, sr = load_audio_file(
            input_path,
            max_duration_seconds=(
                pipeline.asr_engine.max_input_duration_seconds
            ),
        )
    else:
        audio, sr = prepared_audio

    print(f"   Duration: {len(audio)/sr:.2f}s | Sample Rate: {sr} Hz\n")

    result = pipeline.translate_speech(
        audio,
        source_lang=source_lang,
        sample_rate=sr,
    )

    print_speech_result(result)

    if result.requires_confirmation and result.safety_passed:
        confirmation = input(
            "⚠️  Type PLAY to confirm this clinical action and synthesize audio: "
        ).strip()
        if confirmation == "PLAY":
            pipeline.confirm_and_play(result, confirmed=True)
        else:
            print("🔇 Playback cancelled; no audio was synthesized.")
    elif result.output_audio is not None:
        play = input("🔊 Play translated audio? (y/n): ").strip().lower()
        if play == 'y':
            pipeline.play(result)


def run_text_translation(pipeline, text: str, source_lang: str):
    """Text-only translation (no ASR/TTS)."""
    print(f"\n📝 Text Translation Mode")
    print(f"   Input ({source_lang.upper()}): {text}\n")

    result = pipeline.translate_text(text, source_lang)

    target_lang = "EN" if source_lang == "vi" else "VI"
    print(f"  🌐 Output ({target_lang}): {format_translation_for_display(result)}")
    print(f"  ⏱️  Latency: {result.latency_ms:.0f}ms")
    print(f"  💾 From Cache: {'Yes ⚡' if result.from_cache else 'No'}")
    print_confirmation_warning(result)


def run_benchmark(pipeline, num_samples: int = 10):
    """Run quick benchmark with synthetic test data."""
    if (
        isinstance(num_samples, bool)
        or not isinstance(num_samples, int)
        or not 1 <= num_samples <= MAX_BENCHMARK_SAMPLES
    ):
        raise ValueError(
            f"num_samples must be between 1 and {MAX_BENCHMARK_SAMPLES}"
        )
    print(f"\n📊 Benchmark Mode — {num_samples} samples\n")

    # Test phrases for both directions
    test_phrases_vi = [
        "Bệnh nhân sốc phản vệ cần tiêm epinephrine ngay",
        "Kiểm tra huyết áp và nhịp tim",
        "Bệnh nhân đau ngực bên trái lan xuống cánh tay",
        "Chuẩn bị phòng mổ cho ca phẫu thuật cấp cứu",
        "Truyền dịch natri clorua 0.9 phần trăm",
    ]

    test_phrases_en = [
        "Patient is in anaphylactic shock requires immediate epinephrine",
        "Check blood pressure and heart rate",
        "Patient reports chest pain radiating to left arm",
        "Prepare the operating room for emergency surgery",
        "Start normal saline IV drip at 150 milliliters per hour",
    ]

    latencies = []
    vi_sample_count = (num_samples + 1) // 2
    en_sample_count = num_samples - vi_sample_count

    # Test VI → EN
    print("  Testing Vietnamese → English:")
    for phrase in test_phrases_vi[:vi_sample_count]:
        result = pipeline.translate_text(phrase, "vi")
        latencies.append(result.latency_ms)
        cache_tag = "⚡" if result.from_cache else "🧠"
        display = format_translation_for_display(result)
        print(f"    {cache_tag} [{result.latency_ms:6.0f}ms] {phrase[:40]}... → {display[:80]}...")

    # Test EN → VI
    print("\n  Testing English → Vietnamese:")
    for phrase in test_phrases_en[:en_sample_count]:
        result = pipeline.translate_text(phrase, "en")
        latencies.append(result.latency_ms)
        cache_tag = "⚡" if result.from_cache else "🧠"
        display = format_translation_for_display(result)
        print(f"    {cache_tag} [{result.latency_ms:6.0f}ms] {phrase[:40]}... → {display[:80]}...")

    # Summary
    if latencies:
        print(f"\n{'─'*60}")
        print(f"  📊 Benchmark Results:")
        print(f"     Samples: {len(latencies)}")
        print(f"     Avg Latency: {np.mean(latencies):.0f}ms")
        print(f"     Min Latency: {np.min(latencies):.0f}ms")
        print(f"     Max Latency: {np.max(latencies):.0f}ms")
        print(f"     P95 Latency: {np.percentile(latencies, 95):.0f}ms")
        print(f"{'─'*60}")

    stats = pipeline.get_performance_stats()
    print(f"\n  Pipeline Stats: {stats}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="MediVoice Edge — CLI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="OneVoice AI Challenge 2026"
    )
    parser.add_argument(
        "--mode", type=str, default="text",
        choices=["interactive", "file", "text", "benchmark"],
        help="Demo mode (default: text)"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input audio file path (for file mode)"
    )
    parser.add_argument(
        "--text", type=str, default="Bệnh nhân sốc phản vệ",
        help="Input text (for text mode)"
    )
    parser.add_argument(
        "--lang", type=str, default="vi",
        choices=["vi", "en"],
        help="Source language (default: vi)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/pipeline_config.yaml",
        help="Pipeline config file path"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="Number of benchmark samples"
    )

    args = parser.parse_args(argv)

    if args.mode == "file" and not args.input:
        parser.error("--input is required for file mode")
    if args.mode == "text" and not args.text.strip():
        parser.error("--text must not be blank for text mode")
    if args.mode == "benchmark" and not (
        1 <= args.num_samples <= MAX_BENCHMARK_SAMPLES
    ):
        parser.error(
            "--num-samples must be between 1 and "
            f"{MAX_BENCHMARK_SAMPLES}"
        )
    if args.mode == "file":
        input_path = Path(args.input)
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Audio input does not exist: {input_path}"
            )
        input_bytes = input_path.stat().st_size
        if input_bytes > MAX_CLI_AUDIO_FILE_BYTES:
            raise ValueError(
                f"Audio file has {input_bytes} bytes; limit is "
                f"{MAX_CLI_AUDIO_FILE_BYTES}"
            )

    print(BANNER)

    # Initialize pipeline
    from src.pipeline.orchestrator import MediVoicePipeline

    print("⏳ Loading pipeline components...\n")
    pipeline = MediVoicePipeline(config_path=args.config)
    prepared_audio = None
    if args.mode == "file":
        prepared_audio = load_audio_file(
            args.input,
            max_duration_seconds=(
                pipeline.asr_engine.max_input_duration_seconds
            ),
        )
    pipeline.load()
    print()

    # Run selected mode
    if args.mode == "interactive":
        run_interactive(pipeline)
    elif args.mode == "file":
        run_file_translation(
            pipeline,
            args.input,
            args.lang,
            prepared_audio=prepared_audio,
        )
    elif args.mode == "text":
        run_text_translation(pipeline, args.text, args.lang)
    elif args.mode == "benchmark":
        run_benchmark(pipeline, args.num_samples)
    return 0


def cli_entrypoint(argv: list[str] | None = None) -> int:
    """Run the CLI with stable exit codes and no unexpected traceback leak."""
    try:
        return main(argv)
    except KeyboardInterrupt:
        print("❌ Interrupted.", file=sys.stderr)
        return 130
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"❌ Unexpected failure ({type(exc).__name__}); details suppressed.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(cli_entrypoint())

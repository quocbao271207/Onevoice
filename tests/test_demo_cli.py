import sys
from types import SimpleNamespace

import numpy as np
import pytest

import demo.demo_cli as demo_cli
from demo.demo_cli import (
    format_translation_for_display,
    load_audio_file,
    print_speech_result,
    run_interactive,
)


def test_cli_never_displays_candidate_that_failed_safety_gate():
    result = SimpleNamespace(
        safety_passed=False,
        safety_issues=[
            "quantity_binding_mismatch:aspirin:[('5', 'mg')]->[('50', 'mg')]",
            "negation_mismatch",
        ],
        translated_text="PATIENT_SECRET_UNSAFE_OUTPUT",
    )

    displayed = format_translation_for_display(result)

    assert displayed == (
        "[BLOCKED BY CLINICAL SAFETY GATE: "
        "negation_mismatch, quantity_binding_mismatch]"
    )
    assert "PATIENT_SECRET_UNSAFE_OUTPUT" not in displayed


def test_cli_displays_only_safety_passed_translation():
    result = SimpleNamespace(
        safety_passed=True,
        safety_issues=[],
        translated_text="Bản dịch an toàn",
    )

    assert format_translation_for_display(result) == "Bản dịch an toàn"


def test_interactive_cli_displays_blocked_result_before_clean_shutdown(capsys):
    unsafe_result = SimpleNamespace(
        asr_language="vi",
        asr_text="Cho bệnh nhân thuốc",
        target_language="en",
        translated_text="PATIENT_SECRET_UNSAFE_OUTPUT",
        safety_passed=False,
        safety_issues=["negation_mismatch"],
        total_latency_ms=25.0,
        overall_rtf=0.25,
        asr_confidence=0.5,
        from_cache=False,
        requires_confirmation=False,
    )

    class FakePipeline:
        def run_interactive(self, *, on_result):
            assert on_result is print_speech_result
            on_result(unsafe_result)
            raise KeyboardInterrupt

    run_interactive(FakePipeline())

    output = capsys.readouterr().out
    assert "Cho bệnh nhân thuốc" in output
    assert "BLOCKED BY CLINICAL SAFETY GATE" in output
    assert "PATIENT_SECRET_UNSAFE_OUTPUT" not in output
    assert "Goodbye" in output


def test_cli_audio_preflight_rejects_overlength_before_decode(
    tmp_path,
    monkeypatch,
):
    audio_path = tmp_path / "long.wav"
    audio_path.write_bytes(b"header")

    def unexpected_read(*_args, **_kwargs):
        raise AssertionError("Overlength audio must be rejected before decode")

    fake_soundfile = SimpleNamespace(
        info=lambda _path: SimpleNamespace(
            frames=16_001,
            samplerate=16_000,
            channels=1,
        ),
        read=unexpected_read,
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    with pytest.raises(ValueError, match=r"1\.000s.*limit is 1\.000s"):
        load_audio_file(audio_path, max_duration_seconds=1.0)


def test_cli_audio_loader_preserves_stereo_for_validated_downmix(
    tmp_path,
    monkeypatch,
):
    audio_path = tmp_path / "stereo.wav"
    audio_path.write_bytes(b"header")
    stereo = np.column_stack(
        (
            np.zeros(160, dtype=np.float32),
            np.full(160, 0.5, dtype=np.float32),
        )
    )
    calls = []

    def fake_read(path, **kwargs):
        calls.append((path, kwargs))
        return stereo.copy(), 16_000

    fake_soundfile = SimpleNamespace(
        info=lambda _path: SimpleNamespace(
            frames=160,
            samplerate=16_000,
            channels=2,
        ),
        read=fake_read,
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    audio, sample_rate = load_audio_file(
        audio_path,
        max_duration_seconds=1.0,
    )

    assert sample_rate == 16_000
    assert audio.shape == (160, 2)
    assert np.array_equal(audio, stereo)
    assert calls == [
        (
            audio_path,
            {"dtype": "float32", "always_2d": False},
        )
    ]


def test_cli_audio_preflight_bounds_file_and_decoded_allocation(
    tmp_path,
    monkeypatch,
):
    audio_path = tmp_path / "bounded.wav"
    audio_path.write_bytes(b"header")
    fake_soundfile = SimpleNamespace(
        info=lambda _path: SimpleNamespace(
            frames=100,
            samplerate=16_000,
            channels=2,
        ),
        read=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Oversized input must stop before decode")
        ),
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    monkeypatch.setattr("demo.demo_cli.MAX_CLI_AUDIO_FILE_BYTES", 4)
    with pytest.raises(ValueError, match="Audio file has 6 bytes; limit is 4"):
        load_audio_file(audio_path, max_duration_seconds=1.0)

    monkeypatch.setattr("demo.demo_cli.MAX_CLI_AUDIO_FILE_BYTES", 64)
    monkeypatch.setattr("demo.demo_cli.MAX_CLI_DECODED_AUDIO_BYTES", 100)
    with pytest.raises(ValueError, match="Decoded audio requires 800 bytes"):
        load_audio_file(audio_path, max_duration_seconds=1.0)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"frames": 0, "samplerate": 16_000, "channels": 1}, "no audio frames"),
        (
            {"frames": 160, "samplerate": 1_000, "channels": 1},
            "sample rate must be an integer",
        ),
        (
            {"frames": 160, "samplerate": 16_000, "channels": 9},
            "channels must be an integer",
        ),
    ],
)
def test_cli_audio_preflight_rejects_invalid_metadata(
    tmp_path,
    monkeypatch,
    metadata,
    message,
):
    audio_path = tmp_path / "invalid.wav"
    audio_path.write_bytes(b"header")
    fake_soundfile = SimpleNamespace(
        info=lambda _path: SimpleNamespace(**metadata),
        read=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Invalid metadata must stop before decode")
        ),
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    with pytest.raises(ValueError, match=message):
        load_audio_file(audio_path, max_duration_seconds=1.0)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--mode", "file"], "--input is required"),
        (["--mode", "text", "--text", "   "], "--text must not be blank"),
        (["--mode", "benchmark", "--num-samples", "0"], "between 1 and 10"),
        (["--mode", "benchmark", "--num-samples", "11"], "between 1 and 10"),
    ],
)
def test_cli_rejects_invalid_arguments_before_pipeline_construction(
    argv,
    message,
    monkeypatch,
    capsys,
):
    def unexpected_pipeline(*_args, **_kwargs):
        raise AssertionError("Invalid CLI input must stop before model setup")

    monkeypatch.setattr(
        "src.pipeline.orchestrator.MediVoicePipeline",
        unexpected_pipeline,
    )

    with pytest.raises(SystemExit) as exc_info:
        demo_cli.main(argv)

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_missing_audio_path_stops_before_pipeline_construction(
    tmp_path,
    monkeypatch,
):
    missing = tmp_path / "missing.wav"

    def unexpected_pipeline(*_args, **_kwargs):
        raise AssertionError("Missing audio must stop before model setup")

    monkeypatch.setattr(
        "src.pipeline.orchestrator.MediVoicePipeline",
        unexpected_pipeline,
    )

    with pytest.raises(FileNotFoundError, match="Audio input does not exist"):
        demo_cli.main(["--mode", "file", "--input", str(missing)])


def test_cli_decodes_file_before_loading_models(tmp_path, monkeypatch):
    audio_path = tmp_path / "malformed.wav"
    audio_path.write_bytes(b"header")
    events = []

    class FakePipeline:
        def __init__(self, *, config_path):
            events.append(("construct", config_path))
            self.asr_engine = SimpleNamespace(
                max_input_duration_seconds=30.0
            )

        def load(self):
            events.append("load")
            raise AssertionError("Models must not load before audio preflight")

    def reject_audio(*_args, **_kwargs):
        events.append("audio_preflight")
        raise ValueError("malformed audio")

    monkeypatch.setattr(
        "src.pipeline.orchestrator.MediVoicePipeline",
        FakePipeline,
    )
    monkeypatch.setattr(demo_cli, "load_audio_file", reject_audio)

    with pytest.raises(ValueError, match="malformed audio"):
        demo_cli.main(
            ["--mode", "file", "--input", str(audio_path)]
        )

    assert events == [
        ("construct", "configs/pipeline_config.yaml"),
        "audio_preflight",
    ]


def test_cli_benchmark_runs_exact_requested_sample_count(capsys):
    calls = []

    class FakePipeline:
        def translate_text(self, text, language):
            calls.append((text, language))
            return SimpleNamespace(
                latency_ms=1.0,
                from_cache=False,
                safety_passed=True,
                safety_issues=[],
                translated_text="safe",
                requires_confirmation=False,
            )

        def get_performance_stats(self):
            return {}

    demo_cli.run_benchmark(FakePipeline(), num_samples=3)

    assert len(calls) == 3
    assert [language for _text, language in calls] == ["vi", "vi", "en"]
    assert "Samples: 3" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("failure", "expected_code", "visible", "hidden"),
    [
        (RuntimeError("sounddevice is unavailable"), 1, "sounddevice", "Traceback"),
        (KeyboardInterrupt(), 130, "Interrupted", "Traceback"),
    ],
)
def test_cli_entrypoint_returns_stable_exit_codes_without_tracebacks(
    failure,
    expected_code,
    visible,
    hidden,
    monkeypatch,
    capsys,
):
    def fail_main(_argv=None):
        raise failure

    monkeypatch.setattr(demo_cli, "main", fail_main)

    assert demo_cli.cli_entrypoint([]) == expected_code
    stderr = capsys.readouterr().err
    assert visible in stderr
    assert hidden not in stderr


def test_cli_entrypoint_redacts_unexpected_exception_details(monkeypatch, capsys):
    class UnexpectedFailure(Exception):
        pass

    def fail_main(_argv=None):
        raise UnexpectedFailure("PATIENT_SECRET_INTERNAL_DETAIL")

    monkeypatch.setattr(demo_cli, "main", fail_main)

    assert demo_cli.cli_entrypoint([]) == 1
    stderr = capsys.readouterr().err
    assert "UnexpectedFailure" in stderr
    assert "PATIENT_SECRET_INTERNAL_DETAIL" not in stderr
    assert "Traceback" not in stderr

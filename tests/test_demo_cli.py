import sys
from types import SimpleNamespace

import numpy as np
import pytest

from demo.demo_cli import format_translation_for_display, load_audio_file


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

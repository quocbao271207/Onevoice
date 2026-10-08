from __future__ import annotations

import io
from pathlib import Path

import pytest

from scripts.release_secret_scan import (
    ReleaseSecretScanner,
    is_scannable_release_path,
    scan_release_stream,
)


def test_release_secret_scanner_detects_cross_chunk_token_without_echoing_it():
    secret = b"gh" + b"p_" + b"A" * 40
    scanner = ReleaseSecretScanner("data/reports/runtime.log")
    scanner.feed(b"prefix " + secret[:17])

    with pytest.raises(ValueError, match=r"github_token.*runtime\.log") as caught:
        scanner.feed(secret[17:] + b" suffix")

    assert secret.decode("ascii") not in str(caught.value)


@pytest.mark.parametrize(
    "payload",
    [
        b"-----BEGIN " + b"PRIVATE KEY-----",
        b"Authorization: Bearer " + b"eyJ" + b"A" * 12 + b"." + b"B" * 12 + b"." + b"C" * 12,
        b"Authorization: Bearer sk-" + b"G" * 32,
        b"https://example.invalid/file?X-Amz-Signature=" + b"D" * 64,
        b'api_token="' + b"E" * 32 + b'"',
    ],
)
def test_release_secret_scanner_covers_private_keys_bearers_urls_and_assignments(payload):
    with pytest.raises(ValueError, match="Potential release secret"):
        scan_release_stream(io.BytesIO(payload), "data/reports/evidence.json")


def test_release_secret_scanner_ignores_placeholders_and_binary_artifacts():
    assert is_scannable_release_path("data/reports/runtime.log")
    assert is_scannable_release_path(Path("data/reports/runtime.log"))
    assert not is_scannable_release_path("models/adapter.safetensors")
    assert scan_release_stream(
        io.BytesIO(b'api_token="REDACTED"'),
        "data/reports/evidence.json",
    ) > 0
    assert scan_release_stream(
        io.BytesIO(b"gh" + b"p_" + b"A" * 40),
        "models/adapter.bin",
    ) > 0


def test_signed_dataset_provenance_is_not_treated_as_a_shareable_report():
    signed_url = b"https://dataset.invalid/audio?X-Amz-Signature=" + b"F" * 64
    key_like_path = b"https://dataset.invalid/audio/sk-" + b"G" * 32 + b"/clip.wav"
    assert scan_release_stream(
        io.BytesIO(signed_url),
        "data/eval/locked_manifest.jsonl",
    ) == len(signed_url)
    assert scan_release_stream(
        io.BytesIO(key_like_path),
        "data/processed/manifest.jsonl",
    ) == len(key_like_path)
    with pytest.raises(ValueError, match="signed_url"):
        scan_release_stream(
            io.BytesIO(signed_url),
            "data/reports/shared_runtime.json",
        )

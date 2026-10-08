"""Bounded streaming credential scan for files entering release evidence."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import BinaryIO


SCANNER_VERSION = 1
SCAN_CHUNK_BYTES = 1024 * 1024
SCAN_OVERLAP_BYTES = 1024
SCANNABLE_SUFFIXES = frozenset(
    {
        ".cfg",
        ".conf",
        ".csv",
        ".env",
        ".err",
        ".html",
        ".ini",
        ".js",
        ".json",
        ".jsonl",
        ".log",
        ".md",
        ".out",
        ".properties",
        ".provenance",
        ".ps1",
        ".py",
        ".sha256",
        ".sh",
        ".toml",
        ".tsv",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)
SCANNABLE_BASENAMES = frozenset({"sha256sums", "sha256sums.txt"})


SECRET_RULES: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "private_key",
        re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "github_token",
        re.compile(
            rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{40,255})\b"
        ),
    ),
    ("huggingface_token", re.compile(rb"\bhf_[A-Za-z0-9]{30,255}\b")),
    (
        "openai_api_key",
        re.compile(
            rb"(?:Bearer\s+|OPENAI[_-]API[_-]KEY\s*[\"']?\s*[:=]\s*[\"']?)"
            rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,255}\b",
            re.IGNORECASE,
        ),
    ),
    ("aws_access_key", re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "bearer_jwt",
        re.compile(
            rb"\bBearer\s+eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
            re.IGNORECASE,
        ),
    ),
    (
        "signed_url",
        re.compile(
            rb"[?&](?:X-Amz-Signature|X-Goog-Signature|Signature|sig|token)="
            rb"[A-Za-z0-9_%+./=-]{20,}",
            re.IGNORECASE,
        ),
    ),
    (
        "credential_assignment",
        re.compile(
            rb"(?:api[_-]?(?:key|token)|access[_-]?token|client[_-]?secret|password)"
            rb"\s*[\"']?\s*[:=]\s*[\"']?"
            rb"(?!REDACTED\b|PLACEHOLDER\b|EXAMPLE\b)"
            rb"[A-Za-z0-9_+./=-]{20,255}",
            re.IGNORECASE,
        ),
    ),
)
REPORT_ONLY_RULES = frozenset({"signed_url"})


def _rule_applies(rule_name: str, logical_path: str) -> bool:
    if rule_name not in REPORT_ONLY_RULES:
        return True
    normalized = logical_path.replace("\\", "/").lower()
    return normalized.startswith("data/reports/") or PurePosixPath(normalized).suffix in {
        ".err",
        ".log",
        ".out",
    }


def is_scannable_release_path(logical_path: str | Path) -> bool:
    """Select textual release artifacts without streaming model/audio binaries."""
    normalized = str(logical_path).replace("\\", "/")
    path = PurePosixPath(normalized)
    return (
        path.suffix.lower() in SCANNABLE_SUFFIXES
        or path.name.lower() in SCANNABLE_BASENAMES
    )


class ReleaseSecretScanner:
    """Incrementally detect credentials while retaining only a small overlap."""

    def __init__(self, logical_path: str) -> None:
        if not isinstance(logical_path, str) or not logical_path:
            raise ValueError("Secret scanner logical path must be a non-empty string")
        self.logical_path = logical_path
        self.enabled = is_scannable_release_path(logical_path)
        self._carry = b""
        self.bytes_scanned = 0

    def feed(self, chunk: bytes) -> None:
        if not isinstance(chunk, bytes):
            raise TypeError("Secret scanner chunks must be bytes")
        self.bytes_scanned += len(chunk)
        if not self.enabled or not chunk:
            return
        window = self._carry + chunk
        for rule_name, pattern in SECRET_RULES:
            if _rule_applies(rule_name, self.logical_path) and pattern.search(window):
                raise ValueError(
                    f"Potential release secret ({rule_name}) in {self.logical_path}"
                )
        self._carry = window[-SCAN_OVERLAP_BYTES:]


def scan_release_stream(handle: BinaryIO, logical_path: str) -> int:
    scanner = ReleaseSecretScanner(logical_path)
    while chunk := handle.read(SCAN_CHUNK_BYTES):
        scanner.feed(chunk)
    return scanner.bytes_scanned


def scan_release_file(path: Path, logical_path: str) -> int:
    if not is_scannable_release_path(logical_path):
        return 0
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Release secret scan requires a regular file: {logical_path}")
    size = path.stat().st_size
    with path.open("rb") as handle:
        scanned = scan_release_stream(handle, logical_path)
    if scanned != size or path.stat().st_size != size:
        raise RuntimeError(f"Release scan source changed while reading: {logical_path}")
    return scanned

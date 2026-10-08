"""Remove transient ASR download URLs from generated JSONL reports."""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (
    ROOT / "data" / "reports" / "eda",
    ROOT / "data" / "reports" / "eda_sample",
)
MAX_JSONL_LINE_BYTES = 4 * 1024 * 1024


def _is_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    return bool(checker and checker())


def _reject_link(path: Path) -> None:
    if path.is_symlink() or _is_junction(path):
        raise ValueError(f"Refusing linked report path: {path}")


def report_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        root = Path(os.path.abspath(root))
        _reject_link(root)
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.rglob("*.jsonl"):
            _reject_link(path)
            for parent in path.parents:
                _reject_link(parent)
                if parent == root:
                    break
            if not path.is_file():
                raise ValueError(f"Report path is not a regular file: {path}")
            files.append(path)
    return sorted(set(files))


def sanitize_report(path: Path, *, apply: bool) -> int:
    """Validate one JSONL report and optionally remove top-level audio_url fields."""
    path = Path(os.path.abspath(path))
    _reject_link(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    original_mode = stat.S_IMODE(path.stat().st_mode)
    temporary = path.with_name(f".{path.name}.sanitize.part")
    if apply and temporary.exists():
        raise FileExistsError(temporary)
    changed = 0
    output = None
    try:
        if apply:
            output = temporary.open("w", encoding="utf-8", newline="\n")
        with path.open("rb") as source:
            line_number = 0
            while True:
                raw = source.readline(MAX_JSONL_LINE_BYTES + 1)
                if not raw:
                    break
                line_number += 1
                if len(raw) > MAX_JSONL_LINE_BYTES:
                    raise ValueError(f"JSONL line exceeds size limit: {path}:{line_number}")
                if not raw.strip():
                    raise ValueError(f"Blank JSONL line is not allowed: {path}:{line_number}")
                try:
                    record = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(f"Invalid UTF-8 JSONL record: {path}:{line_number}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"JSONL record must be an object: {path}:{line_number}")
                if "audio_url" in record:
                    changed += 1
                    record.pop("audio_url")
                if output is not None:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
        if output is not None:
            output.close()
            output = None
            if changed:
                temporary.chmod(original_mode)
                os.replace(temporary, path)
            else:
                temporary.unlink()
    except BaseException:
        if output is not None:
            output.close()
        if temporary.exists() and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        raise
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        help="Report root to scan; repeatable. Defaults to eda and eda_sample.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically remove audio_url fields. Omit for read-only dry-run.",
    )
    args = parser.parse_args()
    roots = args.root or [root for root in DEFAULT_ROOTS if root.is_dir()]
    files = report_files(roots)
    affected_files = 0
    removed_fields = 0
    for path in files:
        count = sanitize_report(path, apply=args.apply)
        if count:
            affected_files += 1
            removed_fields += count
            try:
                label = path.relative_to(ROOT)
            except ValueError:
                label = path
            print(f"{label}: audio_url_fields={count}")
    mode = "applied" if args.apply else "dry-run"
    print(
        f"{mode}: scanned_files={len(files)} affected_files={affected_files} "
        f"audio_url_fields={removed_fields}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

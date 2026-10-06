"""Capture fail-closed QCS6490 identity evidence from Linux sysfs/device-tree.

Run this script on the physical deployment board.  The resulting JSON is an
input to ``prepare_deployment_benchmark.py --action finalize``; it is not a
benchmark result by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_model_bakeoff import qcs6490_identity_failures  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT / "data/reports/model_bakeoff/board-evidence/qcs6490-identity.json"
)


def _read_text(path: Path) -> str:
    return (
        path.read_bytes()
        .replace(b"\x00", b"")
        .decode("utf-8", errors="strict")
        .strip()
    )


def _read_compatible(path: Path) -> list[str]:
    return [
        value.decode("utf-8", errors="strict").strip()
        for value in path.read_bytes().split(b"\x00")
        if value.strip()
    ]


def capture_identity(
    system_root: Path = Path("/"),
    *,
    architecture: str | None = None,
) -> dict[str, Any]:
    device_tree = system_root / "proc/device-tree"
    soc_root = system_root / "sys/devices/soc0"
    payload: dict[str, Any] = {
        "version": 1,
        "capture_source": "linux_sysfs_device_tree",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "architecture": architecture or platform.machine(),
        "board_model": _read_text(device_tree / "model"),
        "device_tree_compatible": _read_compatible(device_tree / "compatible"),
        "soc_family": _read_text(soc_root / "family"),
        "soc_id": _read_text(soc_root / "soc_id"),
        "kernel_release": platform.release(),
    }
    failures = qcs6490_identity_failures(payload)
    if failures:
        raise ValueError("Physical board is not verified QCS6490: " + ", ".join(failures))
    return payload


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite board identity evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = capture_identity()
    write_exclusive(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

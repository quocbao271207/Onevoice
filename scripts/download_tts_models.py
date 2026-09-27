"""Download the exact Piper voice artifacts used by OneVoice."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "rhasspy/piper-voices"
FILES = {
    "vi_VN-vivos-x_low.onnx": "vi/vi_VN/vivos/x_low/vi_VN-vivos-x_low.onnx",
    "vi_VN-vivos-x_low.onnx.json": "vi/vi_VN/vivos/x_low/vi_VN-vivos-x_low.onnx.json",
    "en_US-lessac-medium.onnx": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en_US-lessac-medium.onnx.json": "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "tts")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for local_name, remote_name in FILES.items():
        downloaded = Path(hf_hub_download(REPO_ID, remote_name))
        target = args.output_dir / local_name
        if not target.exists() or target.stat().st_size != downloaded.stat().st_size:
            shutil.copy2(downloaded, target)
        print(f"{target}: {target.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

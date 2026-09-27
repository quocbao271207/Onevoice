from pathlib import Path

import pytest

from src.training import quantize_qnn


def test_snpe_dlc_compile_fails_closed_without_sdk(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"placeholder")
    monkeypatch.setattr(quantize_qnn.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="cannot be skipped"):
        quantize_qnn.compile_to_dlc(str(source), str(tmp_path / "model.dlc"))


def test_snpe_dlc_compile_requires_output(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"placeholder")
    monkeypatch.setattr(quantize_qnn.shutil, "which", lambda _name: "snpe-onnx-to-dlc")
    monkeypatch.setattr(quantize_qnn.subprocess, "run", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="produced no DLC"):
        quantize_qnn.compile_to_dlc(str(source), str(tmp_path / "model.dlc"))


def test_int4_cli_is_not_advertised() -> None:
    source = Path(quantize_qnn.__file__).read_text(encoding="utf-8")
    assert 'choices=["int8"]' in source
    assert "QNN converter does not produce an SNPE DLC" in source

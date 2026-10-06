from pathlib import Path

import pytest

from scripts.capture_qcs6490_identity import capture_identity, write_exclusive


def fake_system_tree(tmp_path: Path, *, qcs6490: bool) -> Path:
    root = tmp_path / "root"
    device_tree = root / "proc/device-tree"
    soc_root = root / "sys/devices/soc0"
    device_tree.mkdir(parents=True)
    soc_root.mkdir(parents=True)
    if qcs6490:
        (device_tree / "model").write_bytes(b"Qualcomm QCS6490 RB3 Gen 2\x00")
        (device_tree / "compatible").write_bytes(
            b"qcom,qcs6490-rb3gen2\x00qcom,qcs6490\x00"
        )
        (soc_root / "family").write_text("Qualcomm QCS6490", encoding="utf-8")
        (soc_root / "soc_id").write_text("QCS6490", encoding="utf-8")
    else:
        (device_tree / "model").write_bytes(b"Arduino Uno\x00")
        (device_tree / "compatible").write_bytes(b"arduino,uno\x00")
        (soc_root / "family").write_text("AVR", encoding="utf-8")
        (soc_root / "soc_id").write_text("ATmega328P", encoding="utf-8")
    return root


def test_capture_identity_accepts_qcs6490_device_tree(tmp_path: Path):
    payload = capture_identity(fake_system_tree(tmp_path, qcs6490=True), architecture="aarch64")

    assert payload["board_model"] == "Qualcomm QCS6490 RB3 Gen 2"
    assert payload["device_tree_compatible"] == [
        "qcom,qcs6490-rb3gen2",
        "qcom,qcs6490",
    ]


def test_capture_identity_rejects_arduino_even_if_arm64_is_claimed(tmp_path: Path):
    with pytest.raises(ValueError, match="not verified QCS6490"):
        capture_identity(fake_system_tree(tmp_path, qcs6490=False), architecture="arm64")


def test_identity_writer_is_immutable(tmp_path: Path):
    output = tmp_path / "identity.json"
    write_exclusive(output, {"version": 1})

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_exclusive(output, {"version": 2})

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from scripts.capture_qcs6490_runtime import (
    RunSample,
    capture_runtime,
    normalize_command,
    read_scaled_sensor,
    write_exclusive,
)
from scripts.seal_qcs6490_measurement import seal_measurement


CAPTURED_AT = datetime.fromisoformat("2026-10-09T10:00:00+07:00")


def fake_system_tree(tmp_path: Path, *, board_model: str = "QCS6490 RB3 Gen 2") -> Path:
    root = tmp_path / "root"
    device_tree = root / "proc/device-tree"
    soc_root = root / "sys/devices/soc0"
    device_tree.mkdir(parents=True)
    soc_root.mkdir(parents=True)
    (device_tree / "model").write_bytes(board_model.encode() + b"\x00")
    (device_tree / "compatible").write_bytes(
        b"qcom,qcs6490-rb3gen2\x00qcom,qcs6490\x00"
    )
    (soc_root / "family").write_text("Qualcomm QCS6490", encoding="utf-8")
    (soc_root / "soc_id").write_text("QCS6490", encoding="utf-8")
    return root


def sample(index: int) -> RunSample:
    return RunSample(
        latency_ms=100.0 + index,
        power_samples_mw=(2_500.0 + index, 2_510.0 + index),
        temperature_samples_c=(45.0 + index / 10.0,),
        peak_ram_bytes=100_000_000 + index,
    )


def test_capture_discards_warmups_and_builds_exact_sealer_schema(tmp_path: Path):
    invocations = iter(sample(index) for index in range(33))

    raw = capture_runtime(
        command=["--", "/opt/onevoice/infer", "--fixture", "clinical.json"],
        power_sensor_path=Path("/sys/power/system_mw"),
        power_scale_to_mw=1.0,
        temperature_sensor_path=Path("/sys/thermal/zone0/temp"),
        temperature_scale_to_c=0.001,
        system_root=fake_system_tree(tmp_path),
        architecture="aarch64",
        sample_runner=lambda: next(invocations),
        now=lambda: CAPTURED_AT,
    )

    assert raw["capture_source"] == "qcs6490_runtime_sampler"
    assert raw["captured_at"] == CAPTURED_AT.isoformat()
    assert raw["latency_samples_ms"] == [100.0 + index for index in range(3, 33)]
    assert len(raw["power_samples_mw"]) == 60
    assert len(raw["temperature_samples_c"]) == 30
    assert raw["peak_ram_bytes"] == 100_000_032
    assert raw["peak_vram_bytes"] == 0


def test_capture_output_is_accepted_by_physical_measurement_sealer(tmp_path: Path):
    system_root = fake_system_tree(tmp_path)
    raw_payload = capture_runtime(
        command=["/opt/onevoice/infer"],
        power_sensor_path=Path("/sys/power/system_mw"),
        power_scale_to_mw=1.0,
        temperature_sensor_path=Path("/sys/thermal/zone0/temp"),
        temperature_scale_to_c=0.001,
        warmup_runs=0,
        system_root=system_root,
        architecture="aarch64",
        sample_runner=lambda: sample(0),
        now=lambda: CAPTURED_AT,
    )
    raw_path = tmp_path / "raw.json"
    write_exclusive(raw_path, raw_payload)
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(
        json.dumps(
            {
                "version": 1,
                "capture_source": "linux_sysfs_device_tree",
                "captured_at": CAPTURED_AT.isoformat(),
                "architecture": "aarch64",
                "board_model": "QCS6490 RB3 Gen 2",
                "device_tree_compatible": [
                    "qcom,qcs6490-rb3gen2",
                    "qcom,qcs6490",
                ],
                "soc_family": "Qualcomm QCS6490",
                "soc_id": "QCS6490",
                "kernel_release": "6.1",
            }
        ),
        encoding="utf-8",
    )
    artifact = tmp_path / "winner.bin"
    artifact.write_bytes(b"qnn-context")

    sealed = seal_measurement(
        identity_path=identity_path,
        artifact_path=artifact,
        raw_measurement_path=raw_path,
        task="asr",
        direction=None,
        candidate_id="asr-winner",
        adapter_manifest_sha256="a" * 64,
        system_root=system_root,
        architecture="aarch64",
        now=CAPTURED_AT,
    )

    assert sealed["latency_samples_ms"] == [100.0] * 30
    assert len(sealed["power_samples_mw"]) == 60
    assert sealed["peak_ram_bytes"] == 100_000_000


def test_capture_continues_until_run_and_soak_duration_gates_pass(tmp_path: Path):
    elapsed_seconds = 0.0

    def runner() -> RunSample:
        nonlocal elapsed_seconds
        elapsed_seconds += 1.0
        return sample(0)

    raw = capture_runtime(
        command=["/opt/onevoice/infer"],
        power_sensor_path=Path("/sys/power/system_mw"),
        power_scale_to_mw=1.0,
        temperature_sensor_path=Path("/sys/thermal/zone0/temp"),
        temperature_scale_to_c=0.001,
        runs=30,
        warmup_runs=0,
        minimum_duration_seconds=35.0,
        system_root=fake_system_tree(tmp_path),
        architecture="aarch64",
        sample_runner=runner,
        now=lambda: CAPTURED_AT,
        monotonic=lambda: elapsed_seconds,
    )

    assert len(raw["latency_samples_ms"]) == 35
    assert elapsed_seconds == 35.0


def test_capture_verifies_board_before_invoking_benchmark(tmp_path: Path):
    invoked = False

    def runner() -> RunSample:
        nonlocal invoked
        invoked = True
        return sample(0)

    with pytest.raises(ValueError, match="not verified QCS6490"):
        capture_runtime(
            command=["/opt/onevoice/infer"],
            power_sensor_path=Path("/sys/power/system_mw"),
            power_scale_to_mw=1.0,
            temperature_sensor_path=Path("/sys/thermal/zone0/temp"),
            temperature_scale_to_c=0.001,
            system_root=fake_system_tree(tmp_path, board_model="Different board"),
            architecture="aarch64",
            sample_runner=runner,
            now=lambda: CAPTURED_AT,
        )
    assert invoked is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"runs": 29}, "runs must be between 30"),
        ({"runs": 10_001}, "runs must be between 30"),
        ({"warmup_runs": -1}, "warmup_runs must be between 0"),
        ({"sample_interval_ms": 49}, "sample_interval_ms must be between 50"),
        ({"timeout_seconds": float("nan")}, "outside the valid range"),
        ({"power_scale_to_mw": 0.0}, "outside the valid range"),
        ({"minimum_duration_seconds": 3_601}, "must be between 0"),
    ],
)
def test_capture_rejects_unbounded_configuration(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
):
    arguments = {
        "command": ["/opt/onevoice/infer"],
        "power_sensor_path": Path("/sys/power/system_mw"),
        "power_scale_to_mw": 1.0,
        "temperature_sensor_path": Path("/sys/thermal/zone0/temp"),
        "temperature_scale_to_c": 0.001,
        "system_root": fake_system_tree(tmp_path),
        "architecture": "aarch64",
        "sample_runner": lambda: sample(0),
        "now": lambda: CAPTURED_AT,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        capture_runtime(**arguments)


@pytest.mark.parametrize(
    "invalid_sample",
    [
        RunSample(float("nan"), (1.0,), (40.0,), 1),
        RunSample(1.0, (), (40.0,), 1),
        RunSample(1.0, (0.0,), (40.0,), 1),
        RunSample(1.0, (1_000_001.0,), (40.0,), 1),
        RunSample(1.0, (1.0,), (-273.15,), 1),
        RunSample(1.0, (1.0,), (250.1,), 1),
        RunSample(1.0, (1.0,), (40.0,), 0),
    ],
)
def test_capture_rejects_invalid_run_telemetry(
    tmp_path: Path,
    invalid_sample: RunSample,
):
    with pytest.raises((ValueError, TypeError)):
        capture_runtime(
            command=["/opt/onevoice/infer"],
            power_sensor_path=Path("/sys/power/system_mw"),
            power_scale_to_mw=1.0,
            temperature_sensor_path=Path("/sys/thermal/zone0/temp"),
            temperature_scale_to_c=0.001,
            warmup_runs=0,
            system_root=fake_system_tree(tmp_path),
            architecture="aarch64",
            sample_runner=lambda: invalid_sample,
            now=lambda: CAPTURED_AT,
        )


def test_sensor_reader_requires_finite_scaled_values(tmp_path: Path):
    sensor = tmp_path / "sensor"
    sensor.write_text("42500\n", encoding="utf-8")
    assert read_scaled_sensor(sensor, scale=0.001, label="temperature") == 42.5

    sensor.write_text("nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the valid range"):
        read_scaled_sensor(sensor, scale=1.0, label="power")


def test_command_is_an_argv_vector_and_output_is_immutable(tmp_path: Path):
    assert normalize_command(["--", "program", "argument with spaces"]) == (
        "program",
        "argument with spaces",
    )
    with pytest.raises(ValueError, match="required"):
        normalize_command(["--"])

    output = tmp_path / "raw.json"
    write_exclusive(output, {"version": 1})
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_exclusive(output, {"version": 2})
    assert json.loads(output.read_text(encoding="utf-8"))["version"] == 1

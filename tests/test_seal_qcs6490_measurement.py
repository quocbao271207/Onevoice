from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

import scripts.seal_qcs6490_measurement as measurement_sealer
from scripts.prepare_deployment_benchmark import _load_measurement_evidence
from scripts.run_model_bakeoff import sha256
from scripts.seal_qcs6490_measurement import (
    seal_measurement,
    write_exclusive,
)


CAPTURED_AT = "2026-10-09T10:00:00+07:00"
NOW = datetime.fromisoformat("2026-10-09T10:15:00+07:00")


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


def write_identity(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "capture_source": "linux_sysfs_device_tree",
                "captured_at": CAPTURED_AT,
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


def write_raw(path: Path, **overrides) -> None:
    payload = {
        "version": 1,
        "capture_source": "qcs6490_runtime_sampler",
        "captured_at": CAPTURED_AT,
        "latency_samples_ms": [100.0 + index for index in range(30)],
        "power_sensor": "sysfs:ina231/system_power",
        "power_samples_mw": [2500.0 + index for index in range(30)],
        "temperature_sensor": "sysfs:thermal_zone0/temp",
        "temperature_samples_c": [45.0 + index * 0.1 for index in range(30)],
        "peak_ram_bytes": 100_000_000,
        "peak_vram_bytes": 0,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    identity = tmp_path / "identity.json"
    artifact = tmp_path / "winner.bin"
    raw = tmp_path / "raw.json"
    write_identity(identity)
    artifact.write_bytes(b"compiled-qnn")
    write_raw(raw)
    return identity, artifact, raw, fake_system_tree(tmp_path)


def test_sealer_binds_live_board_raw_capture_and_artifact(tmp_path: Path):
    identity, artifact, raw, system_root = fixture(tmp_path)

    evidence = seal_measurement(
        identity_path=identity,
        artifact_path=artifact,
        raw_measurement_path=raw,
        task="mt",
        direction="en_to_vi",
        candidate_id="mt-winner",
        adapter_manifest_sha256="a" * 64,
        system_root=system_root,
        architecture="aarch64",
        now=NOW,
    )

    assert evidence["capture_source"] == "physical_qcs6490"
    assert evidence["identity_evidence_sha256"] == sha256(identity)
    assert evidence["artifact_sha256"] == sha256(artifact)
    assert evidence["raw_measurement_sha256"] == sha256(raw)
    assert len(evidence["latency_samples_ms"]) == 30
    assert evidence["peak_vram_bytes"] == 0

    project_root = tmp_path / "project"
    output = (
        project_root
        / "data/reports/model_bakeoff/board-evidence/measurements/mt-winner.json"
    )
    write_exclusive(output, evidence)
    loaded_path, loaded, loaded_sha256 = _load_measurement_evidence(
        {"measurement_evidence_path": str(output.relative_to(project_root))},
        {
            "task": "mt",
            "direction": "en_to_vi",
            "candidate_id": "mt-winner",
            "adapter_manifest_sha256": "a" * 64,
        },
        identity_sha256=sha256(identity),
        artifact_sha256=sha256(artifact),
        artifact_bytes=artifact.stat().st_size,
        project_root=project_root,
    )
    assert loaded_path == output.resolve()
    assert loaded == evidence
    assert loaded_sha256 == sha256(output)


def test_sealer_rejects_live_board_identity_mismatch(tmp_path: Path):
    identity, artifact, raw, _ = fixture(tmp_path)
    mismatched_root = fake_system_tree(tmp_path / "other", board_model="Different QCS6490")

    with pytest.raises(ValueError, match="Live board identity differs"):
        seal_measurement(
            identity_path=identity,
            artifact_path=artifact,
            raw_measurement_path=raw,
            task="asr",
            direction=None,
            candidate_id="asr-winner",
            adapter_manifest_sha256="b" * 64,
            system_root=mismatched_root,
            architecture="aarch64",
            now=NOW,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"latency_samples_ms": [1.0] * 29}, "at least 30 samples"),
        ({"power_samples_mw": [1.0] * 29 + [float("nan")]}, "strict JSON"),
        ({"temperature_sensor": ""}, "non-empty bounded string"),
        ({"peak_ram_bytes": True}, "must be an integer"),
        ({"captured_at": "2026-10-01T00:00:00+07:00"}, "not from the same session"),
    ],
)
def test_sealer_rejects_invalid_or_stale_raw_capture(
    tmp_path: Path,
    overrides: dict,
    message: str,
):
    identity, artifact, raw, system_root = fixture(tmp_path)
    write_raw(raw, **overrides)

    with pytest.raises(ValueError, match=message):
        seal_measurement(
            identity_path=identity,
            artifact_path=artifact,
            raw_measurement_path=raw,
            task="mt",
            direction="vi_to_en",
            candidate_id="mt-winner",
            adapter_manifest_sha256="c" * 64,
            system_root=system_root,
            architecture="aarch64",
            now=NOW,
        )


def test_measurement_writer_is_immutable(tmp_path: Path):
    output = tmp_path / "measurement.json"
    write_exclusive(output, {"version": 1})

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_exclusive(output, {"version": 2})
    assert json.loads(output.read_text(encoding="utf-8"))["version"] == 1


def test_sealer_rejects_duplicate_identity_fields(tmp_path: Path):
    identity, artifact, raw, system_root = fixture(tmp_path)
    identity.write_text(
        '{"version":1,"capture_source":"linux_sysfs_device_tree",'
        '"architecture":"aarch64","board_model":"Arduino Uno",'
        '"board_model":"QCS6490 RB3 Gen 2"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strict JSON"):
        seal_measurement(
            identity_path=identity,
            artifact_path=artifact,
            raw_measurement_path=raw,
            task="asr",
            direction=None,
            candidate_id="asr-winner",
            adapter_manifest_sha256="d" * 64,
            system_root=system_root,
            architecture="aarch64",
            now=NOW,
        )


def test_sealer_caps_compiled_artifact_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    identity, artifact, raw, system_root = fixture(tmp_path)
    monkeypatch.setattr(measurement_sealer, "MAX_DEPLOYMENT_ARTIFACT_BYTES", 4)

    with pytest.raises(ValueError, match="exceeds 4 bytes"):
        seal_measurement(
            identity_path=identity,
            artifact_path=artifact,
            raw_measurement_path=raw,
            task="mt",
            direction="en_to_vi",
            candidate_id="mt-winner",
            adapter_manifest_sha256="e" * 64,
            system_root=system_root,
            architecture="aarch64",
            now=NOW,
        )


def test_measurement_writer_rejects_non_finite_json(tmp_path: Path):
    output = tmp_path / "measurement.json"

    with pytest.raises(ValueError, match="not serializable"):
        write_exclusive(output, {"latency_ms": float("nan")})

    assert not output.exists()

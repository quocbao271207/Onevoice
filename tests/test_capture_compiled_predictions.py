from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

import scripts.capture_compiled_predictions as compiled_capture
from scripts.capture_compiled_predictions import (
    capture_compiled_predictions,
    compiled_prediction_provenance_failures,
    normalize_command_template,
)
from scripts.run_model_bakeoff import sha256


NOW = datetime.fromisoformat("2026-10-09T15:00:00+07:00")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def capture_fixture(tmp_path: Path, *, executor=None):
    manifest = tmp_path / "manifest.jsonl"
    artifact = tmp_path / "winner.bin"
    predictions = tmp_path / "predictions.jsonl"
    provenance = tmp_path / "predictions.provenance.json"
    write_jsonl(manifest, [{"id": "a"}, {"id": "b"}])
    artifact.write_bytes(b"qnn-context")

    def successful(command, timeout):
        assert timeout == 120.0
        assert command[1] == str(artifact.resolve())
        assert command[2] == str(manifest.resolve())
        write_jsonl(
            Path(command[3]),
            [
                {"id": "a", "direction": "en_to_vi", "hypothesis": "mot"},
                {"id": "b", "direction": "en_to_vi", "hypothesis": "hai"},
            ],
        )
        return 0

    result = capture_compiled_predictions(
        task="mt",
        direction="en_to_vi",
        candidate_id="mt-winner",
        adapter_manifest_sha256="a" * 64,
        runtime_kind="qnn_context_binary",
        artifact_path=artifact,
        manifest_path=manifest,
        output_predictions_path=predictions,
        output_provenance_path=provenance,
        num_beams=4,
        timeout_seconds=120.0,
        command=[
            "qnn-runtime",
            "{artifact}",
            "{manifest}",
            "{output_predictions}",
        ],
        executor=executor or successful,
        now=lambda: NOW,
        monotonic=iter([1.0, 2.5]).__next__,
    )
    return result, manifest, artifact, predictions, provenance


def test_capture_binds_runtime_artifact_manifest_predictions_and_decoding(tmp_path: Path):
    result, manifest, artifact, predictions, provenance = capture_fixture(tmp_path)

    assert provenance.is_file()
    assert result["duration_seconds"] == pytest.approx(1.5)
    assert result["decoding"] == {"num_beams": 4, "do_sample": False}
    assert result["artifact"]["sha256"] == sha256(artifact)
    assert result["manifest"]["sha256"] == sha256(manifest)
    assert result["predictions"] == {
        "path": str(predictions.resolve()),
        "bytes": predictions.stat().st_size,
        "sha256": sha256(predictions),
        "rows": 2,
    }
    assert compiled_prediction_provenance_failures(result) == []

    tampered = json.loads(json.dumps(result))
    tampered["command"]["resolved_argv"][1] = "different-artifact"
    assert "command_resolved_argv_mismatch" in compiled_prediction_provenance_failures(
        tampered
    )


def test_capture_fails_closed_and_removes_partial_predictions(tmp_path: Path):
    def failing(command, _timeout):
        write_jsonl(Path(command[3]), [{"id": "a", "hypothesis": "partial"}])
        return 9

    with pytest.raises(RuntimeError, match="return code 9"):
        capture_fixture(tmp_path, executor=failing)
    assert not (tmp_path / "predictions.jsonl").exists()
    assert not (tmp_path / "predictions.provenance.json").exists()


def test_capture_requires_each_file_placeholder_exactly_once():
    with pytest.raises(ValueError, match="artifact.*exactly once"):
        normalize_command_template(
            ["runtime", "{manifest}", "{output_predictions}"]
        )
    with pytest.raises(ValueError, match="manifest.*exactly once"):
        normalize_command_template(
            [
                "runtime",
                "{artifact}",
                "{manifest}",
                "{manifest}",
                "{output_predictions}",
            ]
        )


def test_capture_refuses_to_overwrite_existing_output(tmp_path: Path):
    (tmp_path / "predictions.jsonl").write_text("owned", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite predictions"):
        capture_fixture(tmp_path)
    assert (tmp_path / "predictions.jsonl").read_text(encoding="utf-8") == "owned"


def test_capture_rejects_duplicate_prediction_fields_and_removes_output(
    tmp_path: Path,
):
    def duplicate_output(command, _timeout):
        Path(command[3]).write_text(
            '{"id":"a","direction":"en_to_vi","hypothesis":"one",'
            '"hypothesis":"duplicate"}\n'
            '{"id":"b","direction":"en_to_vi","hypothesis":"two"}\n',
            encoding="utf-8",
        )
        return 0

    with pytest.raises(ValueError, match="strict UTF-8 JSONL"):
        capture_fixture(tmp_path, executor=duplicate_output)
    assert not (tmp_path / "predictions.jsonl").exists()
    assert not (tmp_path / "predictions.provenance.json").exists()


def test_capture_rejects_artifact_mutation_during_runtime(tmp_path: Path):
    def mutating_runtime(command, _timeout):
        Path(command[1]).write_bytes(b"changed-qnn-context")
        write_jsonl(
            Path(command[3]),
            [
                {"id": "a", "direction": "en_to_vi", "hypothesis": "mot"},
                {"id": "b", "direction": "en_to_vi", "hypothesis": "hai"},
            ],
        )
        return 0

    with pytest.raises(RuntimeError, match="changed during compiled inference"):
        capture_fixture(tmp_path, executor=mutating_runtime)
    assert not (tmp_path / "predictions.jsonl").exists()
    assert not (tmp_path / "predictions.provenance.json").exists()


def test_capture_caps_compiled_artifact_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(compiled_capture, "MAX_DEPLOYMENT_ARTIFACT_BYTES", 4)

    with pytest.raises(ValueError, match="exceeds 4 bytes"):
        capture_fixture(tmp_path)

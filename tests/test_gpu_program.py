import json
import sys
from pathlib import Path

import pytest

from scripts.run_gpu_program import (
    MT_CANDIDATE_NUM_BEAMS,
    latest_checkpoint,
    newest_complete_run,
    read_complete_run,
    record_recovered_candidate,
    record_recovered_training,
    resume_kind,
    run_stage,
    selected_adapter,
    should_extend_mt,
    validation_history,
    verified_candidate_result,
    verified_training_result,
    write_state,
)
from scripts.candidate_evidence import (
    adapter_identity,
    archive_evidence,
    canonical_sha256,
    evidence_sidecars,
    sha256,
)
from scripts.run_gpu_rounds import archive_round
from scripts.run_mt_candidate_suite import DEFAULT_NUM_BEAMS


def make_run(root: Path, *, task: str = "mt", status: str = "complete") -> Path:
    run = root / f"{task}-20261005-000000"
    model = run / "01-final" / "model"
    model.mkdir(parents=True)
    (model / "adapter_config.json").write_text("{}", encoding="utf-8")
    (run / "summary.json").write_text(
        json.dumps(
            {
                "task": task,
                "status": status,
                "selected_round": "final",
                "rounds": [{"name": "final", "status": status, "metric": 1.0}],
            }
        ),
        encoding="utf-8",
    )
    return run


def test_mt_candidate_evaluation_uses_locked_greedy_policy():
    assert MT_CANDIDATE_NUM_BEAMS == 1
    assert DEFAULT_NUM_BEAMS == 1


def test_selected_adapter_requires_completed_selected_model(tmp_path: Path):
    run = make_run(tmp_path)
    summary = read_complete_run(run, "mt")
    assert selected_adapter(run, summary) == run / "01-final" / "model"


def test_read_complete_run_rejects_failed_training(tmp_path: Path):
    run = make_run(tmp_path, status="failed")
    with pytest.raises(ValueError, match="not complete"):
        read_complete_run(run, "mt")


def test_newest_complete_run_skips_incomplete_newer_run(tmp_path: Path):
    complete = make_run(tmp_path, task="asr")
    newer = tmp_path / "asr-20261005-000001"
    newer.mkdir()
    (newer / "summary.json").write_text(
        json.dumps({"task": "asr", "status": "failed"}), encoding="utf-8"
    )
    assert newest_complete_run(tmp_path, "asr")[0] == complete


def test_run_stage_accepts_candidate_gate_failure_as_evidence(tmp_path: Path):
    state_path = tmp_path / "state.json"
    state = {"stages": {}}
    write_state(state_path, state)
    code = run_stage(
        name="candidate",
        command=[sys.executable, "-c", "raise SystemExit(2)"],
        log_path=tmp_path / "candidate.log",
        state_path=state_path,
        state=state,
        accepted_codes={0, 2},
    )
    assert code == 2
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["stages"]["candidate"]["status"] == "complete"
    assert persisted["stages"]["candidate"]["return_code"] == 2


def test_mt_extension_uses_recent_validation_trend_only(tmp_path: Path):
    model = tmp_path / "model"
    checkpoint = model / "checkpoint-3500"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "log_history": [
                    {"step": 2000, "eval_loss": 1.66},
                    {"step": 2500, "eval_loss": 1.64},
                    {"step": 3000, "eval_loss": 1.62},
                    {"step": 3500, "eval_loss": 1.60},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert latest_checkpoint(model) == checkpoint
    extend, evidence = should_extend_mt(validation_history(checkpoint))
    assert extend is True
    assert evidence["best_is_latest"] is True
    assert evidence["monotonic"] is True


def test_mt_extension_stops_on_validation_plateau():
    extend, evidence = should_extend_mt(
        [
            {"step": 2000.0, "metric": 1.60},
            {"step": 2500.0, "metric": 1.59},
            {"step": 3000.0, "metric": 1.591},
            {"step": 3500.0, "metric": 1.5905},
        ]
    )
    assert extend is False
    assert evidence["reason"] == "validation_plateau_or_regression"


def write_prediction_evidence(
    output: Path, stem: str, adapter: Path, *, task: str = "mt"
) -> None:
    prediction_path = output / f"{stem}_predictions.jsonl"
    prediction_path.write_text("{}\n", encoding="utf-8")
    specification = {"task": task, "adapter": adapter_identity(adapter)}
    provenance = {
        "schema_version": 1,
        "specification": specification,
        "specification_sha256": canonical_sha256(specification),
        "predictions": {
            "path": str(prediction_path.resolve()),
            "bytes": prediction_path.stat().st_size,
            "sha256": sha256(prediction_path),
            "rows": 1,
        },
        "decoding": {},
        "runtime": {},
    }
    prediction_path.with_suffix(prediction_path.suffix + ".provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )


def test_resume_kind_accepts_only_idle_wait_or_running_mt_candidate(tmp_path: Path):
    mt_run = tmp_path / "mt-run"
    assert resume_kind(
        {"mt_run": str(mt_run), "stage": "waiting_for_mt", "stages": {}}, mt_run
    ) == "from_wait"
    assert resume_kind(
        {
            "mt_run": str(mt_run),
            "stage": "mt_candidate",
            "stages": {"mt_candidate": {"status": "running"}},
        },
        mt_run,
    ) == "after_recovered_mt_candidate"
    assert resume_kind(
        {
            "mt_run": str(mt_run),
            "stage": "asr_training",
            "stages": {"asr_training": {"status": "running"}},
        },
        mt_run,
    ) == "after_recovered_asr_training"
    assert resume_kind(
        {
            "mt_run": str(mt_run),
            "stage": "asr_candidate",
            "stages": {"asr_candidate": {"status": "running"}},
        },
        mt_run,
    ) == "after_recovered_asr_candidate"
    with pytest.raises(ValueError, match="only safe"):
        resume_kind(
            {
                "mt_run": str(mt_run),
                "stage": "mt_candidate",
                "stages": {"mt_candidate": {"status": "error"}},
            },
            mt_run,
        )


def test_recovered_candidate_requires_verified_complete_bundle(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    output = tmp_path / "mt-candidate"
    output.mkdir()
    for stem in ("mt_candidate", "mt_clinical_candidate"):
        (output / f"{stem}.json").write_text("{}", encoding="utf-8")
        (output / f"{stem}.log").write_text("done\n", encoding="utf-8")
        write_prediction_evidence(output, stem, adapter)
        (output / f"{stem}_resource_monitor.jsonl").write_text(
            "{}\n", encoding="utf-8"
        )
    (output / "candidate_gate.json").write_text(
        json.dumps(
            {
                "status": "fail",
                "promotion_allowed": False,
                "adapter": str(adapter.resolve()),
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    archive, _ = archive_evidence(output)
    (output / "mt_candidate.log").write_text(
        "outer wrapper wrote its terminal JSON after archive creation\n", encoding="utf-8"
    )

    result = verified_candidate_result(output, task="mt", expected_adapter=adapter)

    assert result["return_code"] == 2
    assert result["local_log_mismatches"] == ["mt_candidate.log"]
    assert result["adapter_manifest_sha256"] == adapter_identity(adapter)[
        "manifest_sha256"
    ]
    state_path = tmp_path / "program_state.json"
    state = {"stage": "mt_candidate", "stages": {"mt_candidate": {"status": "running"}}}
    assert record_recovered_candidate(
        name="mt_candidate", evidence=result, state_path=state_path, state=state
    ) == 2
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["stages"]["mt_candidate"]["status"] == "complete"
    assert persisted["stages"]["mt_candidate"]["recovered_out_of_band"] is True

    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="does not bind expected adapter"):
        verified_candidate_result(output, task="mt", expected_adapter=adapter)
    weights.write_bytes(b"adapter-weights")

    checksum, _ = evidence_sidecars(archive)
    checksum.write_text("0" * 64 + f"  {archive.name}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verified_candidate_result(output, task="mt", expected_adapter=adapter)


def test_recovered_candidate_rejects_mutated_prediction(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "mt-candidate"
    output.mkdir()
    for stem in ("mt_candidate", "mt_clinical_candidate"):
        (output / f"{stem}.json").write_text("{}", encoding="utf-8")
        (output / f"{stem}.log").write_text("done\n", encoding="utf-8")
        write_prediction_evidence(output, stem, adapter)
        (output / f"{stem}_resource_monitor.jsonl").write_text("{}\n", encoding="utf-8")
    (output / "candidate_gate.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "promotion_allowed": True,
                "adapter": str(adapter.resolve()),
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    archive_evidence(output)
    (output / "mt_candidate_predictions.jsonl").write_text(
        '{"tampered": true}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not match archive"):
        verified_candidate_result(output, task="mt", expected_adapter=adapter)


def test_recovered_training_requires_verified_round_archives(tmp_path: Path):
    output_root = tmp_path / "asr-runs"
    run = output_root / "asr-20261006-000000"
    round_dir = run / "01-final"
    model = round_dir / "model"
    model.mkdir(parents=True)
    (model / "adapter_config.json").write_text("{}", encoding="utf-8")
    (model / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    (model / "training_run.json").write_text("{}", encoding="utf-8")
    archive = archive_round(round_dir)
    (run / "summary.json").write_text(
        json.dumps(
            {
                "task": "asr",
                "status": "complete",
                "selected_round": "final",
                "rounds": [
                    {
                        "name": "final",
                        "status": "complete",
                        "metric": 0.15,
                        "archive": archive,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    evidence = verified_training_result(output_root, task="asr")

    assert Path(evidence["adapter"]) == model
    assert evidence["adapter_file_count"] == 3
    assert evidence["adapter_content_bytes"] == sum(
        path.stat().st_size for path in model.iterdir()
    )
    assert evidence["verified_archives"][0]["sha256"] == archive["sha256"]
    state_path = tmp_path / "program_state.json"
    state = {"stage": "asr_training", "stages": {"asr_training": {"status": "running"}}}
    record_recovered_training(
        name="asr_training", evidence=evidence, state_path=state_path, state=state
    )
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["stages"]["asr_training"]["recovered_out_of_band"] is True
    assert persisted["stages"]["asr_training"]["adapter_file_count"] == 3

    unexpected = model / "unarchived.bin"
    unexpected.write_bytes(b"not in archive")
    with pytest.raises(ValueError, match="live directory does not match verified archive"):
        verified_training_result(output_root, task="asr")
    unexpected.unlink()

    weights = model / "adapter_model.safetensors"
    weights.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="live file does not match verified archive"):
        verified_training_result(output_root, task="asr")
    weights.write_bytes(b"adapter-weights")

    Path(archive["checksum"]).write_text(
        "0" * 64 + f"  {Path(archive['path']).name}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        verified_training_result(output_root, task="asr")

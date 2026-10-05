from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from scripts.run_gpu_rounds import (
    active_utilization_limits,
    apply_round_overrides,
    archive_round,
    checkpoint_watcher_command,
    cli_args,
    configured_rounds,
    resolve_adaptive_final,
    select_completed_round,
    utilization_throttle_reason,
    validate_resource_limits,
)
from scripts.candidate_evidence import verify_evidence_archive


ROOT = Path(__file__).resolve().parents[1]


def test_gpu_round_policy_keeps_uniform_headroom_below_75_percent():
    limits = yaml.safe_load((ROOT / "configs" / "gpu_rounds.yaml").read_text(encoding="utf-8"))["limits"]
    timezone = ZoneInfo("Asia/Bangkok")
    assert active_utilization_limits(limits, datetime(2026, 10, 4, 1, 59, tzinfo=timezone)) == (
        70.0,
        55.0,
        False,
    )
    assert active_utilization_limits(limits, datetime(2026, 10, 4, 2, 0, tzinfo=timezone)) == (
        70.0,
        55.0,
        False,
    )
    assert active_utilization_limits(limits, datetime(2026, 10, 4, 8, 59, tzinfo=timezone)) == (
        70.0,
        55.0,
        False,
    )
    assert active_utilization_limits(limits, datetime(2026, 10, 4, 9, 0, tzinfo=timezone)) == (
        70.0,
        55.0,
        False,
    )
    assert (
        0.0
        < limits["resume_percent"]
        < limits["utilization_percent"]
        < limits["hard_utilization_percent"]
        < 75.0
    )
    assert validate_resource_limits(limits) == (0.35, 0.40)
    assert limits["sample_seconds"] == 1.0


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("gpu_memory_fraction", 0.3501, "gpu_memory_fraction"),
        ("gpu_memory_hard_fraction", 0.4001, "memory thresholds"),
        ("gpu_memory_hard_fraction", 0.34, "memory thresholds"),
        ("utilization_percent", math.nan, "finite"),
        ("sample_seconds", 0.0, "sample_seconds"),
        ("rolling_samples", 0, "rolling_samples"),
    ],
)
def test_gpu_round_resource_limits_fail_closed(key: str, value: float, message: str):
    limits = yaml.safe_load((ROOT / "configs" / "gpu_rounds.yaml").read_text(encoding="utf-8"))[
        "limits"
    ]
    limits[key] = value
    with pytest.raises(ValueError, match=message):
        validate_resource_limits(limits)


def test_gpu_round_hard_utilization_guard_preempts_a_low_rolling_average():
    common = {
        "sample_count": 10,
        "rolling_samples": 10,
        "utilization_limit": 70.0,
        "hard_utilization_limit": 74.0,
    }
    assert (
        utilization_throttle_reason(current=74.0, rolling=20.0, **common)
        == "hard_utilization"
    )
    assert (
        utilization_throttle_reason(current=73.0, rolling=70.1, **common)
        == "rolling_utilization"
    )
    assert utilization_throttle_reason(current=73.0, rolling=70.0, **common) is None


def test_gpu_round_cli_args_preserve_false_and_skip_none():
    assert cli_args({"method": "lora", "enabled": True, "disabled": False, "empty": None}) == [
        "--method",
        "lora",
        "--enabled",
    ]


def test_gpu_pilots_bound_preprocessing_without_shrinking_final_run_policy():
    config = yaml.safe_load((ROOT / "configs" / "gpu_rounds.yaml").read_text(encoding="utf-8"))
    assert config["tasks"]["mt"]["common_args"]["limit_train"] == 8192
    assert config["tasks"]["asr"]["common_args"]["limit_train"] == 6144


def test_gpu_rounds_can_select_one_named_round_for_immediate_download():
    task = {"rounds": [{"name": "one"}, {"name": "two"}]}
    assert configured_rounds(task, round_name="two") == [{"name": "two"}]


def test_gpu_round_can_resume_selected_round_with_extended_epoch_budget(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    result = apply_round_overrides(
        [{"name": "final", "epochs": 1.0, "max_steps": -1}],
        round_name="final",
        resume_from_checkpoint=checkpoint,
        initial_adapter=None,
        epochs=3.0,
        learning_rate=None,
    )
    assert result == [
        {
            "name": "final",
            "epochs": 3.0,
            "max_steps": -1,
            "resume_from_checkpoint": str(checkpoint.resolve()),
        }
    ]


def test_gpu_round_can_warm_start_adapter_with_fresh_lower_learning_rate(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    result = apply_round_overrides(
        [{"name": "final", "epochs": 1.0, "learning_rate": 1e-4}],
        round_name="final",
        resume_from_checkpoint=None,
        initial_adapter=adapter,
        epochs=2.0,
        learning_rate=5e-5,
    )
    assert result[0]["initial_adapter"] == str(adapter.resolve())
    assert result[0]["epochs"] == 2.0
    assert result[0]["learning_rate"] == 5e-5


def test_mt_full_round_uses_selected_pilot_and_all_training_rows():
    config = yaml.safe_load((ROOT / "configs" / "gpu_rounds.yaml").read_text(encoding="utf-8"))
    final = next(item for item in config["tasks"]["mt"]["rounds"] if item["name"] == "final-r8-lr1e4-full")
    assert final["lora_rank"] == 8
    assert final["learning_rate"] == 1.0e-4
    assert final["limit_train"] == 0
    assert final["batch_size"] * final["gradient_accumulation_steps"] == 32


def test_adaptive_final_inherits_best_pilot_and_applies_full_data_overrides():
    task = {
        "greater_is_better": False,
        "rounds": [
            {"name": "one", "learning_rate": 1e-4, "lora_rank": 8},
            {"name": "two", "learning_rate": 5e-5, "lora_rank": 16},
        ],
        "adaptive_final": {
            "name": "final-selected-full",
            "source_rounds": ["one", "two"],
            "inherit": ["learning_rate", "lora_rank"],
            "overrides": {"max_steps": -1, "limit_train": 0},
        },
    }
    resolved, source = resolve_adaptive_final(
        task,
        [
            {"name": "one", "status": "complete", "metric": 0.22},
            {"name": "two", "status": "complete", "metric": 0.18},
        ],
    )
    assert source == "two"
    assert resolved == {
        "name": "final-selected-full",
        "learning_rate": 5e-5,
        "lora_rank": 16,
        "max_steps": -1,
        "limit_train": 0,
    }


def test_adaptive_selection_ignores_failed_and_out_of_scope_rounds():
    selected = select_completed_round(
        [
            {"name": "failed", "status": "failed", "metric": 0.1},
            {"name": "pilot", "status": "complete", "metric": 0.2},
            {"name": "other", "status": "complete", "metric": 0.15},
        ],
        greater_is_better=False,
        allowed_names={"failed", "pilot"},
    )
    assert selected and selected["name"] == "pilot"


def test_asr_adaptive_final_preserves_effective_batch_and_uses_full_train():
    config = yaml.safe_load((ROOT / "configs" / "gpu_rounds.yaml").read_text(encoding="utf-8"))
    assert config["tasks"]["asr"]["common_args"]["limit_validation"] == 512
    adaptive = config["tasks"]["asr"]["adaptive_final"]
    overrides = adaptive["overrides"]
    assert overrides["limit_train"] == 0
    assert overrides["batch_size"] * overrides["gradient_accumulation_steps"] == 32
    assert set(adaptive["source_rounds"]) == {
        "pilot-r8-lr1e4",
        "pilot-r16-lr5e5",
        "pilot-r32-lr2e5",
    }


def test_asr_pilot_validation_window_covers_medical_and_code_switch_sources():
    manifest = ROOT / "data" / "processed" / "manifests" / "asr--validation-local.jsonl"
    with manifest.open(encoding="utf-8") as handle:
        rows = [json.loads(next(handle)) for _ in range(512)]
    assert {row["source"] for row in rows} == {"vietmed", "vimedcss"}
    assert {row["language"] for row in rows} == {"vi", "vi-code-switch"}
    assert sum(row["language"] == "vi-code-switch" for row in rows) >= 128


def test_gpu_round_starts_checkpoint_watcher_for_the_training_pid(tmp_path: Path):
    command = checkpoint_watcher_command(
        "python",
        tmp_path / "model",
        tmp_path / "archives",
        training_pid=123,
        poll_seconds=15.0,
    )
    assert command[-4:] == ["--watch-pid", "123", "--poll-seconds", "15.0"]
    assert Path(command[1]).name == "watch_checkpoints.py"


def test_gpu_round_archive_has_verified_checksum_and_content_manifest(tmp_path: Path):
    round_dir = tmp_path / "01-pilot"
    model_dir = round_dir / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (round_dir / "training.log").write_text("complete\n", encoding="utf-8")

    record = archive_round(round_dir)

    archive = Path(record["path"])
    manifest = verify_evidence_archive(archive)
    assert record["sha256"] == manifest["archive_sha256"]
    assert record["bytes"] == manifest["archive_bytes"]
    assert Path(record["checksum"]).is_file()
    assert Path(record["manifest"]).is_file()
    assert {item["path"] for item in manifest["files"]} == {
        "01-pilot/model/adapter_config.json",
        "01-pilot/training.log",
    }

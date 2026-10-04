from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from scripts.run_gpu_rounds import (
    active_utilization_limits,
    checkpoint_watcher_command,
    cli_args,
    configured_rounds,
)


ROOT = Path(__file__).resolve().parents[1]


def test_gpu_round_schedule_raises_utilization_only_from_02_to_09_bangkok():
    limits = yaml.safe_load((ROOT / "configs" / "gpu_rounds.yaml").read_text(encoding="utf-8"))["limits"]
    timezone = ZoneInfo("Asia/Bangkok")
    assert active_utilization_limits(limits, datetime(2026, 10, 4, 1, 59, tzinfo=timezone)) == (
        38.0,
        24.0,
        False,
    )
    assert active_utilization_limits(limits, datetime(2026, 10, 4, 2, 0, tzinfo=timezone)) == (
        75.0,
        55.0,
        True,
    )
    assert active_utilization_limits(limits, datetime(2026, 10, 4, 8, 59, tzinfo=timezone)) == (
        75.0,
        55.0,
        True,
    )
    assert active_utilization_limits(limits, datetime(2026, 10, 4, 9, 0, tzinfo=timezone)) == (
        38.0,
        24.0,
        False,
    )


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


def test_mt_full_round_uses_selected_pilot_and_all_training_rows():
    config = yaml.safe_load((ROOT / "configs" / "gpu_rounds.yaml").read_text(encoding="utf-8"))
    final = next(item for item in config["tasks"]["mt"]["rounds"] if item["name"] == "final-r8-lr1e4-full")
    assert final["lora_rank"] == 8
    assert final["learning_rate"] == 1.0e-4
    assert final["limit_train"] == 0
    assert final["batch_size"] * final["gradient_accumulation_steps"] == 32


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

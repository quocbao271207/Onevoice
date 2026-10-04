from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from scripts.run_gpu_rounds import active_utilization_limits, cli_args


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

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

from scripts.run_gpu_rounds import (
    active_utilization_limits,
    apply_round_overrides,
    archive_round,
    attributed_process_gpu_memory,
    checkpoint_watcher_command,
    cli_args,
    configured_rounds,
    gpu_process_memory_by_pid,
    exact_process_memory_attribution,
    gpu_spawn_capacity,
    historical_peak_process_memory,
    process_pid_aliases,
    resolve_adaptive_final,
    select_completed_round,
    utilization_throttle_reason,
    validate_resource_limits,
    wait_for_gpu_spawn_capacity,
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


def test_gpu_spawn_capacity_requires_room_for_the_hard_process_budget():
    limits = yaml.safe_load((ROOT / "configs" / "gpu_rounds.yaml").read_text(encoding="utf-8"))[
        "limits"
    ]
    sample = {
        "utilization": 20.0,
        "memory_total_mib": 40_960.0,
        "memory_used_mib": 31_734.0,
    }

    capacity = gpu_spawn_capacity(sample, limits)

    assert capacity["allowed"] is False
    assert capacity["reason"] == "insufficient_memory_headroom"
    assert capacity["required_free_memory_mib"] == 16_384.0


def test_gpu_spawn_capacity_uses_resume_threshold_before_launch():
    limits = yaml.safe_load((ROOT / "configs" / "gpu_rounds.yaml").read_text(encoding="utf-8"))[
        "limits"
    ]
    sample = {
        "utilization": 55.1,
        "memory_total_mib": 40_960.0,
        "memory_used_mib": 10_000.0,
    }

    assert gpu_spawn_capacity(sample, limits)["reason"] == "utilization_above_resume"
    sample["utilization"] = 55.0
    assert gpu_spawn_capacity(sample, limits)["allowed"] is True


def test_gpu_spawn_capacity_uses_historical_peak_with_headroom_for_resume():
    limits = yaml.safe_load((ROOT / "configs" / "gpu_rounds.yaml").read_text(encoding="utf-8"))[
        "limits"
    ]
    sample = {
        "utilization": 10.0,
        "memory_total_mib": 40_960.0,
        "memory_used_mib": 30_900.0,
    }

    capacity = gpu_spawn_capacity(
        sample,
        limits,
        historical_peak_process_memory_mib=8_000.0,
    )

    assert capacity["capacity_strategy"] == "historical_peak_with_headroom"
    assert capacity["required_free_memory_mib"] == 10_048.0
    assert capacity["hard_memory_budget_mib"] == 16_384.0
    assert capacity["allowed"] is True


def test_historical_peak_requires_a_sustained_prior_attempt(tmp_path: Path):
    short = tmp_path / "mt-short" / "01-pilot" / "resource_monitor.jsonl"
    short.parent.mkdir(parents=True)
    short.write_text(
        "".join(
            json.dumps(
                {
                    "process_memory_mib": 20_000.0,
                    "process_memory_attribution": "process_group_pid",
                }
            )
            + "\n"
            for _ in range(2)
        ),
        encoding="utf-8",
    )
    sustained = tmp_path / "mt-sustained" / "01-pilot" / "resource_monitor.jsonl"
    sustained.parent.mkdir(parents=True)
    sustained.write_text(
        "".join(
            json.dumps(
                {
                    "process_memory_mib": float(8_000 + index),
                    "process_memory_attribution": "process_namespace_pid",
                }
            )
            + "\n"
            for index in range(50)
        ),
        encoding="utf-8",
    )

    assert historical_peak_process_memory(tmp_path) == 8_049.0


def test_historical_peak_rejects_post_spawn_telemetry(tmp_path: Path):
    monitor = tmp_path / "mt-run" / "01-pilot" / "resource_monitor.jsonl"
    monitor.parent.mkdir(parents=True)
    monitor.write_text(
        "".join(
            json.dumps(
                {
                    "process_memory_mib": 24_714.0,
                    "process_memory_attribution": "post_spawn_memory_growth",
                    "hard_memory_enforced_from_nvidia_pid": False,
                }
            )
            + "\n"
            for _ in range(100)
        ),
        encoding="utf-8",
    )

    assert historical_peak_process_memory(tmp_path) is None


def test_gpu_spawn_capacity_waiter_retries_and_records_evidence(tmp_path: Path):
    limits = yaml.safe_load((ROOT / "configs" / "gpu_rounds.yaml").read_text(encoding="utf-8"))[
        "limits"
    ]
    samples = iter(
        [
            {
                "utilization": 10.0,
                "memory_total_mib": 40_960.0,
                "memory_used_mib": 31_000.0,
            },
            {
                "utilization": 10.0,
                "memory_total_mib": 40_960.0,
                "memory_used_mib": 20_000.0,
            },
        ]
    )
    sleeps: list[float] = []
    evidence = tmp_path / "pre_spawn_capacity.jsonl"

    result = wait_for_gpu_spawn_capacity(
        limits,
        evidence,
        sample_reader=lambda: next(samples),
        sleeper=sleeps.append,
        poll_seconds=2.0,
    )

    records = [json.loads(line) for line in evidence.read_text(encoding="utf-8").splitlines()]
    assert result["samples"] == 2
    assert sleeps == [2.0]
    assert [record["allowed"] for record in records] == [False, True]
    assert records[0]["reason"] == "insufficient_memory_headroom"


def test_gpu_spawn_capacity_waiter_fails_closed_on_sample_error(tmp_path: Path):
    limits = yaml.safe_load((ROOT / "configs" / "gpu_rounds.yaml").read_text(encoding="utf-8"))[
        "limits"
    ]
    outcomes = iter(
        [
            OSError("nvidia-smi unavailable"),
            {
                "utilization": 5.0,
                "memory_total_mib": 40_960.0,
                "memory_used_mib": 20_000.0,
            },
        ]
    )

    def sample_reader() -> dict[str, float]:
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    evidence = tmp_path / "pre_spawn_capacity.jsonl"
    result = wait_for_gpu_spawn_capacity(
        limits,
        evidence,
        sample_reader=sample_reader,
        sleeper=lambda _: None,
        poll_seconds=1.0,
    )

    records = [json.loads(line) for line in evidence.read_text(encoding="utf-8").splitlines()]
    assert result["samples"] == 2
    assert records[0]["reason"] == "gpu_sample_error"
    assert records[0]["allowed"] is False


def test_gpu_spawn_capacity_waiter_rejects_invalid_historical_peak(tmp_path: Path):
    limits = yaml.safe_load((ROOT / "configs" / "gpu_rounds.yaml").read_text(encoding="utf-8"))[
        "limits"
    ]

    with pytest.raises(ValueError, match="Historical GPU process-memory peak"):
        wait_for_gpu_spawn_capacity(
            limits,
            tmp_path / "pre_spawn_capacity.jsonl",
            historical_peak_process_memory_mib=math.nan,
        )


def test_gpu_process_memory_parser_ignores_malformed_rows(monkeypatch: pytest.MonkeyPatch):
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="101, 3130\ninvalid\n102, N/A\n103, 512.5\n",
        stderr="",
    )
    monkeypatch.setattr("scripts.run_gpu_rounds.subprocess.run", lambda *args, **kwargs: completed)

    assert gpu_process_memory_by_pid() == {101: 3130.0, 103: 512.5}


def test_gpu_memory_prefers_exact_process_group_pids(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("scripts.run_gpu_rounds.process_group_pids", lambda group_id: {10, 11})
    monkeypatch.setattr(
        "scripts.run_gpu_rounds.gpu_process_memory_by_pid",
        lambda: {10: 100.0, 20: 4000.0},
    )

    sample = attributed_process_gpu_memory(10, {20: 3000.0})

    assert sample == {
        "memory_mib": 100.0,
        "attribution": "process_group_pid",
        "gpu_pids": [10],
        "process_pids": [10, 11],
    }


def test_process_pid_aliases_reads_linux_namespace_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    status = tmp_path / "356822" / "status"
    status.parent.mkdir()
    status.write_text(
        "Name:\tpython\nNSpid:\t451951\t356822\nState:\tR (running)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.run_gpu_rounds.os.name", "posix")

    assert process_pid_aliases({356822}, tmp_path) == {356822, 451951}


def test_gpu_memory_prefers_namespace_pid_over_unrelated_new_gpu_jobs(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "scripts.run_gpu_rounds.process_group_pids", lambda group_id: {356822}
    )
    monkeypatch.setattr(
        "scripts.run_gpu_rounds.process_pid_aliases",
        lambda pids: {*pids, 451951},
    )
    monkeypatch.setattr(
        "scripts.run_gpu_rounds.gpu_process_memory_by_pid",
        lambda: {451951: 3130.0, 489181: 4678.0, 494517: 1626.0},
    )

    sample = attributed_process_gpu_memory(356820, {})

    assert sample == {
        "memory_mib": 3130.0,
        "attribution": "process_namespace_pid",
        "gpu_pids": [451951],
        "process_pids": [356822],
    }


def test_gpu_memory_uses_only_post_spawn_pids_when_namespaces_differ(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("scripts.run_gpu_rounds.process_group_pids", lambda group_id: {10, 11})
    monkeypatch.setattr(
        "scripts.run_gpu_rounds.gpu_process_memory_by_pid",
        lambda: {1001: 3130.0, 1002: 4678.0},
    )

    sample = attributed_process_gpu_memory(10, {1002: 4200.0})

    assert sample == {
        "memory_mib": 3130.0,
        "attribution": "post_spawn_pid",
        "gpu_pids": [1001],
        "process_pids": [10, 11],
    }


def test_gpu_memory_uses_positive_growth_when_host_pid_is_stable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("scripts.run_gpu_rounds.process_group_pids", lambda group_id: {10})
    monkeypatch.setattr(
        "scripts.run_gpu_rounds.gpu_process_memory_by_pid",
        lambda: {1001: 4096.0, 1002: 2048.0, 1003: 512.0},
    )

    sample = attributed_process_gpu_memory(
        10,
        {1001: 1024.0, 1002: 2048.0, 1003: 1024.0},
    )

    assert sample == {
        "memory_mib": 3072.0,
        "attribution": "post_spawn_memory_growth",
        "gpu_pids": [1001],
        "process_pids": [10],
    }


def test_gpu_memory_without_pid_match_or_baseline_is_unattributed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("scripts.run_gpu_rounds.process_group_pids", lambda group_id: {10})
    monkeypatch.setattr(
        "scripts.run_gpu_rounds.gpu_process_memory_by_pid",
        lambda: {1001: 3130.0},
    )

    sample = attributed_process_gpu_memory(10)

    assert sample["memory_mib"] == 0.0
    assert sample["attribution"] == "unattributed"
    assert sample["gpu_pids"] == []


@pytest.mark.parametrize(
    ("attribution", "expected"),
    [
        ("process_group_pid", True),
        ("process_namespace_pid", True),
        ("post_spawn_pid", False),
        ("post_spawn_memory_growth", False),
        ("unattributed", False),
    ],
)
def test_hard_memory_kill_requires_exact_process_attribution(
    attribution: str, expected: bool
):
    assert exact_process_memory_attribution({"attribution": attribution}) is expected


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

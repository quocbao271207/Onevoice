from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.run_baseline_benchmarks import (
    attach_adapter,
    model_load_kwargs,
    prepare_runtime,
    resolve_device,
    source_balanced_sample,
)
from scripts.run_mt_candidate_suite import candidate_checks


def test_explicit_benchmark_device_is_preserved():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_cpu_candidate_benchmark_rejects_reduced_precision():
    with pytest.raises(ValueError, match="only fp32"):
        model_load_kwargs("cpu", "bf16")


def test_missing_candidate_adapter_fails_closed(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Missing adapter"):
        attach_adapter(object(), tmp_path / "missing-adapter")


def test_candidate_benchmark_memory_fraction_cannot_exceed_40_percent():
    args = SimpleNamespace(device="cpu", gpu_memory_fraction=0.4001)
    with pytest.raises(ValueError, match=r"\(0, 0\.40\]"):
        prepare_runtime(args)


def test_cuda_dtype_selection_is_explicit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert model_load_kwargs("cuda", "bf16") == {"torch_dtype": torch.bfloat16}


def test_source_balanced_sample_can_take_full_manifest():
    rows = [
        {"id": "a", "source": "one"},
        {"id": "b", "source": "two"},
        {"id": "c", "source": "one"},
    ]
    selected = source_balanced_sample(rows, len(rows), seed=7)
    assert {row["id"] for row in selected} == {"a", "b", "c"}


def test_candidate_gate_requires_every_clinical_slice():
    config = {
        "release_gates": {
            "aggregate": {"mt_sacrebleu_min": 25.0, "mt_chrf2_min": 46.0},
            "clinical_safety": {"terminology_failure_rate_max": 0.0},
        },
        "evaluation": {"required_slices": ["drug_name", "dose"]},
    }
    checks = candidate_checks(
        config,
        {"sacrebleu": 30.0, "chrf2": 50.0},
        {"categories": {"drug_name": {"safety_failure_rate": 0.0}}},
    )
    assert any(check["name"] == "clinical_dose" and not check["pass"] for check in checks)

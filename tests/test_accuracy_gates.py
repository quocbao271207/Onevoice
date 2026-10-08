import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts import run_accuracy_gates


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return path


def gate_fixture(tmp_path: Path) -> dict[str, Path]:
    config = tmp_path / "accuracy.yaml"
    config.write_text(
        """release_gates:
  aggregate:
    asr_vi_wer_max: 0.2
    asr_en_wer_max: 0.2
    mt_sacrebleu_min: 25.0
    mt_chrf2_min: 46.0
  slices:
    asr_code_switch_wer_max: 0.2
  clinical_safety:
    drug_failure_rate_max: 0.0
    terminology_failure_rate_max: 0.0
evaluation:
  required_slices: [drug]
""",
        encoding="utf-8",
    )
    return {
        "config": config,
        "asr_vi": write_json(
            tmp_path / "asr-vi.json",
            {
                "wer": 0.1,
                "slices": {"code_switch": {"True": {"wer": 0.1}}},
            },
        ),
        "asr_en": write_json(tmp_path / "asr-en.json", {"wer": 0.1}),
        "mt": write_json(
            tmp_path / "mt.json",
            {"sacrebleu": 30.0, "chrf2": 50.0},
        ),
        "clinical": write_json(
            tmp_path / "clinical.json",
            {"categories": {"drug": {"safety_failure_rate": 0.0}}},
        ),
        "output": tmp_path / "release-gate.json",
    }


def run_gate(monkeypatch: pytest.MonkeyPatch, paths: dict[str, Path]) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_accuracy_gates.py",
            "--config",
            str(paths["config"]),
            "--asr-vi",
            str(paths["asr_vi"]),
            "--asr-en",
            str(paths["asr_en"]),
            "--mt",
            str(paths["mt"]),
            "--clinical-mt",
            str(paths["clinical"]),
            "--output",
            str(paths["output"]),
        ],
    )
    return run_accuracy_gates.main()


def test_accuracy_gate_binds_strict_inputs_and_publishes_durably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = gate_fixture(tmp_path)

    assert run_gate(monkeypatch, paths) == 0

    report = json.loads(paths["output"].read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["promotion_allowed"] is True
    assert set(report["input_evidence"]) == {
        "asr_vi",
        "asr_en",
        "mt",
        "clinical_mt",
    }
    for name, path_key in (
        ("asr_vi", "asr_vi"),
        ("asr_en", "asr_en"),
        ("mt", "mt"),
        ("clinical_mt", "clinical"),
    ):
        payload = paths[path_key].read_bytes()
        assert report["input_evidence"][name] == {
            "path": str(paths[path_key].resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    assert not list(tmp_path.glob(".release-gate.json.*.tmp"))


@pytest.mark.parametrize(
    "unsafe_payload",
    [
        '{"sacrebleu": 30.0, "sacrebleu": 31.0, "chrf2": 50.0}',
        '{"sacrebleu": NaN, "chrf2": 50.0}',
    ],
)
def test_accuracy_gate_rejects_ambiguous_or_nonfinite_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_payload: str,
):
    paths = gate_fixture(tmp_path)
    paths["mt"].write_text(unsafe_payload, encoding="utf-8")

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        run_gate(monkeypatch, paths)
    assert not paths["output"].exists()


@pytest.mark.parametrize("unsafe_value", ["30.0", True])
def test_accuracy_gate_requires_numeric_metric_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_value: object,
):
    paths = gate_fixture(tmp_path)
    write_json(
        paths["mt"],
        {"sacrebleu": unsafe_value, "chrf2": 50.0},
    )

    with pytest.raises(ValueError, match="mt_sacrebleu actual must be a number"):
        run_gate(monkeypatch, paths)
    assert not paths["output"].exists()


def test_accuracy_gate_rejects_overflowed_finite_number(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = gate_fixture(tmp_path)
    paths["mt"].write_text(
        '{"sacrebleu": 1e400, "chrf2": 50.0}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        run_gate(monkeypatch, paths)
    assert not paths["output"].exists()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"sacrebleu": 101.0, "chrf2": 50.0}, "within 0.0..100.0"),
        ({"sacrebleu": -1.0, "chrf2": 50.0}, "within 0.0..100.0"),
    ],
)
def test_accuracy_gate_rejects_out_of_domain_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict,
    message: str,
):
    paths = gate_fixture(tmp_path)
    write_json(paths["mt"], payload)

    with pytest.raises(ValueError, match=message):
        run_gate(monkeypatch, paths)
    assert not paths["output"].exists()

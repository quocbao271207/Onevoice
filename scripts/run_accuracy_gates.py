"""Combine aggregate and clinical-slice reports into one fail-closed release gate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.durable_json import write_durable_json  # noqa: E402
from src.pipeline.stable_json import (  # noqa: E402
    StableJsonDocument,
    read_stable_json_mapping,
)
from src.pipeline.stable_yaml import read_stable_yaml_mapping  # noqa: E402


MAX_ACCURACY_CONFIG_BYTES = 1_000_000
MAX_ACCURACY_INPUT_BYTES = 50_000_000
MAX_ACCURACY_GATE_BYTES = 5_000_000


def load_accuracy_input(path: Path, *, label: str) -> StableJsonDocument:
    return read_stable_json_mapping(
        path,
        maximum_bytes=MAX_ACCURACY_INPUT_BYTES,
        label=label,
    )


def bounded_number(
    value: object,
    *,
    label: str,
    minimum: float,
    maximum: float | None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if number < minimum or (maximum is not None and number > maximum):
        upper = "infinity" if maximum is None else str(maximum)
        raise ValueError(f"{label} must be within {minimum}..{upper}")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "accuracy_program.yaml")
    parser.add_argument("--asr-vi", type=Path, default=ROOT / "data" / "reports" / "baselines" / "asr_vi_base.json")
    parser.add_argument("--asr-en", type=Path, default=ROOT / "data" / "reports" / "baselines" / "asr_en_base.json")
    parser.add_argument("--mt", type=Path, default=ROOT / "data" / "reports" / "baselines" / "mt_base.json")
    parser.add_argument(
        "--clinical-mt",
        type=Path,
        default=ROOT / "data" / "reports" / "accuracy" / "mt_clinical_base.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "reports" / "accuracy" / "release_gate.json",
    )
    args = parser.parse_args()

    config = read_stable_yaml_mapping(
        args.config,
        maximum_bytes=MAX_ACCURACY_CONFIG_BYTES,
        label="Accuracy gate config",
    ).mapping
    thresholds = config["release_gates"]
    input_documents = {
        "asr_vi": load_accuracy_input(args.asr_vi, label="ASR VI accuracy input"),
        "asr_en": load_accuracy_input(args.asr_en, label="ASR EN accuracy input"),
        "mt": load_accuracy_input(args.mt, label="MT accuracy input"),
        "clinical_mt": load_accuracy_input(
            args.clinical_mt,
            label="Clinical MT accuracy input",
        ),
    }
    asr_vi = input_documents["asr_vi"].mapping
    asr_en = input_documents["asr_en"].mapping
    mt = input_documents["mt"].mapping
    clinical = input_documents["clinical_mt"].mapping

    checks: list[dict[str, Any]] = []

    def maximum(
        name: str,
        actual: object,
        limit: object,
        *,
        actual_upper: float | None,
        limit_upper: float,
    ) -> None:
        actual_number = bounded_number(
            actual,
            label=f"{name} actual",
            minimum=0.0,
            maximum=actual_upper,
        )
        limit_number = bounded_number(
            limit,
            label=f"{name} limit",
            minimum=0.0,
            maximum=limit_upper,
        )
        checks.append(
            {
                "name": name,
                "actual": actual_number,
                "operator": "<=",
                "limit": limit_number,
                "pass": actual_number <= limit_number,
            }
        )

    def minimum(name: str, actual: object, limit: object) -> None:
        actual_number = bounded_number(
            actual,
            label=f"{name} actual",
            minimum=0.0,
            maximum=100.0,
        )
        limit_number = bounded_number(
            limit,
            label=f"{name} limit",
            minimum=0.0,
            maximum=100.0,
        )
        checks.append(
            {
                "name": name,
                "actual": actual_number,
                "operator": ">=",
                "limit": limit_number,
                "pass": actual_number >= limit_number,
            }
        )

    aggregate = thresholds["aggregate"]
    maximum(
        "asr_vi_wer",
        asr_vi["wer"],
        aggregate["asr_vi_wer_max"],
        actual_upper=None,
        limit_upper=1.0,
    )
    maximum(
        "asr_en_wer",
        asr_en["wer"],
        aggregate["asr_en_wer_max"],
        actual_upper=None,
        limit_upper=1.0,
    )
    minimum("mt_sacrebleu", mt["sacrebleu"], aggregate["mt_sacrebleu_min"])
    minimum("mt_chrf2", mt["chrf2"], aggregate["mt_chrf2_min"])
    code_switch = asr_vi.get("slices", {}).get("code_switch", {}).get("True")
    if code_switch is None:
        checks.append(
            {
                "name": "asr_code_switch_wer",
                "pass": False,
                "reason": "missing required slice",
            }
        )
    else:
        maximum(
            "asr_code_switch_wer",
            code_switch["wer"],
            thresholds["slices"]["asr_code_switch_wer_max"],
            actual_upper=None,
            limit_upper=1.0,
        )

    clinical_thresholds = thresholds["clinical_safety"]
    for category in config["evaluation"]["required_slices"]:
        category_report = clinical.get("categories", {}).get(category)
        key = f"{category}_failure_rate_max"
        if key not in clinical_thresholds:
            key = "terminology_failure_rate_max"
        if category_report is None:
            checks.append(
                {
                    "name": f"clinical_{category}",
                    "pass": False,
                    "reason": "missing required slice",
                }
            )
            continue
        maximum(
            f"clinical_{category}_failure_rate",
            category_report["safety_failure_rate"],
            clinical_thresholds[key],
            actual_upper=1.0,
            limit_upper=1.0,
        )

    report = {
        "status": "pass" if all(check["pass"] for check in checks) else "fail",
        "promotion_allowed": all(check["pass"] for check in checks),
        "checks": checks,
        "inputs": {
            "config": str(args.config),
            "asr_vi": str(args.asr_vi),
            "asr_en": str(args.asr_en),
            "mt": str(args.mt),
            "clinical_mt": str(args.clinical_mt),
        },
        "input_evidence": {
            name: {
                "path": str(document.path),
                "sha256": document.sha256,
                "bytes": document.bytes,
            }
            for name, document in input_documents.items()
        },
        "policy": "Aggregate quality and every critical clinical slice must pass.",
    }
    write_durable_json(
        args.output,
        report,
        maximum_bytes=MAX_ACCURACY_GATE_BYTES,
        label="Accuracy release gate",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["promotion_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

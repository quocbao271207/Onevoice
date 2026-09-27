"""Combine aggregate and clinical-slice reports into one fail-closed release gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    thresholds = config["release_gates"]
    asr_vi = load_json(args.asr_vi)
    asr_en = load_json(args.asr_en)
    mt = load_json(args.mt)
    clinical = load_json(args.clinical_mt)

    checks: list[dict[str, Any]] = []

    def maximum(name: str, actual: float, limit: float) -> None:
        checks.append({"name": name, "actual": actual, "operator": "<=", "limit": limit, "pass": actual <= limit})

    def minimum(name: str, actual: float, limit: float) -> None:
        checks.append({"name": name, "actual": actual, "operator": ">=", "limit": limit, "pass": actual >= limit})

    aggregate = thresholds["aggregate"]
    maximum("asr_vi_wer", float(asr_vi["wer"]), float(aggregate["asr_vi_wer_max"]))
    maximum("asr_en_wer", float(asr_en["wer"]), float(aggregate["asr_en_wer_max"]))
    minimum("mt_sacrebleu", float(mt["sacrebleu"]), float(aggregate["mt_sacrebleu_min"]))
    minimum("mt_chrf2", float(mt["chrf2"]), float(aggregate["mt_chrf2_min"]))
    code_switch = asr_vi.get("slices", {}).get("code_switch", {}).get("True")
    if code_switch is None:
        checks.append({"name": "asr_code_switch_wer", "pass": False, "reason": "missing required slice"})
    else:
        maximum(
            "asr_code_switch_wer",
            float(code_switch["wer"]),
            float(thresholds["slices"]["asr_code_switch_wer_max"]),
        )

    clinical_thresholds = thresholds["clinical_safety"]
    for category in config["evaluation"]["required_slices"]:
        category_report = clinical.get("categories", {}).get(category)
        key = f"{category}_failure_rate_max"
        if key not in clinical_thresholds:
            key = "terminology_failure_rate_max"
        if category_report is None:
            checks.append({"name": f"clinical_{category}", "pass": False, "reason": "missing required slice"})
            continue
        maximum(
            f"clinical_{category}_failure_rate",
            float(category_report["safety_failure_rate"]),
            float(clinical_thresholds[key]),
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
        "policy": "Aggregate quality and every critical clinical slice must pass.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["promotion_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

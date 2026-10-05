"""Canonical bake-off selection policy bindings shared by selection and blind gates."""

from __future__ import annotations

import hashlib
import json
from typing import Any


SELECTION_POLICY_VERSION = 1


def selection_policy_record(config: dict[str, Any]) -> dict[str, Any]:
    """Return a checksum-bound description of the winner-selection contract."""
    gate = config["promotion_gate"]
    principles = config.get("principles") or {}
    policy = {
        "version": SELECTION_POLICY_VERSION,
        "winner_rule": "multi_metric_pareto_95ci_v1",
        "mt_metrics": list(gate["mt_metrics"]),
        "asr_metrics": list(gate["asr_metrics"]),
        "asr_code_switch_wer_max": float(gate["asr_code_switch_wer_max"]),
        "critical_slices": list(gate["critical_slices"]),
        "policy_slices": list(gate["policy_slices"]),
        "candidate_a_auto_promotion_forbidden": bool(
            principles.get("candidate_a_auto_promotion_forbidden")
        ),
    }
    policy["sha256"] = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return policy


def configured_selection_hashes(config: dict[str, Any]) -> dict[str, str]:
    """Return every immutable input hash that must bind a selection comparison."""
    data = config["data"]
    return {
        "mt": str(data["selection_dev"]["mt"]["sha256"]),
        "asr": str(data["selection_dev"]["asr"]["sha256"]),
        "accuracy_program": str(data["accuracy_program"]["sha256"]),
    }

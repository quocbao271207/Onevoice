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


def selection_identity(comparison: dict[str, Any]) -> dict[str, Any]:
    """Return immutable selection, license and winner bindings only."""
    winners: dict[str, dict[str, dict[str, Any]]] = {}
    for task in ("mt", "asr"):
        task_winners = comparison.get("results", {}).get(task, {}).get("winners", {})
        winners[task] = {
            key: {
                "candidate_id": winner.get("candidate_id"),
                "adapter": winner.get("adapter"),
                "adapter_manifest_sha256": winner.get("adapter_manifest_sha256"),
                "direction": winner.get("direction"),
                "profile": winner.get("profile"),
            }
            for key, winner in sorted(task_winners.items())
        }
    return {
        "scope": comparison.get("scope"),
        "candidate_a_freeze": comparison.get("candidate_a_freeze"),
        "candidate_a_locked_evaluations": comparison.get(
            "candidate_a_locked_evaluations"
        ),
        "selection_sha256": comparison.get("selection_sha256"),
        "selection_policy": comparison.get("selection_policy"),
        "research_license_approvals": comparison.get(
            "research_license_approvals"
        ),
        "license_decisions": comparison.get("license_decisions"),
        "winners": winners,
    }


def selection_identity_sha256(comparison: dict[str, Any]) -> str:
    """Hash the canonical immutable selection identity."""
    return hashlib.sha256(
        json.dumps(
            selection_identity(comparison),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

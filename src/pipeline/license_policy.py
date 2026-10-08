"""Canonical model-license decisions shared by bake-off and promotion gates."""

from __future__ import annotations

from typing import Any


def license_gate(
    candidate: dict[str, Any],
    research_approvals: set[str],
) -> tuple[bool, str]:
    """Return whether GPU research use is allowed and the auditable reason."""
    license_info = candidate.get("license") or {}
    if license_info.get("gpu_eligible") and license_info.get("review_status") in {
        "approved",
        "approved_research_only",
    }:
        return True, "approved"
    if (
        candidate["id"] in research_approvals
        and license_info.get("research_reference_allowed")
        and license_info.get("review_status") == "legal_review_required"
    ):
        return True, "explicit_research_approval"
    return False, str(license_info.get("review_status") or "missing_license_review")


def license_decisions(
    config: dict[str, Any],
    research_approvals: set[str],
) -> dict[str, dict[str, Any]]:
    """Build exact license decisions and reject approvals for unknown candidates."""
    candidates = [
        (task, candidate)
        for task in ("mt", "asr")
        for candidate in config["candidates"][task]
    ]
    known_ids = {str(candidate["id"]) for _, candidate in candidates}
    if len(known_ids) != len(candidates):
        raise ValueError("Candidate IDs must be unique for license decisions")
    unknown = sorted(research_approvals - known_ids)
    if unknown:
        raise ValueError(
            "Research license approval names unknown candidates: " + ", ".join(unknown)
        )
    result: dict[str, dict[str, Any]] = {}
    for task, candidate in candidates:
        allowed, reason = license_gate(candidate, research_approvals)
        license_info = candidate.get("license") or {}
        result[str(candidate["id"])] = {
            "task": task,
            "gpu_allowed": allowed,
            "reason": reason,
            "license_id": str(license_info.get("id") or ""),
            "source": str(license_info.get("source") or ""),
            "review_status": str(license_info.get("review_status") or ""),
            "gpu_eligible": license_info.get("gpu_eligible") is True,
            "production_eligible": license_info.get("production_eligible") is True,
            "research_reference_allowed": (
                license_info.get("research_reference_allowed") is True
            ),
        }
    return result

from types import SimpleNamespace

from demo.demo_cli import format_translation_for_display


def test_cli_never_displays_candidate_that_failed_safety_gate():
    result = SimpleNamespace(
        safety_passed=False,
        safety_issues=[
            "quantity_binding_mismatch:aspirin:[('5', 'mg')]->[('50', 'mg')]",
            "negation_mismatch",
        ],
        translated_text="PATIENT_SECRET_UNSAFE_OUTPUT",
    )

    displayed = format_translation_for_display(result)

    assert displayed == (
        "[BLOCKED BY CLINICAL SAFETY GATE: "
        "negation_mismatch, quantity_binding_mismatch]"
    )
    assert "PATIENT_SECRET_UNSAFE_OUTPUT" not in displayed


def test_cli_displays_only_safety_passed_translation():
    result = SimpleNamespace(
        safety_passed=True,
        safety_issues=[],
        translated_text="Bản dịch an toàn",
    )

    assert format_translation_for_display(result) == "Bản dịch an toàn"

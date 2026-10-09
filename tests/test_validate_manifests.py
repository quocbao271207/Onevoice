import json
from pathlib import Path

import pytest

from scripts.validate_manifests import ROLES, main, validate_manifest_dir


def _row(task: str, role: str, suffix: str) -> dict:
    row = {
        "id": f"{task}-{role}-{suffix}",
        "source": "fixture",
        "merge_policy": "official_split",
        "role": role,
        "task": task,
        "quality_flags": [],
    }
    if task == "asr":
        row.update(
            {
                "duration_s": 1.25,
                "group": f"group-{role}-{suffix}",
                "speaker": f"speaker-{role}-{suffix}",
                "text_fingerprint": f"text-{role}-{suffix}",
            }
        )
    else:
        row["pair_fingerprint"] = f"pair-{role}-{suffix}"
    return row


def _write_manifests(root: Path) -> None:
    root.mkdir()
    for task in ("asr", "mt"):
        for role in ROLES:
            payload = json.dumps(_row(task, role, "one"), ensure_ascii=False) + "\n"
            (root / f"{task}--{role}.jsonl").write_text(payload, encoding="utf-8")


def test_validation_binds_strict_manifest_identity(tmp_path: Path):
    manifests = tmp_path / "manifests"
    _write_manifests(manifests)

    report = validate_manifest_dir(manifests)

    assert report["status"] == "pass"
    assert report["errors"] == []
    for task in ("asr", "mt"):
        for role in ROLES:
            evidence = report["tasks"][task]["roles"][role]
            assert evidence["rows"] == 1
            assert evidence["bytes"] > 0
            assert len(evidence["sha256"]) == 64


@pytest.mark.parametrize(
    "payload,match",
    [
        ('{"id":"one","id":"two"}\n', "strict UTF-8 JSONL"),
        ('{"duration_s":1e400}\n', "strict UTF-8 JSONL"),
        ('[]\n', "not an object"),
    ],
)
def test_validation_rejects_ambiguous_or_non_mapping_jsonl(
    tmp_path: Path,
    payload: str,
    match: str,
):
    manifests = tmp_path / "manifests"
    _write_manifests(manifests)
    (manifests / "asr--train.jsonl").write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        validate_manifest_dir(manifests)


def test_validation_reports_schema_and_within_role_duplicate_failures(tmp_path: Path):
    manifests = tmp_path / "manifests"
    _write_manifests(manifests)
    bad = _row("asr", "train", "duplicate")
    bad["id"] = "ASR-TRAIN-DUPLICATE"
    bad["task"] = "mt"
    bad["duration_s"] = -1
    duplicate = dict(bad)
    duplicate["id"] = "asr-train-duplicate"
    (manifests / "asr--train.jsonl").write_text(
        "\n".join(json.dumps(row) for row in (bad, duplicate)) + "\n",
        encoding="utf-8",
    )

    report = validate_manifest_dir(manifests)

    assert report["status"] == "fail"
    assert "asr:train:row=1:task_mismatch" in report["errors"]
    assert "asr:train:row=1:invalid_duration_s" in report["errors"]
    assert "asr:train:row=2:duplicate_id" in report["errors"]


def test_validation_rejects_manifest_symlink(tmp_path: Path):
    manifests = tmp_path / "manifests"
    _write_manifests(manifests)
    original = manifests / "asr--train.jsonl"
    target = tmp_path / "linked.jsonl"
    target.write_bytes(original.read_bytes())
    original.unlink()
    try:
        original.symlink_to(target)
    except OSError:
        pytest.skip("Symlink creation is unavailable")

    with pytest.raises(ValueError, match="cannot be a link"):
        validate_manifest_dir(manifests)


def test_main_publishes_durable_validation_report(tmp_path: Path, monkeypatch):
    manifests = tmp_path / "manifests"
    output = tmp_path / "reports" / "validation.json"
    _write_manifests(manifests)
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_manifests.py",
            "--manifest-dir",
            str(manifests),
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    assert output.read_bytes().endswith(b"\n")
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"


def test_invalid_manifest_does_not_replace_existing_report(tmp_path: Path, monkeypatch):
    manifests = tmp_path / "manifests"
    output = tmp_path / "validation.json"
    _write_manifests(manifests)
    (manifests / "mt--test.jsonl").write_text(
        '{"id":"one","id":"two"}\n', encoding="utf-8"
    )
    output.write_text("sentinel", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_manifests.py",
            "--manifest-dir",
            str(manifests),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(ValueError, match="strict UTF-8 JSONL"):
        main()
    assert output.read_text(encoding="utf-8") == "sentinel"

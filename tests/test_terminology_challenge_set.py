from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
import yaml

from scripts.build_terminology_challenge_set import (
    LoadedSource,
    SourceSpec,
    build_report,
    load_source,
    mine_candidates,
    parse_source_spec,
    select_challenges,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]


def manifest_evidence(task: str) -> dict[str, object]:
    return {
        "task": task,
        "path": f"data/eval/{task}-validation.jsonl",
        "bytes": 1,
        "sha256": hashlib.sha256(task.encode()).hexdigest(),
        "rows": 1,
    }


def loaded(label: str, rows: list[dict]) -> LoadedSource:
    return LoadedSource(
        spec=SourceSpec(label=label, path=Path(f"{label}.jsonl")),
        rows=rows,
        evidence={"label": label},
        provenance=None,
    )


def test_mines_only_verified_errors_and_keeps_rows_review_only():
    manifests = {
        "mt": {
            "mt-1": {
                "id": "mt-1",
                "role": "validation",
                "source_text": "Do not give aspirin 5 mg.",
                "target_text": "Không dùng aspirin 5 mg.",
                "categories": ["drug_name", "dose", "negation", "terminology"],
                "terminology": {
                    "en_to_vi": {"aspirin": ["aspirin"]},
                    "vi_to_en": {"aspirin": ["aspirin"]},
                },
            }
        },
        "asr": {
            "asr-1": {
                "id": "asr-1",
                "role": "validation",
                "text": "không dùng aspirin",
                "audio_path": "audio.flac",
                "audio_sha256": "a" * 64,
                "speaker": "speaker-1",
                "group": "group-1",
            }
        },
    }
    evidence = {"mt": manifest_evidence("mt"), "asr": manifest_evidence("asr")}
    sources = [
        loaded(
            "mt_model_a",
            [
                {
                    "id": "mt-1",
                    "direction": "en_to_vi",
                    "source": "Do not give aspirin 5 mg.",
                    "reference": "Không dùng aspirin 5 mg.",
                    "hypothesis": "Dùng thuốc 10 mg.",
                }
            ],
        ),
        loaded(
            "mt_model_b",
            [
                {
                    "id": "mt-1",
                    "direction": "en_to_vi",
                    "source": "Do not give aspirin 5 mg.",
                    "reference": "Không dùng aspirin 5 mg.",
                    "hypothesis": "Không dùng aspirin 5 mg.",
                }
            ],
        ),
        loaded(
            "asr_model",
            [
                {
                    "id": "asr-1",
                    "reference": "không dùng aspirin",
                    "hypothesis": "dùng aspirin",
                }
            ],
        ),
    ]

    rows = mine_candidates(sources, manifests, evidence)

    assert len(rows) == 2
    mt = next(row for row in rows if row["task"] == "mt")
    assert mt["hazard_level"] == "critical"
    assert {"negation_mismatch", "number_mismatch", "terminology_missing"} <= set(
        mt["hazard_basis"]
    )
    assert [item["model_run"] for item in mt["observations"]] == [
        "mt_model_a",
        "mt_model_b",
    ]
    assert mt["review_status"] == "pending"
    assert mt["evaluation_only"] is True
    assert mt["auto_correction_eligible"] is False
    asr = next(row for row in rows if row["task"] == "asr")
    assert asr["candidate_terms"] == ["không"]
    assert asr["hazard_basis"] == ["asr_word_error"]


def test_manifest_mismatch_fails_closed():
    manifests = {
        "mt": {
            "mt-1": {
                "id": "mt-1",
                "role": "validation",
                "source_text": "Give aspirin.",
                "target_text": "Dùng aspirin.",
            }
        },
        "asr": {},
    }
    source = loaded(
        "bad",
        [
            {
                "id": "mt-1",
                "direction": "en_to_vi",
                "source": "Give aspirin.",
                "reference": "Sai tham chiếu.",
                "hypothesis": "Dùng aspirin.",
            }
        ],
    )
    with pytest.raises(ValueError, match="reference mismatch"):
        mine_candidates(
            [source],
            manifests,
            {"mt": manifest_evidence("mt"), "asr": manifest_evidence("asr")},
        )


def test_archive_source_requires_matching_provenance(tmp_path: Path):
    predictions = (
        json.dumps(
            {
                "id": "mt-1",
                "direction": "en_to_vi",
                "source": "Give aspirin.",
                "reference": "Dùng aspirin.",
                "hypothesis": "Dùng thuốc.",
            }
        )
        + "\n"
    ).encode()
    provenance = {
        "schema_version": 1,
        "specification": {
            "task": "mt",
            "manifest": {"sha256": "a" * 64},
        },
        "predictions": {
            "bytes": len(predictions),
            "sha256": hashlib.sha256(predictions).hexdigest(),
            "rows": 1,
        },
    }
    archive_path = tmp_path / "predictions.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, data in (
            ("run/predictions.jsonl", predictions),
            (
                "run/predictions.jsonl.provenance.json",
                json.dumps(provenance).encode(),
            ),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    spec = parse_source_spec(f"model={archive_path}::run/predictions.jsonl")
    source = load_source(spec)

    assert source.rows[0]["id"] == "mt-1"
    assert source.evidence["member_sha256"] == hashlib.sha256(predictions).hexdigest()
    with pytest.raises(ValueError, match="Unsafe archive member"):
        parse_source_spec(f"model={archive_path}::../predictions.jsonl")


def test_selection_enforces_100_to_200_and_is_deterministic():
    candidates = []
    for index in range(205):
        candidates.append(
            {
                "challenge_id": f"challenge-{index:03d}",
                "task": "mt" if index % 2 else "asr",
                "direction": "en_to_vi" if index % 2 else None,
                "hazard_basis": ["number_mismatch" if index % 3 else "asr_word_error"],
                "observations": [
                    {
                        "safety_issues": ["number_mismatch"] if index % 2 else [],
                        "error_detected": True,
                        "word_error_rate": 0.5 if index % 2 == 0 else None,
                    }
                ],
            }
        )

    first = select_challenges(candidates, 128)
    second = select_challenges(list(reversed(candidates)), 128)

    assert [row["challenge_id"] for row in first] == [
        row["challenge_id"] for row in second
    ]
    with pytest.raises(ValueError, match="between 100 and 200"):
        select_challenges(candidates, 99)


def test_report_locks_artifact_and_pending_policy(tmp_path: Path):
    output = tmp_path / "challenge.jsonl"
    rows = [
        {
            "task": "asr",
            "direction": None,
            "hazard_level": "moderate",
            "review_status": "pending",
            "auto_correction_eligible": False,
        }
    ]
    output_data = write_jsonl(output, rows)
    report = build_report(
        output,
        output_data,
        rows,
        [loaded("asr_model", [])],
        {"asr": manifest_evidence("asr")},
    )

    assert report["status"] == "awaiting_human_review"
    assert report["artifact"]["sha256"] == hashlib.sha256(output_data).hexdigest()
    assert report["policy"]["auto_correction_eligible"] is False


def test_committed_challenge_set_is_validation_only_hash_locked_and_pending():
    challenge_path = ROOT / "data/eval/terminology_challenge_set.jsonl"
    report_path = ROOT / "data/eval/terminology_challenge_set.report.json"
    source_list_path = ROOT / "configs/terminology_challenge_sources.txt"
    lock = yaml.safe_load((ROOT / "configs/artifact_lock.yaml").read_text(encoding="utf-8"))[
        "evaluation"
    ]
    rows = [
        json.loads(line)
        for line in challenge_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert 100 <= len(rows) <= 200
    assert len(rows) == report["artifact"]["rows"] == 128
    assert len({row["challenge_id"] for row in rows}) == len(rows)
    assert {row["task"] for row in rows} == {"asr", "mt"}
    assert {row["direction"] for row in rows if row["task"] == "mt"} == {
        "en_to_vi",
        "vi_to_en",
    }
    assert all(row["review_status"] == "pending" for row in rows)
    assert all(row["evaluation_only"] is True for row in rows)
    assert all(row["auto_correction_eligible"] is False for row in rows)
    assert all(any(item["error_detected"] for item in row["observations"]) for row in rows)
    assert all(
        any(item["safety_issues"] for item in row["observations"])
        for row in rows
        if row["task"] == "mt"
    )
    assert all(
        any(float(item["word_error_rate"] or 0.0) > 0.0 for item in row["observations"])
        for row in rows
        if row["task"] == "asr"
    )
    assert {
        row["provenance"]["manifest_path"] for row in rows
    } == {
        "data/eval/mt_selection_dev.jsonl",
        "data/processed/manifests/asr--validation-local.jsonl",
    }

    challenge_sha = hashlib.sha256(challenge_path.read_bytes()).hexdigest()
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    source_list_sha = hashlib.sha256(source_list_path.read_bytes()).hexdigest()
    assert b"\r\n" not in challenge_path.read_bytes()
    assert b"\r\n" not in report_path.read_bytes()
    assert b"\r\n" not in source_list_path.read_bytes()
    assert challenge_sha == report["artifact"]["sha256"]
    assert challenge_sha == lock["terminology_challenge_set_sha256"]
    assert report_sha == lock["terminology_challenge_report_sha256"]
    assert source_list_sha == report["source_list"]["sha256"]
    assert source_list_sha == lock["terminology_challenge_sources_sha256"]
    assert report["status"] == lock["terminology_challenge_status"]

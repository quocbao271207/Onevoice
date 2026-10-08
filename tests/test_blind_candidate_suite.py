from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import scripts.run_blind_candidate_suite as blind
from scripts.run_blind_candidate_suite import (
    _mutex_path,
    adapter_tree_manifest,
    blind_report_coverage_failures,
    blind_quality_failures,
    coverage_counts,
    ensure_unseen,
    evaluate,
    exclusive_mutex,
    load_locked_accuracy_config,
    read_jsonl,
    sha256,
    validate_content_integrity,
    validate_identifiers,
    validate_minimum_coverage,
    verify_lock,
    verify_report_provenance,
    verify_selection_winner,
)
from src.data.quality import fingerprint_text
from src.pipeline.license_policy import license_decisions
from src.pipeline.selection_policy import (
    configured_selection_hashes,
    selection_policy_record,
)


ROOT = Path(__file__).resolve().parents[1]


def full_bakeoff_config() -> dict:
    path = ROOT / "configs/model_bakeoff.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_lock(path: Path, mt: Path, asr: Path) -> None:
    mt_coverage = coverage_counts("mt", read_jsonl_fixture(mt))
    asr_coverage = coverage_counts("asr", read_jsonl_fixture(asr))
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "status": "locked_unopened",
                "required_slices": {"mt": [], "asr": []},
                "minimum_coverage": {
                    "mt": {"rows": 1, "slice_samples": {}},
                    "asr": {"rows": 1, "slice_samples": {}},
                },
                "manifests": {
                    "mt": {
                        "path": str(mt),
                        "rows": mt_coverage["rows"],
                        "sha256": sha256(mt),
                        "coverage": mt_coverage,
                    },
                    "asr": {
                        "path": str(asr),
                        "rows": asr_coverage["rows"],
                        "sha256": sha256(asr),
                        "coverage": asr_coverage,
                    },
                },
                "selection": None,
                "opened": {
                    "mt": {"en_to_vi": None, "vi_to_en": None},
                    "asr": None,
                },
            }
        ),
        encoding="utf-8",
    )


def read_jsonl_fixture(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_blind_lock_verifies_checksums_and_detects_tampering(tmp_path: Path):
    mt = tmp_path / "mt.jsonl"
    asr = tmp_path / "asr.jsonl"
    mt.write_text('{"id":"mt"}\n', encoding="utf-8")
    asr.write_text('{"id":"asr"}\n', encoding="utf-8")
    lock = tmp_path / "lock.json"
    write_lock(lock, mt, asr)
    assert verify_lock(lock)["status"] == "locked_unopened"

    locked = json.loads(lock.read_text(encoding="utf-8"))
    locked["manifests"]["mt"]["coverage"]["rows"] = 2
    lock.write_text(json.dumps(locked), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage record mismatch"):
        verify_lock(lock)

    write_lock(lock, mt, asr)
    mt.write_text('{"id":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_lock(lock)


def test_blind_lock_rejects_linked_manifest(tmp_path: Path):
    mt = tmp_path / "mt.jsonl"
    asr = tmp_path / "asr.jsonl"
    mt.write_text('{"id":"mt"}\n', encoding="utf-8")
    asr.write_text('{"id":"asr"}\n', encoding="utf-8")
    linked_mt = tmp_path / "mt-link.jsonl"
    try:
        linked_mt.symlink_to(mt.name)
    except OSError as exc:
        pytest.skip(f"Symlink creation is unavailable: {exc}")
    lock = tmp_path / "lock.json"
    write_lock(lock, mt, asr)
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["manifests"]["mt"]["path"] = str(linked_mt)
    lock.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="symlink or junction"):
        verify_lock(lock)


def test_blind_jsonl_is_bounded_and_object_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest = tmp_path / "blind.jsonl"
    manifest.write_text('["not-an-object"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        read_jsonl(manifest)

    monkeypatch.setattr(blind, "MAX_BLIND_MANIFEST_LINE_BYTES", 8)
    manifest.write_text('{"id":"too-long"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 exceeds 8 bytes"):
        read_jsonl(manifest)


def test_blind_lock_rejects_legacy_schema(tmp_path: Path):
    mt = tmp_path / "mt.jsonl"
    asr = tmp_path / "asr.jsonl"
    mt.write_text('{"id":"mt"}\n', encoding="utf-8")
    asr.write_text('{"id":"asr"}\n', encoding="utf-8")
    lock = tmp_path / "lock.json"
    write_lock(lock, mt, asr)
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["version"] = 1
    lock.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema version 2"):
        verify_lock(lock)


def test_blind_lock_rejects_unknown_schema_fields(tmp_path: Path):
    mt = tmp_path / "mt.jsonl"
    asr = tmp_path / "asr.jsonl"
    mt.write_text('{"id":"mt"}\n', encoding="utf-8")
    asr.write_text('{"id":"asr"}\n', encoding="utf-8")
    lock = tmp_path / "lock.json"
    write_lock(lock, mt, asr)
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["trusted_override"] = True
    lock.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="lock schema is invalid"):
        verify_lock(lock)


def test_blind_lock_status_must_match_opened_slots(tmp_path: Path):
    mt = tmp_path / "mt.jsonl"
    asr = tmp_path / "asr.jsonl"
    mt.write_text('{"id":"mt"}\n', encoding="utf-8")
    asr.write_text('{"id":"asr"}\n', encoding="utf-8")
    lock = tmp_path / "lock.json"
    write_lock(lock, mt, asr)
    payload = json.loads(lock.read_text(encoding="utf-8"))
    payload["status"] = "partially_opened"
    lock.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="status does not match opened slots"):
        verify_lock(lock)


def test_blind_slot_mutex_rejects_concurrent_duplicate_open(tmp_path: Path):
    lock_path = tmp_path / "blind.lock.json"
    mutex = _mutex_path(lock_path, "slot-mt-en_to_vi")

    with exclusive_mutex(mutex):
        with pytest.raises(RuntimeError, match="already active"):
            with exclusive_mutex(mutex):
                pytest.fail("duplicate blind slot mutex was acquired")


def test_placeholder_blind_lock_cannot_be_opened():
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="not locked"):
        verify_lock(root / "data/eval/blind_test_v2.lock.json")


def test_blind_winner_is_bound_to_selection_and_exact_adapter_tree(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"winner-weights")
    manifest = adapter_tree_manifest(adapter)
    comparison = tmp_path / "comparison.json"
    config = full_bakeoff_config()
    candidate_id = "mt_m2m100_418m"
    approvals: set[str] = set()
    comparison.write_text(
        json.dumps(
            {
                "status": "selection_complete",
                "scope": "research",
                "selection_policy": selection_policy_record(config),
                "selection_sha256": configured_selection_hashes(config),
                "research_license_approvals": sorted(approvals),
                "license_decisions": license_decisions(config, approvals),
                "results": {
                    "mt": {
                        "winners": {
                            "en_to_vi": {
                                "candidate_id": candidate_id,
                                "adapter": str(adapter),
                                "adapter_manifest_sha256": manifest["manifest_sha256"],
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    selected = verify_selection_winner(
        comparison,
        "mt",
        "en_to_vi",
        candidate_id,
        adapter,
        config,
        "research",
    )

    assert selected["sha256"] == sha256(comparison)
    assert selected["adapter_manifest"]["manifest_sha256"] == manifest["manifest_sha256"]
    valid_comparison = comparison.read_text(encoding="utf-8")
    vinai_id = "mt_vinai_en2vi_v2"
    vinai = json.loads(valid_comparison)
    vinai["results"]["mt"]["winners"]["en_to_vi"]["candidate_id"] = vinai_id
    comparison.write_text(json.dumps(vinai), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks the license approval"):
        verify_selection_winner(
            comparison,
            "mt",
            "en_to_vi",
            vinai_id,
            adapter,
            config,
            "research",
        )
    vinai["research_license_approvals"] = [vinai_id]
    vinai["license_decisions"] = license_decisions(config, {vinai_id})
    comparison.write_text(json.dumps(vinai), encoding="utf-8")
    assert verify_selection_winner(
        comparison,
        "mt",
        "en_to_vi",
        vinai_id,
        adapter,
        config,
        "research",
    )["winner"]["candidate_id"] == vinai_id
    comparison.write_text(valid_comparison, encoding="utf-8")
    stale = json.loads(valid_comparison)
    stale["selection_sha256"]["mt"] = "0" * 64
    comparison.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="selection inputs"):
        verify_selection_winner(
            comparison,
            "mt",
            "en_to_vi",
            candidate_id,
            adapter,
            config,
            "research",
        )
    comparison.write_text(valid_comparison, encoding="utf-8")
    with pytest.raises(ValueError, match="not the selected winner"):
        verify_selection_winner(
            comparison,
            "mt",
            "en_to_vi",
            "other",
            adapter,
            config,
            "research",
        )

    weights.write_bytes(b"changed-after-selection")
    with pytest.raises(ValueError, match="adapter checksum"):
        verify_selection_winner(
            comparison,
            "mt",
            "en_to_vi",
            candidate_id,
            adapter,
            config,
            "research",
        )


def test_blind_rejects_legacy_selection_without_bound_policy(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    comparison = tmp_path / "legacy-comparison.json"
    comparison.write_text(
        json.dumps({"status": "selection_complete", "results": {}}),
        encoding="utf-8",
    )
    config = full_bakeoff_config()

    with pytest.raises(ValueError, match="selection policy"):
        verify_selection_winner(
            comparison,
            "mt",
            "en_to_vi",
            "legacy",
            adapter,
            config,
            "research",
        )


def test_blind_report_provenance_rejects_tampering(tmp_path: Path):
    report = tmp_path / "blind.json"
    report.write_text('{"wer":0.1}', encoding="utf-8")
    provenance = tmp_path / "blind_provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "specification_sha256": "a" * 64,
                "blind_manifest_sha256": "b" * 64,
                "selection_sha256": "c" * 64,
                "report_sha256": sha256(report),
                "report_bytes": report.stat().st_size,
            }
        ),
        encoding="utf-8",
    )

    assert verify_report_provenance(
        report, provenance, "a" * 64, "b" * 64, "c" * 64
    )["report_sha256"] == sha256(report)

    report.write_text('{"wer":0.9}', encoding="utf-8")
    with pytest.raises(ValueError, match="provenance mismatch"):
        verify_report_provenance(
            report, provenance, "a" * 64, "b" * 64, "c" * 64
        )


def test_blind_evaluate_resumes_only_verified_selected_report(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"selected-weights")
    adapter_manifest = adapter_tree_manifest(adapter)

    mt_manifest = tmp_path / "blind-mt.jsonl"
    asr_manifest = tmp_path / "blind-asr.jsonl"
    mt_manifest.write_text('{"id":"mt"}\n', encoding="utf-8")
    asr_manifest.write_text('{"id":"asr"}\n', encoding="utf-8")
    comparison = tmp_path / "selection.json"
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "version": 2,
                "status": "locked_unopened",
                "required_slices": {"mt": [], "asr": []},
                "minimum_coverage": {
                    "mt": {"rows": 1, "slice_samples": {}},
                    "asr": {"rows": 1, "slice_samples": {}},
                },
                "manifests": {
                    "mt": {
                        "path": str(mt_manifest),
                        "rows": 1,
                        "sha256": sha256(mt_manifest),
                        "coverage": {"rows": 1, "slice_samples": {}},
                    },
                    "asr": {
                        "path": str(asr_manifest),
                        "rows": 1,
                        "sha256": sha256(asr_manifest),
                        "coverage": {
                            "rows": 1,
                            "slice_samples": {},
                            "unique_speakers": 0,
                            "unique_groups": 0,
                        },
                    },
                },
                "selection": None,
                "opened": {"mt": {"en_to_vi": None, "vi_to_en": None}, "asr": None},
            }
        ),
        encoding="utf-8",
    )
    accuracy = tmp_path / "accuracy.yaml"
    accuracy.write_text(
        "release_gates:\n  aggregate:\n    mt_sacrebleu_min: 25\n    mt_chrf2_min: 46\n",
        encoding="utf-8",
    )
    config = {
        "principles": {"candidate_a_auto_promotion_forbidden": True},
        "candidates": {
            "mt": [
                {
                    "id": "mt-winner",
                    "model": "base-model",
                    "revision": "r" * 40,
                    "model_family": "nllb",
                    "license": {
                        "id": "test-license",
                        "review_status": "approved",
                        "production_eligible": True,
                        "gpu_eligible": True,
                    },
                }
            ],
            "asr": [],
        },
        "promotion_gate": {
            "critical_slices": [],
            "policy_slices": [],
            "mt_metrics": ["sacrebleu", "chrf2"],
            "asr_metrics": ["wer", "cer", "code_switch_wer"],
            "asr_code_switch_wer_max": 0.21,
        },
        "data": {
            "accuracy_program": {"path": str(accuracy), "sha256": sha256(accuracy)},
            "selection_dev": {
                "mt": {"sha256": "a" * 64},
                "asr": {"sha256": "b" * 64},
            },
        },
    }
    comparison.write_text(
        json.dumps(
            {
                "status": "selection_complete",
                "scope": "research",
                "selection_policy": selection_policy_record(config),
                "selection_sha256": configured_selection_hashes(config),
                "research_license_approvals": [],
                "license_decisions": license_decisions(config, set()),
                "results": {
                    "mt": {
                        "winners": {
                            "en_to_vi": {
                                "candidate_id": "mt-winner",
                                "adapter": str(adapter),
                                "adapter_manifest_sha256": adapter_manifest[
                                    "manifest_sha256"
                                ],
                            }
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()
    stem = "blind_v2_mt_en_to_vi_mt-winner"
    report = output / f"{stem}.json"
    report.write_text(
        json.dumps(
            {
                "directions": {
                    "en_to_vi": {
                        "samples": 1,
                        "sacrebleu": 30.0,
                        "sacrebleu_bootstrap_95ci": [29.0, 31.0],
                        "chrf2": 50.0,
                        "chrf2_bootstrap_95ci": [49.0, 51.0],
                    }
                },
                "categories": {},
            }
        ),
        encoding="utf-8",
    )
    specification = {
        "candidate": "mt-winner",
        "model": "base-model",
        "revision": "r" * 40,
        "adapter": str(adapter.resolve()),
        "adapter_manifest_sha256": adapter_manifest["manifest_sha256"],
        "selection_sha256": sha256(comparison),
        "blind_manifest_sha256": sha256(mt_manifest),
        "direction": "en_to_vi",
        "scope": "research",
    }
    specification_sha = hashlib.sha256(
        json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    provenance = output / f"{stem}_provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "specification_sha256": specification_sha,
                "blind_manifest_sha256": sha256(mt_manifest),
                "selection_sha256": sha256(comparison),
                "report_sha256": sha256(report),
                "report_bytes": report.stat().st_size,
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        task="mt",
        direction="en_to_vi",
        candidate="mt-winner",
        adapter=adapter,
        selection_comparison=comparison,
        scope="research",
        output_dir=output,
        python="unused",
    )

    result = evaluate(args, lock_path, config)

    locked = json.loads(lock_path.read_text(encoding="utf-8"))
    assert result["promotion_allowed"] is True
    assert result["coverage_gate"] == "pass"
    assert result["report_sha256"] == sha256(report)
    assert locked["selection"]["sha256"] == sha256(comparison)
    assert locked["opened"]["mt"]["en_to_vi"]["candidate_sha256"] == specification_sha


@pytest.mark.parametrize(
    ("task", "rows", "message"),
    [
        ("mt", [{"pair_fingerprint": "pair"}], "missing id"),
        ("mt", [{"id": "mt-1"}], "missing pair_fingerprint"),
        (
            "asr",
            [{"id": "asr-1", "audio_sha256": "audio"}],
            "missing text_fingerprint",
        ),
        (
            "asr",
            [{"id": "asr-1", "text_fingerprint": "text-1"}],
            "missing audio_sha256",
        ),
        (
            "asr",
            [
                {
                    "id": "asr-1",
                    "text_fingerprint": "text-1",
                    "audio_sha256": "audio",
                    "group": "group-1",
                }
            ],
            "missing speaker",
        ),
        (
            "asr",
            [
                {
                    "id": "asr-1",
                    "text_fingerprint": "text-1",
                    "audio_sha256": "audio",
                    "speaker": "speaker-1",
                }
            ],
            "missing group",
        ),
    ],
)
def test_blind_identifiers_require_leakage_fingerprints(task, rows, message):
    with pytest.raises(ValueError, match=message):
        validate_identifiers(task, rows)


@pytest.mark.parametrize(
    ("task", "rows", "message"),
    [
        (
            "mt",
            [
                {"id": "mt", "pair_fingerprint": "pair-1"},
                {"id": "mt", "pair_fingerprint": "pair-2"},
            ],
            "duplicate id",
        ),
        (
            "mt",
            [
                {"id": "mt-1", "pair_fingerprint": "pair"},
                {"id": "mt-2", "pair_fingerprint": "pair"},
            ],
            "duplicate pair_fingerprint",
        ),
        (
            "asr",
            [
                {
                    "id": "asr-1",
                    "text_fingerprint": "text-1",
                    "audio_sha256": "audio",
                    "speaker": "speaker-1",
                    "group": "group-1",
                },
                {
                    "id": "asr-2",
                    "text_fingerprint": "text-2",
                    "audio_sha256": "audio",
                    "speaker": "speaker-2",
                    "group": "group-2",
                },
            ],
            "duplicate audio_sha256",
        ),
        (
            "asr",
            [
                {
                    "id": "asr",
                    "text_fingerprint": "text-1",
                    "audio_sha256": "audio-1",
                    "speaker": "speaker-1",
                    "group": "group-1",
                },
                {
                    "id": "asr",
                    "text_fingerprint": "text-2",
                    "audio_sha256": "audio-2",
                    "speaker": "speaker-2",
                    "group": "group-2",
                },
            ],
            "duplicate id",
        ),
    ],
)
def test_blind_identifiers_reject_duplicate_samples(task, rows, message):
    with pytest.raises(ValueError, match=message):
        validate_identifiers(task, rows)


def test_blind_mt_fingerprint_is_recomputed_from_text():
    source = "Take aspirin 81 mg daily."
    target = "Uống aspirin 81 mg mỗi ngày."
    row = {
        "id": "mt-1",
        "source_language": "en",
        "target_language": "vi",
        "source_text": source,
        "target_text": target,
        "pair_fingerprint": fingerprint_text(source + "\x1f" + target),
    }
    validate_content_integrity("mt", [row])

    row["target_text"] = "Uống aspirin 810 mg mỗi ngày."
    with pytest.raises(ValueError, match="pair_fingerprint does not match"):
        validate_content_integrity("mt", [row])


@pytest.mark.parametrize("missing", ["source_text", "target_text"])
def test_blind_mt_requires_both_text_sides(missing):
    row = {
        "source_language": "en",
        "target_language": "vi",
        "source_text": "No penicillin.",
        "target_text": "Không dùng penicillin.",
        "pair_fingerprint": "declared",
    }
    row.pop(missing)
    with pytest.raises(ValueError, match="missing source_text or target_text"):
        validate_content_integrity("mt", [row])


def test_blind_asr_fingerprints_are_recomputed_from_text_and_audio(tmp_path: Path):
    audio = tmp_path / "blind.flac"
    audio.write_bytes(b"blind-audio-payload")
    text = "Không dùng penicillin 500 mg."
    row = {
        "id": "asr-1",
        "language": "vi",
        "text": text,
        "text_fingerprint": fingerprint_text(text),
        "audio_path": str(audio),
        "audio_sha256": sha256(audio),
    }
    validate_content_integrity("asr", [row])

    row["text"] = "Dùng penicillin 500 mg."
    with pytest.raises(ValueError, match="text_fingerprint does not match"):
        validate_content_integrity("asr", [row])

    row["text"] = text
    audio.write_bytes(b"tampered-audio-payload")
    with pytest.raises(ValueError, match="audio_sha256 does not match"):
        validate_content_integrity("asr", [row])


def test_blind_asr_requires_existing_audio(tmp_path: Path):
    text = "Không dùng penicillin."
    row = {
        "language": "vi",
        "text": text,
        "text_fingerprint": fingerprint_text(text),
        "audio_path": str(tmp_path / "missing.flac"),
        "audio_sha256": "declared",
    }
    with pytest.raises(FileNotFoundError, match="audio is missing"):
        validate_content_integrity("asr", [row])


def test_blind_content_requires_canonical_task_languages(tmp_path: Path):
    mt = {
        "source_language": "vi",
        "target_language": "en",
        "source_text": "Không dùng aspirin.",
        "target_text": "Do not use aspirin.",
        "pair_fingerprint": "declared",
    }
    with pytest.raises(ValueError, match="canonical source_language=en"):
        validate_content_integrity("mt", [mt])

    audio = tmp_path / "english.flac"
    audio.write_bytes(b"audio")
    asr = {
        "language": "en",
        "text": "Do not use aspirin.",
        "text_fingerprint": fingerprint_text("Do not use aspirin."),
        "audio_path": str(audio),
        "audio_sha256": sha256(audio),
    }
    with pytest.raises(ValueError, match="Vietnamese language"):
        validate_content_integrity("asr", [asr])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _unseen_config(task: str, train: Path, selection: Path) -> dict:
    other = "asr" if task == "mt" else "mt"
    return {
        "data": {
            "train": {task: str(train), other: str(train)},
            "selection_dev": {
                task: {"path": str(selection)},
                other: {"path": str(selection)},
            },
            "forbidden_selection_inputs": [],
        }
    }


def test_blind_mt_leakage_is_recomputed_when_historical_fingerprint_is_missing(
    tmp_path: Path,
):
    source = "Take aspirin 81 mg daily."
    target = "Uống aspirin 81 mg mỗi ngày."
    train = tmp_path / "mt-train.jsonl"
    selection = tmp_path / "mt-selection.jsonl"
    _write_jsonl(
        train,
        [{"id": "historical", "source_text": source, "target_text": target}],
    )
    _write_jsonl(
        selection,
        [{"id": "other", "source_text": "No fever.", "target_text": "Không sốt."}],
    )
    blind = [
        {
            "id": "new-id",
            "source_text": source,
            "target_text": target,
            "pair_fingerprint": fingerprint_text(source + "\x1f" + target),
        }
    ]

    with pytest.raises(ValueError, match="leakage against"):
        ensure_unseen("mt", blind, _unseen_config("mt", train, selection))


def test_blind_asr_leakage_hashes_audio_when_historical_sha_is_missing(
    tmp_path: Path,
):
    historical_audio = tmp_path / "historical.flac"
    blind_audio = tmp_path / "blind.flac"
    historical_audio.write_bytes(b"same-audio-content")
    blind_audio.write_bytes(b"same-audio-content")
    train = tmp_path / "asr-train.jsonl"
    selection = tmp_path / "asr-selection.jsonl"
    _write_jsonl(
        train,
        [
            {
                "id": "historical",
                "text": "Một câu lịch sử.",
                "audio_path": str(historical_audio),
                "speaker": "historical-speaker",
                "group": "historical-group",
            }
        ],
    )
    other_audio = tmp_path / "other.flac"
    other_audio.write_bytes(b"other-audio-content")
    _write_jsonl(
        selection,
        [
            {
                "id": "other",
                "text": "Một câu khác.",
                "audio_path": str(other_audio),
                "speaker": "other-speaker",
                "group": "other-group",
            }
        ],
    )
    blind = [
        {
            "id": "new-id",
            "text": "Nội dung hoàn toàn mới.",
            "audio_path": str(blind_audio),
            "audio_sha256": sha256(blind_audio),
            "speaker": "blind-speaker",
            "group": "blind-group",
        }
    ]

    with pytest.raises(ValueError, match="leakage against"):
        ensure_unseen("asr", blind, _unseen_config("asr", train, selection))


@pytest.mark.parametrize(("key", "value"), [("speaker", "speaker-1"), ("group", "group-1")])
def test_blind_asr_rejects_historical_speaker_or_group_reuse(
    tmp_path: Path, key: str, value: str
):
    train = tmp_path / "asr-train.jsonl"
    selection = tmp_path / "asr-selection.jsonl"
    _write_jsonl(
        train,
        [
            {
                "id": "train",
                "text": "Câu train.",
                "audio_sha256": "a" * 64,
                "speaker": value if key == "speaker" else "train-speaker",
                "group": value if key == "group" else "train-group",
            }
        ],
    )
    _write_jsonl(
        selection,
        [
            {
                "id": "selection",
                "text": "Câu selection.",
                "audio_sha256": "b" * 64,
                "speaker": "selection-speaker",
                "group": "selection-group",
            }
        ],
    )
    blind = [
        {
            "id": "blind",
            "text": "Câu blind mới.",
            "audio_sha256": "c" * 64,
            "speaker": value if key == "speaker" else "blind-speaker",
            "group": value if key == "group" else "blind-group",
        }
    ]

    with pytest.raises(ValueError, match="leakage against"):
        ensure_unseen("asr", blind, _unseen_config("asr", train, selection))


def test_blind_coverage_counts_unique_slices_per_row():
    mt_rows = [
        {
            "source_text": "Take aspirin 81 mg.",
            "target_text": "Uống aspirin 81 mg.",
            "categories": ["drug_name", "drug_name", "dose"],
        },
        {
            "source_text": "No insulin.",
            "target_text": "Không dùng insulin.",
            "categories": ["drug_name", "negation"],
        },
    ]
    assert coverage_counts("mt", mt_rows) == {
        "rows": 2,
        "slice_samples": {
            "dose": 1,
            "drug_name": 2,
            "en_to_vi": 2,
            "negation": 1,
            "vi_to_en": 2,
        },
    }

    asr_rows = [
        {
            "categories": ["code_switch"],
            "blind_dimensions": {
                "accent_region": "Central",
                "role": "Doctor",
                "noise": True,
            },
        },
        {
            "categories": ["code_switch"],
            "selection_dimensions": {
                "accent_region": "central",
                "role": "patient",
                "noise": False,
            },
        },
    ]
    assert coverage_counts("asr", asr_rows) == {
        "rows": 2,
        "slice_samples": {
            "central": 2,
            "code_switch": 2,
            "doctor": 1,
            "noise": 1,
            "patient": 1,
        },
        "unique_speakers": 0,
        "unique_groups": 0,
    }


def test_blind_minimum_coverage_is_a_hard_gate():
    rows = [
        {
            "source_text": "Take aspirin 81 mg.",
            "target_text": "Uống aspirin 81 mg.",
            "categories": ["drug_name", "dose"],
        },
        {
            "source_text": "No insulin.",
            "target_text": "Không dùng insulin.",
            "categories": ["drug_name", "negation"],
        },
    ]
    required = ["drug_name", "dose", "negation", "en_to_vi", "vi_to_en"]
    policy = {
        "rows": 2,
        "slice_samples": {
            "drug_name": 2,
            "dose": 1,
            "negation": 1,
            "en_to_vi": 2,
            "vi_to_en": 2,
        },
    }
    assert validate_minimum_coverage("mt", rows, policy, required)["rows"] == 2

    policy["slice_samples"]["dose"] = 2
    with pytest.raises(ValueError, match=r"dose=1/2"):
        validate_minimum_coverage("mt", rows, policy, required)

    del policy["slice_samples"]["negation"]
    with pytest.raises(ValueError, match="lacks required slice quotas"):
        validate_minimum_coverage("mt", rows, policy, required)


def test_blind_asr_requires_independent_speaker_and_group_coverage():
    rows = [
        {
            "speaker": f"speaker-{index % 2}",
            "group": f"group-{index % 3}",
            "categories": ["code_switch"],
        }
        for index in range(6)
    ]
    policy = {
        "rows": 6,
        "unique_speakers": 3,
        "unique_groups": 3,
        "slice_samples": {"code_switch": 6},
    }

    with pytest.raises(ValueError, match=r"unique_speakers=2/3"):
        validate_minimum_coverage("asr", rows, policy, ["code_switch"])

    policy["unique_speakers"] = 2
    coverage = validate_minimum_coverage("asr", rows, policy, ["code_switch"])
    assert coverage["unique_speakers"] == 2
    assert coverage["unique_groups"] == 3


def test_blind_report_coverage_must_match_locked_mt_manifest():
    report = {
        "directions": {"en_to_vi": {"samples": 2}},
        "categories": {
            "drug_name": {"samples": 2},
            "dose": {"samples": 1},
        },
    }
    expected = {
        "drug_name": 2,
        "dose": 1,
        "en_to_vi": 2,
        "vi_to_en": 2,
    }
    assert blind_report_coverage_failures(
        report, "mt", "en_to_vi", 2, expected
    ) == []

    report["categories"]["dose"]["samples"] = 0
    assert blind_report_coverage_failures(
        report, "mt", "en_to_vi", 2, expected
    ) == ["slice:dose:0/1"]


def test_blind_report_coverage_must_match_locked_asr_dimensions():
    report = {
        "samples": 2,
        "categories": {"code_switch": {"samples": 1}},
        "slices": {
            "accent": {"Central": {"samples": 2}},
            "role": {
                "Doctor": {"samples": 1},
                "patient": {"samples": 1},
            },
            "noise": {"True": {"samples": 1}, "False": {"samples": 1}},
        },
    }
    expected = {
        "code_switch": 1,
        "central": 2,
        "doctor": 1,
        "patient": 1,
        "noise": 1,
    }
    report.update(
        {
            "unique_speakers": 2,
            "unique_groups": 2,
            "wer_bootstrap_unit": "group",
            "wer_bootstrap_clusters": 2,
        }
    )
    expected_unique = {"unique_speakers": 2, "unique_groups": 2}
    assert blind_report_coverage_failures(
        report, "asr", None, 2, expected, expected_unique
    ) == []

    del report["slices"]["accent"]["Central"]
    report["wer_bootstrap_unit"] = "row"
    report["wer_bootstrap_clusters"] = 1
    assert blind_report_coverage_failures(
        report, "asr", None, 2, expected, expected_unique
    ) == [
        "wer_bootstrap_unit:not_group",
        "wer_bootstrap_clusters:1/2",
        "slice:central:None/2",
    ]


def accuracy_config():
    return {
        "release_gates": {
            "aggregate": {
                "asr_vi_wer_max": 0.19,
                "mt_sacrebleu_min": 25.0,
                "mt_chrf2_min": 46.0,
            }
        }
    }


def bakeoff_config():
    return {"promotion_gate": {"asr_code_switch_wer_max": 0.21}}


def test_blind_mt_quality_requires_lower_ci_bounds_above_policy():
    report = {
        "directions": {
            "en_to_vi": {
                "samples": 100,
                "sacrebleu": 30.0,
                "sacrebleu_bootstrap_95ci": [26.0, 34.0],
                "chrf2": 50.0,
                "chrf2_bootstrap_95ci": [47.0, 53.0],
            }
        }
    }
    assert blind_quality_failures(
        report, "mt", "en_to_vi", accuracy_config(), bakeoff_config()
    ) == []

    report["directions"]["en_to_vi"]["chrf2_bootstrap_95ci"] = [45.0, 53.0]
    assert blind_quality_failures(
        report, "mt", "en_to_vi", accuracy_config(), bakeoff_config()
    ) == ["chrf2_bootstrap_95ci:lower_bound_below_policy"]


def test_blind_asr_quality_requires_upper_ci_and_code_switch_policy():
    report = {
        "samples": 100,
        "wer": 0.15,
        "cer": 0.10,
        "wer_bootstrap_95ci": [0.13, 0.18],
        "slices": {"code_switch": {"True": {"wer": 0.20}}},
    }
    assert blind_quality_failures(
        report, "asr", None, accuracy_config(), bakeoff_config()
    ) == []

    report["wer_bootstrap_95ci"] = [0.13, 0.20]
    report["slices"]["code_switch"]["True"]["wer"] = float("nan")
    failures = blind_quality_failures(
        report, "asr", None, accuracy_config(), bakeoff_config()
    )
    assert failures == [
        "wer_bootstrap_95ci:upper_bound_above_policy",
        "code_switch_wer:missing_or_invalid",
    ]


def test_accuracy_policy_is_checksum_locked(tmp_path: Path):
    path = tmp_path / "accuracy.yaml"
    path.write_text("version: 1\n", encoding="utf-8")
    config = {
        "data": {
            "accuracy_program": {
                "path": str(path),
                "sha256": sha256(path),
            }
        }
    }
    assert load_locked_accuracy_config(config) == {"version": 1}

    path.write_text("version: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Accuracy policy checksum mismatch"):
        load_locked_accuracy_config(config)

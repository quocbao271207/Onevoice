import hashlib
import json
from pathlib import Path

import pytest

from scripts import preflight_project


def inventory_row(root: Path, audio: Path, *, record_id: str = "sample") -> dict:
    payload = audio.read_bytes()
    return {
        "audio_path": audio.relative_to(root).as_posix(),
        "bytes": len(payload),
        "id": record_id,
        "role": "train",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source": "sample",
    }


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_preflight_json_inputs_are_strict_and_bounded(tmp_path: Path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"status":"pass","status":"fail"}', encoding="utf-8")
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        preflight_project.load_json(duplicate)

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"id":"one","id":"two","audio_path":"missing.flac"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strict UTF-8 JSONL"):
        preflight_project.local_audio_manifest(manifest)


def test_preflight_local_manifest_rejects_invalid_identity_and_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(preflight_project, "ROOT", tmp_path)
    audio = tmp_path / "data" / "processed" / "audio_16k" / "sample" / "one.flac"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio")
    manifest = write_jsonl(
        tmp_path / "manifest.jsonl",
        [
            {"id": "valid", "audio_path": audio.relative_to(tmp_path).as_posix()},
            {"id": 2, "audio_path": "outside.flac"},
        ],
    )

    assert preflight_project.local_audio_manifest(manifest) == (2, 1, 1)


def test_preflight_audio_inventory_verifies_stable_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(preflight_project, "ROOT", tmp_path)
    audio = tmp_path / "data" / "processed" / "audio_16k" / "sample" / "train" / "one.flac"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio-payload")
    inventory = write_jsonl(
        tmp_path / "inventory.jsonl",
        [inventory_row(tmp_path, audio)],
    )

    assert preflight_project.verify_audio_inventory(inventory) == (
        True,
        f"files=1 bytes={len(audio.read_bytes())}",
    )

    audio.write_bytes(b"tampered")
    passed, detail = preflight_project.verify_audio_inventory(inventory)
    assert passed is False
    assert detail == "size_mismatch=data/processed/audio_16k/sample/train/one.flac"


def test_preflight_audio_inventory_rejects_escape_and_duplicate_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(preflight_project, "ROOT", tmp_path)
    audio_root = tmp_path / "data" / "processed" / "audio_16k" / "sample" / "train"
    audio_root.mkdir(parents=True)
    first = audio_root / "one.flac"
    second = audio_root / "two.flac"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    inventory = tmp_path / "inventory.jsonl"

    escaped = inventory_row(tmp_path, first)
    escaped["audio_path"] = "outside.flac"
    write_jsonl(inventory, [escaped])
    passed, detail = preflight_project.verify_audio_inventory(inventory)
    assert passed is False
    assert "invalid_audio=" in detail

    write_jsonl(
        inventory,
        [
            inventory_row(tmp_path, first, record_id="one"),
            inventory_row(tmp_path, second, record_id="two"),
        ],
    )
    passed, detail = preflight_project.verify_audio_inventory(inventory)
    assert passed is False
    assert "duplicate_audio_hash=" in detail


def test_preflight_audio_inventory_rejects_ambiguous_or_overflowed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(preflight_project, "ROOT", tmp_path)
    inventory = tmp_path / "inventory.jsonl"

    inventory.write_text(
        '{"audio_path":"file.flac","bytes":1,"bytes":2,'
        '"id":"one","sha256":"' + "0" * 64 + '"}\n',
        encoding="utf-8",
    )
    passed, detail = preflight_project.verify_audio_inventory(inventory)
    assert passed is False
    assert "strict UTF-8 JSONL" in detail

    inventory.write_text(
        '{"audio_path":"file.flac","bytes":1e400,'
        '"id":"one","sha256":"' + "0" * 64 + '"}\n',
        encoding="utf-8",
    )
    passed, detail = preflight_project.verify_audio_inventory(inventory)
    assert passed is False
    assert "strict UTF-8 JSONL" in detail

import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_audio_inventory
from src.pipeline.stable_json import read_stable_json_mapping
from src.pipeline.stable_jsonl import read_stable_jsonl_mappings


def configure_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    duplicate_audio: bool = False,
) -> tuple[Path, tuple[Path, ...]]:
    manifest_root = tmp_path / "data" / "processed" / "manifests"
    audio_root = tmp_path / "data" / "processed" / "audio_16k" / "sample"
    manifest_root.mkdir(parents=True)
    manifests: list[Path] = []
    for index, role in enumerate(("train", "validation", "test"), 1):
        audio = audio_root / role / f"{index}.flac"
        audio.parent.mkdir(parents=True)
        payload = b"same" if duplicate_audio and role == "test" else f"audio-{index}".encode()
        audio.write_bytes(payload)
        manifest = manifest_root / f"asr--{role}-local.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "audio_path": audio.relative_to(tmp_path).as_posix(),
                    "id": f"id-{index}",
                    "role": role,
                    "source": "sample",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        manifests.append(manifest)
    if duplicate_audio:
        validation_audio = audio_root / "validation" / "2.flac"
        validation_audio.write_bytes(b"same")
    monkeypatch.setattr(build_audio_inventory, "ROOT", tmp_path)
    monkeypatch.setattr(build_audio_inventory, "LOCAL_MANIFESTS", tuple(manifests))
    return tmp_path / "inventory.jsonl", tuple(manifests)


def test_audio_inventory_is_deterministic_strict_and_crash_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output, _manifests = configure_inputs(tmp_path, monkeypatch)

    summary = build_audio_inventory.build_inventory(output, workers=2)

    document = read_stable_jsonl_mappings(
        output,
        maximum_bytes=10_000,
        maximum_line_bytes=1_000,
        maximum_rows=10,
        label="Test inventory",
    )
    summary_path = tmp_path / "inventory_summary.json"
    persisted_summary = read_stable_json_mapping(
        summary_path,
        maximum_bytes=10_000,
        label="Test inventory summary",
    ).mapping
    assert len(document.rows) == 3
    assert [row["role"] for row in document.rows] == ["test", "train", "validation"]
    assert summary == persisted_summary
    assert summary["inventory_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert summary["files"] == 3
    assert summary["unique_audio_hashes"] == 3
    assert not list(tmp_path.glob(".inventory*.tmp"))


def test_audio_inventory_rejects_duplicate_content_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output, _manifests = configure_inputs(
        tmp_path,
        monkeypatch,
        duplicate_audio=True,
    )

    with pytest.raises(ValueError, match="Exact-audio dedup gate failed"):
        build_audio_inventory.build_inventory(output, workers=2)

    assert not output.exists()
    assert not (tmp_path / "inventory_summary.json").exists()


def test_audio_inventory_rejects_ambiguous_manifest_and_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output, manifests = configure_inputs(tmp_path, monkeypatch)
    manifests[0].write_text(
        '{"id":"one","id":"two","audio_path":"outside.flac",'
        '"role":"train","source":"sample"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strict UTF-8 JSONL"):
        build_audio_inventory.build_inventory(output, workers=1)

    outside = tmp_path / "outside.flac"
    outside.write_bytes(b"outside")
    manifests[0].write_text(
        json.dumps(
            {
                "id": "one",
                "audio_path": "outside.flac",
                "role": "train",
                "source": "sample",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must remain under"):
        build_audio_inventory.build_inventory(output, workers=1)
    assert not output.exists()


@pytest.mark.parametrize("workers", [0, 33, True])
def test_audio_inventory_bounds_worker_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workers: object,
):
    output, _manifests = configure_inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="workers must be an integer within"):
        build_audio_inventory.build_inventory(output, workers=workers)

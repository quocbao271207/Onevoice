from __future__ import annotations

from pathlib import Path

import pytest

import src.pipeline.adapter_evidence as adapter_evidence
from scripts.candidate_evidence import adapter_identity
from scripts.run_model_bakeoff import tree_manifest
from src.pipeline.adapter_evidence import stable_adapter_tree_manifest


def make_adapter(root: Path) -> Path:
    adapter = root / "adapter"
    nested = adapter / "nested"
    nested.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (nested / "weights.bin").write_bytes(b"weights")
    return adapter


def test_all_adapter_evidence_consumers_share_one_canonical_identity(tmp_path: Path):
    adapter = make_adapter(tmp_path)

    stable = stable_adapter_tree_manifest(adapter)
    candidate = adapter_identity(adapter)
    bakeoff = tree_manifest(adapter)

    assert candidate == {
        "path": stable["root"],
        "file_count": stable["file_count"],
        "bytes": stable["bytes"],
        "manifest_sha256": stable["manifest_sha256"],
    }
    assert bakeoff == {
        "root": stable["root"],
        "files": stable["files"],
        "manifest_sha256": stable["manifest_sha256"],
    }


def test_adapter_evidence_rejects_linked_ancestor(tmp_path: Path):
    real_parent = tmp_path / "real-parent"
    adapter = make_adapter(real_parent)
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="symlink or junction"):
        stable_adapter_tree_manifest(linked_parent / adapter.name)


def test_adapter_evidence_rejects_simulated_nested_reparse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = make_adapter(tmp_path)
    weights = adapter / "nested" / "weights.bin"
    original = adapter_evidence.is_link_or_junction
    monkeypatch.setattr(
        adapter_evidence,
        "is_link_or_junction",
        lambda path: Path(path) == weights or original(Path(path)),
    )

    with pytest.raises(ValueError, match="symlinks or junctions"):
        stable_adapter_tree_manifest(adapter)


def test_adapter_evidence_rejects_missing_config_and_file_cap(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "weights.bin").write_bytes(b"weights")

    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        stable_adapter_tree_manifest(adapter)

    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds 1 files"):
        stable_adapter_tree_manifest(adapter, maximum_files=1)


def test_adapter_evidence_detects_content_change_between_tree_passes(
    tmp_path: Path,
):
    adapter = make_adapter(tmp_path)
    weights = adapter / "nested" / "weights.bin"
    from src.utils.bounded_file import sha256_stable_regular_file

    hashed_weights = False

    def mutate_after_first_hash(path: Path, **kwargs):
        nonlocal hashed_weights
        result = sha256_stable_regular_file(path, **kwargs)
        if Path(path) == weights and not hashed_weights:
            hashed_weights = True
            weights.write_bytes(b"changed")
        return result

    with pytest.raises(RuntimeError, match="changed while hashing"):
        stable_adapter_tree_manifest(adapter, hash_file=mutate_after_first_hash)

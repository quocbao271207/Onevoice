"""Explicit, registry-locked Hugging Face dataset downloader.

This utility is intentionally not part of the normal OneVoice pipeline: the
audited path is ``audit_datasets.py`` -> ``merge_manifests.py`` ->
``materialize_audio.py``. It exists only for an owner-requested local snapshot.

It reads ``configs/datasets.yaml`` and ``configs/artifact_lock.yaml`` so a
stale hard-coded source cannot silently enter the project. Disabled sources are
never downloadable here, and evaluation-only sources require an explicit flag.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = ROOT / "configs" / "datasets.yaml"
ARTIFACT_LOCK = ROOT / "configs" / "artifact_lock.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_registry() -> tuple[dict, dict]:
    registry = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))["datasets"]
    revisions = yaml.safe_load(ARTIFACT_LOCK.read_text(encoding="utf-8"))["datasets"]
    return registry, revisions


def download_dataset(
    name: str,
    output_dir: Path,
    max_samples: int | None,
    allow_evaluation_only: bool,
) -> Path:
    from datasets import load_dataset

    registry, revisions = load_registry()
    if name not in registry:
        raise KeyError(f"Unknown dataset: {name}")

    spec = registry[name]
    if not spec.get("enabled", False):
        raise ValueError(
            f"Dataset '{name}' is disabled: {spec.get('reason_disabled', 'no reason recorded')}"
        )
    if spec.get("evaluation_only", False) and not allow_evaluation_only:
        raise ValueError(
            f"Dataset '{name}' is evaluation-only; pass --allow-evaluation-only to snapshot it."
        )

    repo_id = spec["repo_id"]
    revision = revisions.get(repo_id)
    if not revision:
        raise ValueError(f"No immutable revision lock for enabled source {repo_id}")

    kwargs = {"revision": revision}
    config_name = spec.get("config")
    if config_name and config_name != "default":
        kwargs["name"] = config_name

    logger.info("Loading %s at revision %s", repo_id, revision)
    dataset = load_dataset(repo_id, **kwargs)
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("--max-samples must be positive")
        for split_name in list(dataset):
            dataset[split_name] = dataset[split_name].select(
                range(min(max_samples, len(dataset[split_name])))
            )

    save_path = output_dir / name
    save_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(save_path))
    metadata = {
        "dataset_key": name,
        "repo_id": repo_id,
        "revision": revision,
        "task": spec.get("task"),
        "evaluation_only": bool(spec.get("evaluation_only", False)),
        "splits": {split: len(rows) for split, rows in dataset.items()},
        "note": "Snapshot only; merge/training eligibility is controlled by configs/datasets.yaml.",
    }
    (save_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Saved immutable snapshot to %s", save_path)
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Enabled dataset key; required unless --list is used")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw_snapshots")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--allow-evaluation-only", action="store_true")
    parser.add_argument("--list", action="store_true", help="List registry status and exit")
    args = parser.parse_args()

    registry, _ = load_registry()
    if args.list:
        for name, spec in registry.items():
            status = "enabled" if spec.get("enabled", False) else "disabled"
            role = "evaluation-only" if spec.get("evaluation_only", False) else spec.get("task", "")
            print(f"{name:24s} {status:8s} {role:16s} {spec['repo_id']}")
        return
    if not args.dataset:
        parser.error("--dataset is required; bulk download is intentionally disabled")

    download_dataset(
        args.dataset,
        args.output_dir,
        args.max_samples,
        args.allow_evaluation_only,
    )


if __name__ == "__main__":
    main()

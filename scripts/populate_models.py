#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

# Mapping of local model names to Hugging Face repo IDs.
MODELS = {
    "tiny-random-gpt2": "hf-internal-testing/tiny-random-gpt2",
}

ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.bin",
    "*.model",
    "*.txt",
    "*.py",
]


def _remove_extra_dirs(base_dir: Path) -> None:
    for cache_dir in base_dir.rglob(".cache"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
    for inf_dir in base_dir.rglob("inference"):
        if inf_dir.is_dir():
            shutil.rmtree(inf_dir)


def populate_models(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)

    for name, repo_id in MODELS.items():
        target_dir = base_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            allow_patterns=ALLOW_PATTERNS,
        )

    _remove_extra_dirs(base_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("models"),
        help="Directory where model subfolders will be created.",
    )
    args = parser.parse_args()
    populate_models(args.base_dir)


if __name__ == "__main__":
    main()

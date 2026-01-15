#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

# Mapping of local tokenizer names to Hugging Face repo IDs.
TOKENIZERS = {
    "gpt2": "openai-community/gpt2",
    "gpt3": "Xenova/gpt-3",
    "gpt-oss": "openai/gpt-oss-20b",
    "claude1": "Xenova/claude-tokenizer",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "llama4": "meta-llama/Llama-4-Scout-17B-16E",
    "qwen15": "Qwen/Qwen1.5-0.5B",
    "qwen25": "Qwen/Qwen2.5-1.5B",
    "qwen3": "Qwen/Qwen3-1.7B-Base",
    "deepseek15": "deepseek-ai/deepseek-coder-7b-base-v1.5",
    "deepseek25": "deepseek-ai/DeepSeek-V2.5",
    "deepseek32": "deepseek-ai/DeepSeek-V3.2-Exp",
    "glm45": "zai-org/GLM-4.5",
    "smol1": "HuggingFaceTB/SmolLM-135M",
    "smol2": "HuggingFaceTB/SmolLM2-135M",
    "smol3": "HuggingFaceTB/SmolLM3-3B-Base",
    "olmo2": "allenai/OLMo-2-0425-1B",
    "neox": "EleutherAI/gpt-neox-20b",
    "phi1": "microsoft/phi-1",
    "phi2": "microsoft/phi-2",
    "phi4": "microsoft/phi-4",
    "grok2": "alvarobartt/grok-2-tokenizer",
    "mistralnemo": "mistralai/Mistral-Nemo-Base-2407",
    "starcoder1": "bigcode/starcoder",
    "starcoder2": "bigcode/starcoder2-15b",
    "comma": "common-pile/comma-v0.1-1t",
    "apertus": "swiss-ai/Apertus-8B-2509",
    "kl3m": "alea-institute/kl3m-003-1.7b",
    "nemotronh": "nvidia/Nemotron-H-8B-Base-8K",
    "nemotronnano": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    "bpe": "UW/OLMo2-8B-BPE",
    "superbpe": "UW/OLMo2-8B-SuperBPE-t180k",
}

ALLOW_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "vocab.txt",
    "vocab.*",
    "*.model",
    "*.spm",
    "*.tiktoken",
    "*.txt",
    "config.json",
]


def _remove_extra_dirs(base_dir: Path) -> None:
    for cache_dir in base_dir.rglob(".cache"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
    for inf_dir in base_dir.rglob("inference"):
        if inf_dir.is_dir():
            shutil.rmtree(inf_dir)


def populate_tokenizers(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)

    for name, repo_id in TOKENIZERS.items():
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
        default=Path("tokenizers"),
        help="Directory where tokenizer subfolders will be created.",
    )
    args = parser.parse_args()
    populate_tokenizers(args.base_dir)


if __name__ == "__main__":
    main()

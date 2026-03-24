# Download Llama 3 8B Instruct model files from HuggingFace.
# Requires HF_TOKEN env var for gated model access.
#
# Usage:
#   HF_TOKEN=... python tools/llama3_downloader.py --out ./assets/llama3/
#
# THE MODEL FILES "MUST" BE ASSUMED TO BE IN ./assets/llama3/ FOR OTHER SCRIPTS.
# DO NOT CHANGE THIS FILE

import os
import argparse
from huggingface_hub import snapshot_download

# HuggingFace repo for Llama 3 8B Instruct (gated — requires accepted license).
REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# By default, only download the files needed for inference (weights, configs, tokenizer).
# Use --full to download the entire repo including README, license, etc.
DEFAULT_ALLOW = [
    # model weights (Transformers / safetensors)
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
    # configs
    "config.json",
    "generation_config.json",
    # tokenizer (some repos have tokenizer.json, some have tokenizer.model too)
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        required=False,
        help="Output directory, e.g. ./checkpoints/llama3_8b_instruct",
    )
    ap.add_argument(
        "--revision", default="main", help="Branch/tag/commit (default: main)"
    )
    ap.add_argument(
        "--full", action="store_true", help="Download full repo (no filtering)"
    )
    args = ap.parse_args()

    # Require a HuggingFace token (gated model).
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "Missing HF token.\n"
            "Do one of:\n"
            "  export HF_TOKEN=...   (or HUGGINGFACE_HUB_TOKEN)\n"
            "or\n"
            "  huggingface-cli login\n"
        )

    allow_patterns = None if args.full else DEFAULT_ALLOW

    # Download the snapshot — supports resume if interrupted.
    path = snapshot_download(
        repo_id=REPO_ID,
        revision=args.revision,
        local_dir=args.out or "./assets/llama3/",
        local_dir_use_symlinks=False,  # real copy, not symlinks (portable for containers)
        token=token,
        allow_patterns=allow_patterns,
        max_workers=8,
        resume_download=True,
    )

    print("Downloaded to:", path)
    if not args.full:
        print("Downloaded (filtered) patterns:", DEFAULT_ALLOW)


if __name__ == "__main__":
    main()



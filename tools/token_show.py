"""Reference tokenizer for verifying our C++ BPE implementation.

Loads the HuggingFace Llama 3 tokenizer from local assets and prints
the token IDs for a given sentence. Compare output with the C++ tokenizer
to confirm they produce identical IDs.

Usage:
  python tools/token_show.py "Hello world"
  python tools/token_show.py          # prompts for input

Requires: model files already downloaded to assets/llama3/ (run llama3_downloader.py first).

DO NOT CHANGE THIS FILE
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from transformers import AutoTokenizer


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tokenize a sentence with Hugging Face and print token IDs",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Sentence to tokenize. If omitted, you will be prompted.",
    )

    parser.set_defaults(add_special_tokens=True)

    args = parser.parse_args(argv)

    # Get input text from CLI args or interactive prompt.
    if args.text:
        text = " ".join(args.text)
    else:
        try:
            text = input("Enter a sentence: ").strip()
        except EOFError:
            print("No input provided.", file=sys.stderr)
            return 2

    if not text:
        print("Empty input.", file=sys.stderr)
        return 2

    # Load the HF tokenizer from local files only (no network calls).
    tokenizer = AutoTokenizer.from_pretrained(
        'assets/llama3',
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )

    # Encode text and print the resulting token IDs for comparison with C++ output.
    token_ids = tokenizer.encode(text, add_special_tokens=args.add_special_tokens)
    print(token_ids)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

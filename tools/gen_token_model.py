#!/usr/bin/env python3
"""Convert HuggingFace tokenizer.json to the BPE rank file the C++ tokenizer reads.

Output format: one line per token as `<base64-bytes> <rank>`.
"""

import argparse
import base64
import json
from pathlib import Path


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="assets/llama3/tokenizer.json")
    ap.add_argument("--out", default="assets/llama3/token.model")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        raise SystemExit(f"input not found: {inp}")

    vocab = json.loads(inp.read_text())["model"]["vocab"]
    u2b = {chr(v): bytes([k]) for k, v in bytes_to_unicode().items()}

    def to_bytes(s: str) -> bytes:
        return b"".join((u2b[c] if c in u2b else c.encode("utf-8")) for c in s)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for s, r in sorted(vocab.items(), key=lambda kv: kv[1]):
            f.write(f"{base64.b64encode(to_bytes(s)).decode()} {r}\n")
    print(f"wrote {len(vocab)} entries -> {out}")


if __name__ == "__main__":
    main()

# Test 4 — Embedding lookup (larger)

**Source:** `tests/test.cpp:176-185`
**Fixture:** `tests/data/test4_embedding.bin`
**Run:** `./bin/tests 4`

## What it does

Same logic as Test 3 — look up rows from the embedding table — but with a
larger list of tokens.

## Why run two embedding tests?

Because shape-dependent bugs are sneaky. With a small fixture (Test 3), a
stride error might accidentally land on the same correct row. A larger
fixture (Test 4) makes the error surface.

Common examples of bugs Test 4 catches that Test 3 misses:

- Off-by-one in the output buffer size (`n * 4096` vs `(n-1) * 4096`).
- Incorrect loop bounds that only fail past a certain `n`.
- Memory reuse bugs where each new lookup corrupts the previous one.
- Integer overflow in an index calculation with big `n`.

## Format

Identical to Test 3:

- 4 bytes: `int n`
- `n * 4` bytes: token IDs
- `n * 4096 * 4` bytes: expected FP32 embeddings

## What a failure usually means

If Test 3 passes but Test 4 fails, the bug is almost always **size- or
stride-dependent**. Suspect: buffer allocation, loop bounds, or a
per-element state that isn't being reset.

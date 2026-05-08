# Test 3 — Embedding lookup (small)

**Source:** `tests/test.cpp:165-174`
**Fixture:** `tests/data/test3_embedding.bin`
**Run:** `./bin/tests 3`

## What it does

Given a list of token IDs, returns the corresponding rows from the model's
**embedding table** — one row of 4096 floating-point numbers per token. The
test compares your output to a known-good answer within `1e-2` tolerance.

## Why it matters

Every token has a "meaning vector" — a list of 4096 numbers that the model
uses internally. These live in a big table on disk (about 500 MB). For any
token ID `i`, its vector is row `i` of that table.

This test is the first place floating-point comes in. If your row indexing
is off, or your type conversion is broken, you'll see it here.

## The tricky bit: BF16 → FP32

The embedding table is stored in **BF16** — a compact 16-bit float format
that saves half the memory. Your code has to unpack each value into a
regular 32-bit float before comparing.

BF16 is easy to convert: take the 16 bits, shift them into the top half of
a 32-bit integer, and reinterpret as a float. But it's easy to get wrong
(e.g., wrong byte order, wrong shift amount).

## Fixture format

- 4 bytes: `int n` (number of tokens)
- `n * 4` bytes: token IDs
- `n * 4096 * 4` bytes: expected embeddings (FP32)

## What a failure usually means

- Wrong row offset: `token_id * 4096` is right; `token_id * EMBEDDING_DIM`
  is also right; anything else is wrong.
- BF16 conversion bug: off by a large factor, or sign-flipped.
- Header parse bug: the binary dump has a 280-byte header you must skip.

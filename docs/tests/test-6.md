# Test 6 — Matmul, seq_len = 10

**Source:** `tests/test.cpp:209-213`
**Fixture:** `tests/data/test6_matmul.bin`
**Run:** `./bin/tests 6`

## What it does

Same kernel as Test 5, but with `M = 10` rows instead of 1. Checks the
output matches within `1e-2`.

## Why this shape?

`seq_len = 10` is the "in-between" shape. It's not the single-row case
from Test 5, but it's also not a full 100+ row batch. A 10-row matmul
typically spans a **partial tile** on the GPU.

## What it catches

GPU kernels usually process data in fixed-size tiles (say, 32 rows at a
time). With 10 rows, the kernel has to handle a tile that's only
**partially full**. Common bugs at this size:

- The partially-filled tile reads uninitialized memory.
- Reduction logic double-counts or skips rows on the partial tile.
- Thread indexing wraps around and writes to the wrong output position.

If Test 5 (M=1) and Test 7 (M=100) both pass but Test 6 fails, you almost
certainly have a partial-tile bug.

## Fixture format

Identical to Test 5:

- 4 bytes: `int M` (= 10)
- 4 bytes: `int K`
- 4 bytes: `int N`
- `M*K * 4` bytes: A
- `K*N * 4` bytes: B
- `M*N * 4` bytes: C (expected)

## What a failure usually means

- Partial-tile handling is broken.
- You assumed M is always a multiple of your tile size.
- Your bounds check uses `<` where it should use `<=` (or vice versa).

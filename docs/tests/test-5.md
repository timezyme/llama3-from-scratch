# Test 5 — Matmul, seq_len = 1

**Source:** `tests/test.cpp:203-207`
**Fixture:** `tests/data/test5_matmul.bin`
**Run:** `./bin/tests 5`

## What it does

Multiplies a **1-row matrix** by a weight matrix on the GPU and checks the
result matches the expected output within `1e-2`.

Think of it as: `C = A × B`, where `A` has shape `1 × K`, `B` has shape
`K × N`, and `C` has shape `1 × N`.

## Why this shape?

`seq_len = 1` is the shape used during **token-by-token generation**. After
the prompt is processed, every new token the model produces goes through
this exact skinny-matrix multiply, many times per token.

## Why it's harder than it looks

A 1-row matrix is a **tough shape for a GPU**. GPUs are happiest when
they can chew on big square tiles — that's how they get their speed. With
only 1 row to work on, most of the tile slots sit idle, and you have to
make sure:

- Your kernel doesn't read past the end of the 1-row input.
- Your tile boundary guards kick in correctly.
- You don't accidentally return garbage from the unused tile slots.

## Fixture format

- 4 bytes: `int M` (rows of A = 1)
- 4 bytes: `int K` (cols of A, rows of B)
- 4 bytes: `int N` (cols of B)
- `M*K * 4` bytes: A (FP32)
- `K*N * 4` bytes: B (FP32)
- `M*N * 4` bytes: C (expected, FP32)

## Extra: timing

Unlike Tests 1-4, the matmul tests print how long your GPU kernel took.
This isn't graded for correctness, but gives you a feel for how fast your
implementation is.

## What a failure usually means

- Boundary guard bug (reading past row 0 into garbage memory).
- Wrong layout assumption (row-major vs column-major).
- Kernel launch config that assumes M > 1.

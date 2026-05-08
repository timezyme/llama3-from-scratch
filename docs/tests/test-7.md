# Test 7 — Matmul, seq_len = 100

**Source:** `tests/test.cpp:215-219`
**Fixture:** `tests/data/test7_matmul.bin`
**Run:** `./bin/tests 7`

## What it does

Same kernel as Tests 5 and 6, but with `M = 100` rows — the full "batch"
shape. Checks the result matches within `1e-2`.

## Why this shape?

`seq_len = 100` is the **prefill** shape: when you first feed a prompt
into the model, all the prompt's tokens are processed together in one big
batch. This is the shape where GPUs actually get to show off.

## What it catches

With 100 rows, the GPU is running at full occupancy — many tiles in
flight, threads coordinating through shared memory, the whole machine
humming. Bugs that appear here but not at M=1 or M=10:

- **Shared-memory race conditions.** Multiple thread blocks or warps
  writing to the same shared-memory slot without proper synchronization.
- **Accumulation drift.** Over many K iterations, small floating-point
  errors compound. If your order of operations is subtly wrong, the final
  sum won't match.
- **Double-buffering bugs.** High-performance kernels overlap loading the
  next tile with computing the current one. Getting the hand-off wrong
  produces stale data.
- **Register pressure.** A kernel that works on small inputs may spill
  registers to slow memory at this size, causing perf to crater (though
  correctness should still hold).

## Fixture format

Identical to Tests 5 and 6:

- 4 bytes: `int M` (= 100)
- 4 bytes: `int K`
- 4 bytes: `int N`
- `M*K * 4` bytes: A
- `K*N * 4` bytes: B
- `M*N * 4` bytes: C (expected)

## What the tolerance means

`EPSILON = 1e-2` is loose enough to absorb legitimate floating-point
reordering differences across GPU vs. CPU (the expected output is
computed with a different reduction order). It's tight enough that a real
arithmetic bug produces a max-error in the 1+ range, not a tenth.

## What a failure usually means

- Synchronization bug (missing `__syncthreads()` inside the kernel).
- Wrong shared-memory layout.
- Double-buffer hand-off is stale.
- Accumulation done in a pathological order.

---

## Step 8: RMSNorm Kernel

**File:** `kernel/rmsnorm.cu:96-147`
**Where in the pipeline:** We're now inside the decoder block. RMSNorm is the *very first* operation applied to the input `X` before anything else happens. It runs at three places in the model:
1. Before the Q/K/V projections (pre-attention norm) — every layer
2. Before the SwiGLU FFN (pre-FFN norm) — every layer
3. Once at the end, before lm_head (final norm)

That's 2 x 32 + 1 = **65 RMSNorm calls** per forward pass.

### High-level picture

RMSNorm answers the question: "how do I keep numbers from exploding or vanishing as they pass through 32 layers?" Each row of the input (one token's 4,096-dimensional vector) gets scaled so its values have a consistent magnitude, then multiplied by a learned scale vector `gamma`.

The formula for one row `x` of 4,096 values:

```
rms(x) = sqrt( mean(x_i^2) + epsilon )
y_i    = (x_i / rms(x)) * gamma_i
```

In plain English: compute the root-mean-square of the row, divide every element by it (normalizing), then multiply by `gamma` (re-scaling with learned weights).

### How the kernel works

One thread block per row. 256 threads stride across the 4,096 columns. Two passes:

**Pass 1 — Sum of squares** (lines 111-128). Each thread sums `x_i^2` for its ~16 columns, stores the partial sum in shared memory, then a tree reduction collapses 256 partial sums down to one total in `sdata[0]`. The tree reduction halves active threads at each step: 256 -> 128 -> 64 -> ... -> 1, with a `__syncthreads()` between each step.

**Compute** (line 138). Every thread independently computes `rms = sqrt(total / 4096 + epsilon)`. It's cheaper to have all 256 threads redo this one division+sqrt than to have one thread compute it and broadcast.

**Pass 2 — Scale and write** (lines 144-146). Each thread writes `y[i] = x[i] / rms * gamma[i]` for its columns.

### New concept: tree reduction

When 256 threads each have a partial result and you need the total, you can't just have thread 0 add them all up — that's sequential. Instead, the **tree reduction** pattern halves the work at each step:
- Step 1: threads 0-127 add threads 128-255's values
- Step 2: threads 0-63 add threads 64-127's values
- ... and so on for 8 steps (log2(256) = 8)

This gives you a parallel sum in O(log n) steps instead of O(n).

### TA-scrutiny items (Part 2 section 4 pitfalls)

Three things a TA will check here:

1. **Epsilon goes INSIDE the sqrt** (line 138): `sqrt(mean + epsilon)`, not `sqrt(mean) + epsilon`. If a row is all zeros, `mean = 0`, and `sqrt(0) + epsilon` gives a tiny divisor that blows the output up. `sqrt(0 + epsilon)` gives `sqrt(1e-5) ≈ 0.003`, which is safe.

2. **Gamma must not be skipped** (line 145): the `* gamma[i]` is not optional. Each call site has its own learned gamma vector. Leaving it out makes RMSNorm output look "roughly right" but the model produces garbage after a few layers.

3. **Two separate gamma vectors per layer**: `input_layernorm.weight` (before attention) and `post_attention_layernorm.weight` (before FFN). Using the same one for both is a loading bug we covered in step 5.

---

**TA-style question:**

This kernel reads the entire row from HBM twice — once in pass 1 (sum of squares) and once in pass 2 (normalize + write). For `cols = 4096` in FP32, how many bytes of HBM traffic does that produce per row? If you could keep the row in shared memory between the two passes, how much would you save — and why doesn't this kernel do that?

**answer**

Two reads + one write of 4,096 floats per row = 3 x 4,096 x 4 = **49,152 bytes** per row.

If you cached the row in shared memory after pass 1, you'd skip the second HBM read, saving 16,384 bytes — cutting read traffic by half (2 reads down to 1).

Why this kernel doesn't do it: shared memory is tiny (~48-100 KB per block depending on GPU config). One row of 4,096 floats = 16 KB. The reduction already uses 256 x 4 = 1 KB of shared memory. Fitting the full row alongside that is *possible* on most GPUs (16 KB + 1 KB = 17 KB), but it tightens the shared-memory budget and can reduce **occupancy** — the number of thread blocks that can run simultaneously on one SM. Lower occupancy means fewer blocks to hide memory latency. For a kernel this simple (two short passes over 4,096 elements), the second HBM read is fast enough that the tradeoff isn't worth the occupancy hit.

---
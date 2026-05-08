---

## Step 9: Q/K/V Projections

**File:** `src/inference_layer.cu:306-348`
**Where in the pipeline:** Inside the decoder block, right after RMSNorm. The normalized hidden state `X_norm` gets projected into three different spaces: queries (Q), keys (K), and values (V). These are the inputs to attention.

### High-level picture

After RMSNorm normalizes `X` into `X_norm` (shape `[s, 4096]`), three matmuls produce the raw material for attention:

```
Q = X_norm @ W_Q    →  [s, 4096]    (32 heads x 128 dims)
K = X_norm @ W_K    →  [s, 1024]    ( 8 heads x 128 dims)
V = X_norm @ W_V    →  [s, 1024]    ( 8 heads x 128 dims)
```

Each is a direct call to your matmul kernel — this is the first place where the matmul you built in step 7 gets used for real inference work. The weights were already transposed at load time (step 5), so the kernel computes `X_norm @ W_stored` with no runtime transpose.

Q is wider than K/V because of **GQA (Grouped-Query Attention)** — 32 query heads but only 8 key/value heads. Every 4 query heads share one K/V head, so K and V only need 8 x 128 = 1,024 columns instead of 4,096.

### How the code handles it

Look at lines 323-348. There's a subtlety in how Q vs K/V are computed:

**Q** (line 324): Computed for the whole batch at once — one matmul call with `rows = batch * q_seq`. Q is a temporary buffer used this step only.

**K and V** (lines 326-335): Computed per batch slot in a loop. Why? Because K and V get written directly into the **KV cache** at the correct position: `cache.k_at(layer, len_before, b)`. The cache stores all past K/V values so future decode steps can see them. Each batch slot has its own region in the cache, so the writes must be per-slot.

Both the resident path (BF16 weights, line 324) and streaming path (FP32 weights, line 337) do the same math — they just call different matmul entry points.

### New concept: KV cache write-through

This is the first time the KV cache appears in the actual compute flow. K and V aren't just computed and discarded — they're written into a persistent per-layer buffer. When the model generates token 2, the attention mechanism needs K/V from token 1. Instead of recomputing them, the cache holds them. We'll cover the cache in detail at step 17, but notice that the K/V projections here are already writing into it at line 331/334.

### TA-scrutiny items

- **Weight transpose**: `X_norm @ W_stored` works because W was transposed at load time. If a TA asks "isn't the math supposed to be `X @ W^T`?", yes — and it is, because W_stored IS W^T.
- **GQA dimension difference**: Q output is 4096-wide, K/V outputs are 1024-wide. Know that this comes from 32 vs 8 heads, not from a different head dimension (head_dim = 128 for all three).

---

**TA-style question:**

The Q projection is a single matmul for the entire batch (`rows = batch * q_seq`), but K and V each run in a per-batch-slot loop. Both take the same `X_norm` input and are the same kind of matmul. Why can't K and V be computed as a single batched matmul the same way Q is?

**answer**

Q gets written into a temporary buffer `d_Q` that's a flat `[batch * q_seq, 4096]` matrix — all batch slots packed contiguously in memory. A single matmul fills the whole thing because the output rows are adjacent.

K and V are different: they get written into the **KV cache**, where each batch slot has its own separate memory region. Batch 0's K cache and batch 1's K cache aren't contiguous — they're at different base pointers (`cache.k_at(layer, len_before, 0)` vs `cache.k_at(layer, len_before, 1)`). A single matmul can only write to one contiguous output buffer, so you need one matmul per batch slot to target each slot's cache region.

You *could* compute K/V into a flat temporary buffer (like Q) and then copy the results into the cache, but that adds an extra memcpy. The per-slot loop avoids that by writing directly where the data needs to live.

---
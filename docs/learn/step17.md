---

## Step 17: KV Cache (Bonus Feature)

**Files:** `include/kv_cache.h`, `src/kv_cache.cu`
**Where in the pipeline:** Cross-cutting — the KV cache connects every decode step to the past. It's what makes multi-token generation efficient.

### The problem it solves

Without a KV cache, generating T tokens from a prompt of length s₀ requires T full forward passes on sequences of increasing length: s₀+1, s₀+2, ..., s₀+T. Each pass recomputes K and V for *every* token, including the ones you already processed. The total work scales as **O(T² · s₀)** — quadratic in the number of generated tokens.

With a KV cache, each decode step only processes the *one new token* (q_seq=1). The K/V values for all previous positions are read from the cache instead of recomputed. Total work scales as **O(T · s₀)** — linear.

### How it works

The cache is a set of pre-allocated GPU buffers, one K and one V per layer:

```
Per layer:  K buffer → [batch, max_len, 1024]   (8 KV heads × 128 dims)
            V buffer → [batch, max_len, 1024]
× 32 layers × 2 (K + V)
```

Allocated at full capacity up front (`kv_cache.cu:39-58`). For S_MAX=1024, batch=1, that's 32 × 2 × 1024 × 1024 × 4 bytes = **256 MiB** — modest next to the ~16 GB of model weights.

### Two-phase usage

**Prefill** (first forward pass): The full prompt goes through. K/V projections write rows `[0, s₀)` into every layer's cache buffers. `cache.advance(s₀)` records that s₀ positions are now cached.

**Decode** (each subsequent step): Only the new token is projected. K/V projections write one row at position `len_before` (the current cache length). Attention reads the full `[0, len_before+1)` prefix from cache. `cache.advance(1)` bumps the count.

### The pointer API

The header (`kv_cache.h:76-91`) provides direct pointer access so the matmul kernel can write K/V straight into the cache:

- `k_at(layer, row, batch)` → pointer to the start of row `row` in layer's K buffer for batch slot `b`
- `k_batch(layer, b)` → pointer to batch slot `b`'s full K buffer (for attention to read)
- `v_at(...)` / `v_batch(...)` — same for V

This is how step 9 writes directly into the cache instead of an intermediate buffer.

### Why allocate up front

The buffers are `cudaMalloc`'d at full S_MAX capacity on construction. No resizing during generation. This avoids:
1. Expensive `cudaMalloc`/`cudaFree` mid-generation
2. Breaking pointers that kernels already hold
3. Device-to-device copies to migrate data to a larger buffer

`reset()` (line 41 of the header) rewinds the logical length to 0 without freeing — so the REPL mode can reuse the same cache for back-to-back prompts without reallocating.

### TA-scrutiny items

- **O(T) vs O(T²)**: The fundamental speedup. Without cache, T tokens costs O(T²) forward-pass work. With cache, O(T). Know why: each decode step processes 1 token instead of re-processing all previous tokens.
- **RoPE on cached K**: Only the *newly written* K rows get RoPE applied (step 10, `inference_layer.cu:357-360`). Older cached K rows already had RoPE applied when they were first written. Applying RoPE twice would double-rotate and corrupt the positional encoding.

---

**TA-style question:**

The KV cache stores K and V in FP32. For S_MAX=1024 and batch=1, it's 256 MiB. How does this scale with batch size and sequence length? At what point would the KV cache itself become a VRAM bottleneck on the L4 (24 GB total, ~16 GB used by BF16 model weights)?

**answer**

The KV cache scales linearly in both batch size and sequence length:

```
bytes = 32 layers × 2 (K+V) × batch × seq_len × 1024 × 4 bytes
      = 256 KiB × batch × seq_len
```

| Batch | Seq len | KV cache | Remaining VRAM (of ~8 GB free) |
| ----- | ------- | -------- | ------------------------------ |
| 1     | 1,024   | 256 MiB  | plenty                         |
| 1     | 4,096   | 1 GiB    | fine                           |
| 4     | 4,096   | 4 GiB    | tight                          |
| 8     | 4,096   | 8 GiB    | exceeds free VRAM              |

With ~16 GB of BF16 weights resident, that leaves ~8 GB for the KV cache, activations, and scratch buffers. At batch=1 you'd hit the wall around seq_len ~32,000 (8 GB / 256 KB ≈ 32K). At batch=4 it's ~8,000. At batch=8 it's ~4,000.

This is why production serving systems use techniques like PagedAttention (vLLM) — they manage KV cache memory like virtual memory pages so multiple requests can share GPU VRAM efficiently without pre-allocating worst-case capacity for every batch slot.

---
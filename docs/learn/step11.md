---

## Step 11: GQA Attention + Softmax

**Files:** `kernel/attention.cu:109-207` (kernels), `src/inference_layer.cu:87-129` (per-head orchestrator)
**Where in the pipeline:** After RoPE rotates Q and K, attention computes "how much should each token look at every other token?" This is the core mechanism that makes transformers work.

### High-level picture

Attention answers the question: for each token in the sequence, what weighted combination of all other tokens' values (V) should I use? The weights come from how similar each token's query (Q) is to every other token's key (K).

For each of the 32 query heads, the orchestrator in `inference_layer.cu:97-128` runs this pipeline:

```
1. Gather Q_i          [s, 128]    — slice head i out of packed Q
2. Gather K_g^T        [128, s]    — slice KV head g, transpose on the fly
3. Gather V_g          [s, 128]    — slice KV head g

4. S = Q_i @ K_g^T     [s, s]      — score matrix (matmul)
5. S *= 1/sqrt(128)                 — scale
6. Causal mask S                    — future tokens → -1e6
7. Softmax(S)                       — rows become probabilities

8. O_i = S @ V_g       [s, 128]    — weighted sum of values (matmul)
9. Scatter O_i                      — stitch back into packed output
```

The GQA mapping: query head `hi` shares KV head `g = hi / 4` (line 99). So heads 0-3 all read from the same K/V, heads 4-7 from the next, etc.

### The score matrix S

`S = Q_i @ K_g^T` produces an `[s, s]` matrix where `S[p, q]` is the raw dot-product similarity between token `p`'s query and token `q`'s key. High value = these tokens are relevant to each other.

**Scale** (line 114, `attention.cu:109-115`): Divide by `sqrt(128) ≈ 11.3`. Without this, the dot products grow proportionally to `sqrt(head_dim)` and push softmax into saturation (all probability on one token, gradients near zero).

**Causal mask** (line 116, `attention.cu:130-138`): Set `S[row, col] = -1e6` wherever `col > row`. This prevents token at position `p` from attending to any token at position `q > p` — it can't see the future. After softmax, `exp(-1e6) ≈ 0`, so those positions contribute nothing.

### Numerically stable softmax (`attention.cu:155-207`)

This is a **major TA-scrutiny point**. Three passes per row:

```
Pass 1: row_max = max(S[row, :])           ← tree reduction in shared memory
Pass 2: S[row, i] = exp(S[row, i] - row_max)   ← the stability trick
Pass 3: S[row, i] /= sum(S[row, :])        ← normalize to probabilities
```

The key is **subtracting `row_max` before `exp()`** in pass 2. Without it, if `S[row, i] = 200`, then `exp(200) ≈ 7.2 × 10^86` which overflows FP32 (max ~3.4 × 10^38). Result: `+inf`, then `inf/inf = NaN`, and the whole model produces garbage. Subtracting the max ensures the largest exponent is `exp(0) = 1` — always safe.

Mathematically equivalent because `exp(x - c) / sum(exp(x - c)) = exp(x) / sum(exp(x))` for any constant `c`.

### Gather/scatter: why not just pointer arithmetic?

Q, K, V are stored packed as `[s, num_heads * 128]`. But each head's matmul needs a contiguous `[s, 128]` slice. The gather kernels (lines 225-235) copy head `i`'s 128-column strip into a contiguous temporary buffer. The scatter kernel (lines 276-286) writes the result back. This is extra HBM traffic, but it lets the standard tiled matmul kernel work without needing a strided-input variant.

The K gather also transposes (`gather_head_transpose_kernel`, line 249), producing `K^T` directly — so the score matmul `Q_i @ K_g^T` is a standard row-major GEMM with no runtime transpose.

### New concepts

- **Score matrix S**: `[s, s]` matrix of pairwise attention scores. `S[p, q]` = how much token `p` attends to token `q`.
- **Causal masking**: The autoregressive constraint. Token 5 can see tokens 0-5 but not 6+. Applied by setting future positions to `-1e6` before softmax.
- **GQA group index**: `g = hi / (NUM_HEADS / NUM_KV_HEADS) = hi / 4`. Query heads 0-3 share KV head 0, heads 4-7 share KV head 1, etc.

### TA-scrutiny items (three from Part 2 section 4)

1. **Stable softmax** — subtract row max before exp(). Not optional. Without it, model produces NaN.
2. **Full causal mask** — every `(p, q)` with `q > p` across the entire `s x s` matrix. Not just the diagonal. Not just the last row.
3. **GQA group index** — `g = floor(i / (h/h_k))`. Know that 4 consecutive query heads share one K/V head.

---

**TA-style question:**

The score matrix S is `[s, s]` and is materialized in FP32 for each of the 32 heads. For a prompt of length `s = 512`, how much GPU memory do all 32 score matrices require simultaneously? How does this scale with `s`, and at what `s` does the score matrix memory exceed the total model weight memory (~16 GB in BF16)?

**answer**

For `s = 512`: each score matrix is `512 x 512 x 4 bytes = 1 MB`. Times 32 heads = **32 MB**. That's tiny.

But it scales as **O(s^2)** — quadratic in sequence length. Double `s` and the score matrices quadruple:

| s      | Per-head | 32 heads |
| ------ | -------- | -------- |
| 512    | 1 MB     | 32 MB    |
| 4,096  | 64 MB    | 2 GB     |
| 16,384 | 1 GB     | 32 GB    |

At `s ≈ 16,384`, the 32 score matrices alone (~32 GB) exceed the total model weights (~16 GB BF16). That's why long-context inference is hard — you hit a memory wall from attention, not from model size. It's also why techniques like FlashAttention exist: they compute softmax in tiles without ever materializing the full `s x s` matrix.

Note: this implementation only materializes one head's score matrix at a time (the per-head loop in `run_attention_heads`), so peak usage is just one `[s, s]` matrix, not 32. But the compute cost is still O(s^2) per head regardless.

---
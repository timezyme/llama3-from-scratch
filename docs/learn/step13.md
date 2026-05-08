---

## Step 13: SwiGLU FFN + Residual #2

**Files:** `src/inference_layer.cu:399-427` (orchestration), `kernel/swiglu.cu:76-86` (activation kernel)
**Where in the pipeline:** The second half of the decoder block. After attention + residual #1 updated `X`, this sub-block applies a second RMSNorm, runs the feed-forward network (FFN), and adds the result back with a second residual. This completes one decoder block.

### High-level picture

The FFN is where the model does per-token "thinking" — attention mixed information across tokens, and now the FFN processes each token independently through a wide hidden layer. The full sub-block:

```
X_norm = RMSNorm(X, post_attn_layernorm)     ← second gamma (line 401-404)
gate   = X_norm @ W_gate^T    [s, 14336]     ← expand to FFN dim (line 407)
up     = X_norm @ W_up^T      [s, 14336]     ← parallel expansion (line 409)
H      = SiLU(gate) * up      [s, 14336]     ← SwiGLU activation (line 417)
ffn_out = H @ W_down^T        [s, 4096]      ← compress back (line 420)
X = X + ffn_out                               ← residual #2 (line 426)
```

The data flows through an expand-activate-compress pattern: 4096 -> 14336 -> 4096. The wide hidden dimension (14336 = 3.5x the model dimension) gives the model more capacity to transform each token's representation.

### SwiGLU: gated activation

The clever part is **SwiGLU** — instead of one projection followed by a standard activation (like ReLU), there are *two* parallel projections (`gate` and `up`) followed by a gated combination:

```
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
H[i] = SiLU(gate[i]) * up[i]
```

`gate` controls *how much* signal passes through (via the sigmoid, which outputs 0-1). `up` provides *what* signal passes through. The gate can learn to selectively suppress or amplify different features — more expressive than a plain activation function.

The kernel (`swiglu.cu:83-85`) is three lines: load `g`, compute `g / (1 + exp(-g))`, multiply by `up[i]`. One thread per element, fully data-parallel.

### Three matmuls, the most expensive part

The FFN contains 3 of the 7 matmuls per layer. The gate and up projections expand from 4096 to 14336 — these are the largest matmuls in the decoder block because `d_ff = 14336` is much bigger than the attention dimensions. The down projection compresses back to 4096.

### The second RMSNorm

Line 401-404 applies RMSNorm with the `post_attention_layernorm` gamma — the **second** gamma for this layer, distinct from the `input_layernorm` gamma used before attention (step 8). This is the Part 2 pitfall about two separate gamma vectors per layer.

### Residual #2

Same kernel as residual #1: `X = X + ffn_out` (line 426). After this, `X` contains the output of the full decoder block — ready to enter the next layer (or the final norm if this is layer 31).

### Where we are in the decoder block

```
[X] ──→ RMSNorm#1 → QKV → RoPE → Attention → O proj → (+) → RMSNorm#2 → FFN → (+) → [X_out]
 |                                                        ↑                           ↑
 └──────────── residual #1 ──────────────────────────────┘                           |
 └──────────────────────────── residual #2 ──────────────────────────────────────────┘
```

Both residuals connect back to the *same* `X`. After residual #1, `X` was updated to `X + attn_out`. After residual #2, it becomes `X + attn_out + ffn_out`. The original signal flows through both skip connections.

---

**TA-style question:**

The FFN expands from 4096 to 14336 (gate and up) then compresses back to 4096 (down). That's two `[s, 4096] x [4096, 14336]` matmuls plus one `[s, 14336] x [14336, 4096]` matmul — three large matmuls per layer. How many total FLOPs do these three FFN matmuls account for relative to the four attention projection matmuls (Q, K, V, O) in the same layer, for a single token (s=1)?

**answer**

For s=1, each matmul is essentially a matrix-vector product. FLOPs = 2 x rows x cols (one multiply + one add per element).

**Attention projections (4 matmuls):**
- Q: 2 x 4096 x 4096 = 33.6M
- K: 2 x 4096 x 1024 = 8.4M
- V: 2 x 4096 x 1024 = 8.4M
- O: 2 x 4096 x 4096 = 33.6M
- **Total: 84M FLOPs**

**FFN projections (3 matmuls):**
- gate: 2 x 4096 x 14336 = 117.4M
- up: 2 x 4096 x 14336 = 117.4M
- down: 2 x 14336 x 4096 = 117.4M
- **Total: 352M FLOPs**

The FFN is **~4.2x more expensive** than the attention projections. The FFN dominates the per-layer compute budget — attention gets all the conceptual attention, but the FFN is where most of the FLOPs actually go. This is why the gate/up/down projections are the main target for optimization techniques like quantization and pruning in production systems.

---
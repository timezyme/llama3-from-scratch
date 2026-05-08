---

## Step 12: O Projection + Residual #1

**Files:** `src/inference_layer.cu:386-393` (orchestration), `kernel/residual.cu:59-66` (kernel)
**Where in the pipeline:** Closing the attention sub-block. After all 32 heads have produced their per-head outputs and been stitched back into a packed `[s, 4096]` tensor, two things happen: the output projection, then the first residual add.

### High-level picture

After attention, we have `O = concat(O_0, O_1, ..., O_31)` — the 32 head outputs concatenated back into `[s, 4096]`. Two operations finish the attention sub-block:

```
attn_out = O @ W_O^T          [s, 4096]  ← output projection (matmul)
X = X + attn_out               [s, 4096]  ← residual add
```

The output projection (line 387-390) is a matmul that mixes information across heads. Each head computed its own 128-dim output independently — the O projection is the first point where those heads can interact, since `W_O` is a full `[4096, 4096]` matrix.

The residual add (line 393) adds the attention result back to the **original input** `X` (before RMSNorm, before attention). This is a **skip connection**.

### Why skip connections matter

Without the residual add, the signal passes through RMSNorm -> Q/K/V -> attention -> O projection — a long chain of transformations. If any of those steps attenuates or distorts the signal, 32 layers of it would compound the damage. The skip connection provides a direct path: the original `X` flows straight through, and the attention result is *added* to it as a refinement.

Think of it as: "keep everything you already know, and add what attention learned."

### The kernel itself

`residual.cu:59-66` is the simplest kernel in the project: `a[i] += b[i]`. One thread per element, no shared memory, no reductions. Fully bandwidth-bound — it reads two floats and writes one, with a single FMA. The same kernel is reused for residual #2 after the FFN.

### Where this sits in the decoder block

```
[X] ──→ RMSNorm ──→ Q/K/V ──→ RoPE ──→ Attention ──→ O proj ──→ (+) ──→ ...
 |                                                                 ↑
 └─────────────── skip connection (residual #1) ──────────────────┘
```

The `X` that enters the residual add is the same `X` from *before* the first RMSNorm. Not the normalized version — the original activations. This is important: if you accidentally used `X_norm` instead of `X`, you'd lose the un-normalized signal path.

### TA-scrutiny items

No direct pitfalls here, but understand the data flow: `X` (pre-norm) is preserved through the entire attention sub-block and added to the result at the end. The pattern repeats identically for residual #2 after the FFN.

---

**TA-style question:**

The residual kernel does `a[i] += b[i]` — one add per element, but it reads 8 bytes (two floats) and writes 4 bytes from/to HBM. What is the arithmetic intensity of this kernel (FLOPs per byte), and is it memory-bound or compute-bound? How does this compare to the matmul kernel from step 7?

**answer**

1 FLOP (one addition) per 12 bytes (read 2 floats = 8 bytes, write 1 float = 4 bytes). Arithmetic intensity = **1/12 ≈ 0.08 FLOPs/byte**. Extremely memory-bound — the GPU's compute units are almost entirely idle waiting for HBM.

Compare to the tiled matmul: ~256 FLOPs/byte. That's roughly **3,000x** more compute-dense. The matmul keeps CUDA cores busy; the residual add just shuffles bytes.

This is why nobody optimizes the residual kernel — it's inherently memory-bound and there's no way to add reuse. The only optimization would be **fusing** it into the preceding matmul (write `attn_out + X` instead of `attn_out`, then skip the separate add), saving one HBM round-trip. This project doesn't do that — the separate kernel is cleaner and the cost is small.

---


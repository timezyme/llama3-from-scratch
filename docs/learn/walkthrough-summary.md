---

## Walkthrough Summary: TA-Scrutiny Cheat Sheet

### The pipeline in one sentence

Prompt -> chat template -> BPE encode -> embed [s, 4096] -> 32 decoder layers -> final RMSNorm -> last-token extract -> lm_head [128256] -> argmax -> text.

### M1 Mandatory (matmul kernel)

| Optimization            | What it does                                   | Where               |
| ----------------------- | ---------------------------------------------- | ------------------- |
| **Tiling**              | 128x128 output tiles, 16-wide K slabs          | `matmul.cu:72-74`   |
| **Shared-memory reuse** | Load from HBM once, reuse 128x from smA/smB    | `matmul.cu:165-166` |
| **Coalesced access**    | Consecutive threads read consecutive addresses | `matmul.cu:194-227` |
| Double-buffering        | Overlap load + compute via ping-pong buffers   | `matmul.cu:244-322` |

### Part 2 Pitfalls (expect TA questions on all of these)

| Pitfall                      | Correct behavior                                    | Where                        |
| ---------------------------- | --------------------------------------------------- | ---------------------------- |
| **Stable softmax**           | Subtract row max before exp() — otherwise NaN       | `attention.cu:155-207`       |
| **RoPE pairing**             | (i, i+h_d/2) NOT (2i, 2i+1)                         | `rope.cu:123-124`            |
| **RoPE base**                | 500,000 NOT 10,000                                  | `rope.cu:186`                |
| **Two RMSNorm gammas/layer** | input_layernorm + post_attention_layernorm          | `model_weights.cpp:124-129`  |
| **Gamma not skipped**        | y = (x/rms) * gamma — the multiply is required      | `rmsnorm.cu:145`             |
| **Transposed weights**       | Loaded as [in, out] — matmul does X @ W_stored      | `model_weights.cpp:232-239`  |
| **Full causal mask**         | Every (p, q) with q > p, not just diagonal/last row | `attention.cu:130-138`       |
| **Last-token-only lm_head**  | Project only row s-1, not all s rows                | `inference_layer.cu:450-460` |
| **lm_head != embed_tokens**  | tie_word_embeddings=false in Instruct checkpoint    | `model_weights.cpp:36-40`    |

### Bonus features shipped

| Feature                   | Key insight for TA                                              | Where                       |
| ------------------------- | --------------------------------------------------------------- | --------------------------- |
| **KV cache**              | O(T) decode instead of O(T^2). Only new K gets RoPE.            | `kv_cache.h/cu`             |
| **B>1 batching**          | Same latency, more throughput — GPU underutilized at B=1 decode | `inference_loop.cu:116-234` |
| **Resident BF16 weights** | 16 GB fits L4 VRAM; eliminates per-step PCIe transfers          | `device_weights.cu`         |

### Key numbers to know

| Quantity              | Value                    |
| --------------------- | ------------------------ |
| Embedding dim (d)     | 4,096                    |
| Query heads (h)       | 32                       |
| KV heads (h_k)        | 8                        |
| Head dim (h_d)        | 128                      |
| FFN hidden dim (d_ff) | 14,336                   |
| Vocab size (V)        | 128,256                  |
| Layers                | 32                       |
| RoPE base             | 500,000                  |
| RMSNorm epsilon       | 1e-5                     |
| FP32 model size       | ~32 GB (won't fit L4)    |
| BF16 model size       | ~16 GB (fits L4's 24 GB) |

All 18 steps complete. The walkthrough journal is in `docs/walkthrough-progress.md` and the diagram in `docs/diagrams/walkthrough.png`.
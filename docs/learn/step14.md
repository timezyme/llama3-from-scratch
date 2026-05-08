---

## Step 14: 32-Layer Loop

**File:** `src/inference_layer.cu:261-438`
**Where in the pipeline:** This is the outermost structure of the GPU compute. Steps 8-13 described one decoder block — step 14 is the `for` loop that runs it 32 times with different weights each time.

### High-level picture

Line 261 is a simple loop:

```cpp
for (int layer = 0; layer < NUM_LAYERS; ++layer) {
```

Each iteration runs the complete decoder block from steps 8-13: RMSNorm #1 -> Q/K/V -> RoPE -> attention -> O proj -> residual #1 -> RMSNorm #2 -> FFN -> residual #2. The output `X` from layer N becomes the input to layer N+1. Same shape (`[s, 4096]`) in and out — 32 layers of refinement on the same representation.

What changes between layers: **the weights**. Each layer has its own 9 tensors (2 norms, 4 attention projections, 3 FFN projections). The scratch buffers (`d_X`, `d_Xnorm`, `d_Q`, `d_attn`, `d_gate`, `d_up`, `d_ffn`) are allocated once before the loop (lines 219-236) and reused across all 32 iterations.

### Two weight delivery modes

The loop body branches on `resident_weights != nullptr`:

**Streaming path** (lines 271-304, 435-438): Load this layer's FP32 weights from disk into CPU RAM (`weights.load_layer(layer)`), upload them to GPU via `cudaMemcpy`, run the decoder block, then `weights.unload_layer(layer)` to free the CPU copy. Only one layer's weights live in memory at a time. Slow (~seconds per layer due to PCIe transfers), but works within any memory budget.

**Resident path** (line 270): `resident_weights->load_layer(layer)` returns pointers to BF16 weights already sitting in GPU VRAM. No copying, no disk I/O — just use them. All 32 layers fit simultaneously in the L4's 24 GB VRAM (~16 GB total in BF16).

### Buffer reuse strategy

The forward_step function allocates about 15 device buffers at the top (lines 219-250) and frees them all at the bottom (lines 463-469). Within the 32-iteration loop, every buffer is overwritten each iteration — no accumulation across layers except `d_X` itself, which carries the evolving hidden state.

The `AttentionScratch` struct (line 240-250) holds 5 per-head temporary buffers (Q_i, K_g^T, V_g, S, O_i) that are reused across all 32 layers x 32 heads = 1,024 attention head computations per forward pass.

### KV cache interaction

Each layer writes its K and V projections into the KV cache at the layer's designated slot. After the loop exits, `cache.advance(q_seq)` (line 442) updates the cache length so the next decode step knows how many past tokens are stored. This is the mechanism that connects one forward pass to the next during autoregressive generation.

### TA-scrutiny items

- **Per-layer weights**: Each layer has distinct weight tensors. A common bug is accidentally using layer 0's weights for all 32 layers.
- **Two RMSNorm gammas per layer**: The loop loads `input_layernorm` (line 280-281) and `post_attn_layernorm` (line 395-398) separately. Swapping them is a silent correctness bug.
- **Buffer reuse**: Scratch buffers are allocated once and overwritten each iteration. This is efficient but means you can't inspect layer 5's intermediate Q after layer 6 has already overwritten the buffer.

---

**TA-style question:**

On the streaming path, each layer requires uploading ~450 MB of FP32 weights over PCIe (9 tensors totaling ~112M parameters x 4 bytes). For 32 layers, that's ~14.4 GB of PCIe transfers per forward pass. On the resident BF16 path, the upload is zero during the forward pass (paid once at startup). For a prompt that generates 8 tokens — requiring 8 forward passes — how much total PCIe weight traffic does each path incur, and why does this gap make multi-token generation impractical on the streaming path?

**answer**

**Streaming path:** ~14.4 GB per forward pass x 8 passes = **~115 GB** of PCIe transfers just for weights. At PCIe Gen4 x16 bandwidth (~25 GB/s), that's ~4.6 seconds of pure transfer time — plus the disk-to-CPU loading before each upload. Each token takes seconds.

**Resident BF16 path:** ~16 GB uploaded once at startup, then **zero** weight traffic for all 8 forward passes. The weights are already in VRAM. Per-token latency is bounded by GPU compute, not PCIe.

The gap: streaming re-transfers the entire model for every token generated. The cost is linear in both layers (32) and tokens (T). For T=1 it's tolerable — you pay once and get one answer. For T=8 you pay 8x. For T=32 it's completely impractical. That's exactly why the multi-token path exists: pay the one-time BF16 upload cost (~165s cold on L4) and amortize it across all subsequent decode steps.

---

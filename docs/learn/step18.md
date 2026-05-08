---

## Step 18: B>1 Batched Generation (Bonus Feature)

**File:** `src/inference_loop.cu:116-234`
**Where in the pipeline:** This is path 3 from step 1 — multiple prompts processed simultaneously in a single forward pass.

### High-level picture

Instead of running two prompts through two separate forward passes, batched generation stacks them into one wider pass. Where B=1 processes `[s, 4096]` activations, B=2 processes `[2*s, 4096]`. The GPU does the same matmuls but on a taller matrix — and since the GPU is designed to saturate on large matrices, you get nearly 2x throughput for free.

### How it works

The orchestrator at line 116 follows the same prefill → decode loop as the B=1 path, with three key differences:

**1. All prompts must be the same token length** (line 140-141). `validate_equal_lengths` enforces this up front. Mixed-length batching would require padding and masking inside the forward pass — out of scope for this project. The CLI in `main.cpp:169-183` pre-checks this before reaching the inference code.

**2. KV cache is sized for the batch** (line 152). `KVCache cache(S_MAX, batch)` allocates B separate K/V regions per layer. Each batch slot has its own `[max_len, 1024]` buffer. This is why K/V projections in step 9 run per-batch-slot — each slot writes to its own cache region.

**3. Finished slots keep feeding EOT** (lines 194-201). When one prompt finishes (emits EOT) before others, it keeps advancing through the forward pass with EOT tokens. The dimensions stay constant so the batch shape and KV cache layout don't need resharding. The generated tokens for finished slots are simply ignored. This wastes a small amount of compute but avoids the complexity of dynamic batching.

### The decode loop

Lines 186-226 — same structure as single-prompt decode:

```
for step in 1..max_new_tokens:
    if all slots done: break
    embed B tokens (one per slot)              ← get_embeddings_batched
    forward_step(q_seq=1, batch=B)             ← one forward pass for all B slots
    for each active slot:
        lm_head → argmax → check EOT
```

Each `forward_step` call processes all B prompts in lockstep. The matmul kernels see `rows = B * q_seq` — for decode that's `B * 1 = B` rows. The matmul tiles fill more completely with B>1, improving GPU utilization.

### Why batching helps performance

A single-token decode (B=1, q_seq=1) produces tiny matmuls: `[1, 4096] × [4096, 4096]`. The GPU has thousands of cores but only 4,096 multiply-adds to do — most cores sit idle. With B=4, the matmul is `[4, 4096] × [4096, 4096]` — 4x more work, but the GPU was already underutilized, so it finishes in nearly the same wall time. That's the throughput win: same latency, more prompts answered.

### Limitations

- Equal-length prompts only (no padding/masking)
- All slots advance together — fast prompts (short answers) waste compute waiting for slow ones
- KV cache scales linearly with B (step 17's scaling table applies)

### TA-scrutiny items

This is a bonus feature (up to 5% extra credit). The key thing to articulate is *why* batching improves throughput: GPU underutilization at B=1 decode. The matmul hardware is designed for large matrices; batching gives it more rows to fill.

---

**TA-style question (final one):**

The batched path requires equal-length prompts. If you wanted to support variable-length prompts in the same batch, what two problems would you need to solve in the forward pass, and which one is harder?

**answer**

**Problem 1: Padding + masking.** Shorter prompts need to be padded to the longest prompt's length so the matrices are rectangular. But padding tokens shouldn't contribute to attention scores or RMSNorm statistics. You'd need a per-slot length mask that zeros out padding in the causal mask, excludes padding from the RMSNorm sum-of-squares, and ensures the "last token" extract (step 15) picks the right row per slot — not the padded row.

**Problem 2: Per-slot attention windows.** In the causal mask, each slot's valid positions are different. Slot 0 (10 tokens) should attend to positions 0-9; slot 1 (15 tokens) should attend to 0-14. The causal mask becomes a per-slot mask rather than a single shared `col > row` check. The KV cache also has different valid ranges per slot — attention for slot 0 must not read slot 1's cached K/V values (which live at different positions in the buffer).

**The harder one is attention masking.** Padding the embeddings and skipping padding in RMSNorm/residual is mechanical — a few `if` checks. But fixing the attention mask requires changing the mask kernel to be batch-aware, and more importantly, the KV cache read window must be per-slot. Every kernel that touches the score matrix or reads from the cache needs to know each slot's true length. This is why production systems (vLLM, TensorRT-LLM) treat variable-length batching as a first-class design problem rather than a bolt-on.

---
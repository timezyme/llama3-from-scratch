# Walkthrough Progress

Live journal for the TA code-review prep walkthrough of this Llama 3 8B inference codebase.

## Current step

**COMPLETE — All 18 steps covered.**

## Outline (18 steps)

| # | Topic | Primary file(s) | TA hot spots |
|---|---|---|---|
| 1 | CLI entry & 4-path dispatch | `main.cpp` | — |
| 2 | Chat template | `src/inference_chat.cu` | — |
| 3 | BPE tokenizer (encode) | `src/tokenizer_bpe.cpp` | special tokens (BOS, header_id) |
| 4 | Dump format & loader | `tools/dumper.py`, `src/loader.cpp`, `include/milifloat.h` | BF16/FP16/FP32 decode |
| 5 | ModelWeights, transpose-at-load | `src/model_weights.cpp` | weights stored transposed (W^T) |
| 6 | Embedding lookup | `src/model_weights.cpp::get_embeddings` | embed table location (CPU vs GPU) |
| 7 | Matmul kernel | `kernel/matmul.cu` | **M1 mandatory: tiling, shared-mem reuse, coalesced HBM**; bonus: tensor cores, vectorization |
| 8 | RMSNorm kernel | `kernel/rmsnorm.cu` | gamma scaling, eps inside sqrt, two RMSNorm gammas/layer |
| 9 | Q/K/V projections | `src/inference_layer.cu` | weight transpose — XW^T not XW |
| 10 | RoPE | `kernel/rope.cu` | **rotate_half pairing (i, i+hd/2) NOT (even, odd)**; **base 500000 NOT 10000** |
| 11 | GQA attention + softmax | `kernel/attention.cu` | **stable softmax (subtract row max)**; full causal mask; GQA group index g = i / (h/h_k) |
| 12 | O proj + residual #1 | `kernel/residual.cu` | — |
| 13 | SwiGLU FFN + residual #2 | `kernel/swiglu.cu` | — |
| 14 | 32-layer loop | `src/inference_loop.cu` | per-layer weights, two RMSNorm gammas |
| 15 | Final RMSNorm + last-token + lm_head | `src/inference_layer.cu` | **lm_head NOT tied to embed_tokens in Llama-3-8B-Instruct**; **last-token-only projection** |
| 16 | Argmax + decode | `src/tokenizer_bpe.cpp::decode` | — |
| 17 | KV cache | `src/kv_cache.cu` | scaling: O(T) vs O(T^2) FLOPs |
| 18 | B>1 batched generation | `src/inference_loop.cu` batched path | — |

## What we have covered

- **Step 1 — `main.cpp` CLI entry & 4-path dispatch.** argv parsing, four paths (single-token FP32-streamed / multi-token BF16-resident / batched B>1 / interactive REPL). Split exists because FP32 weights (~32 GB) don't fit in L4's 24 GB VRAM; BF16 (~16 GB) does. `ModelWeights` = CPU-side (embedding lookup); `DeviceModelWeights` = GPU-resident BF16 (layer compute).
- **Step 2 — `apply_chat_template` (`src/inference_chat.cu:38-55`).** Wraps raw prompt in Llama 3 Instruct chat template (BOS, role headers, EOT, trailing open assistant header). M1 grader bypasses this; end-to-end inference requires it.
- **Step 3 — BPE encode (`src/tokenizer_bpe.cpp:251-307`).** Two-phase: special-token peeling (longest match first), then greedy lowest-rank pair-merge on plain-text chunks. encode() does NOT prepend BOS — callers do. O(n^2) per chunk, acceptable because tokenization is trivial vs the GPU forward pass.
- **Step 4 — Dump format & loader.** Python dumper writes 280-byte header + raw payload per tensor. C++ loader has two modes: FP32-widen (streaming path) and raw-BF16 (resident path). BF16->FP32 is a single left-shift; FP16->FP32 requires exponent rebiasing. Embedding table kept as raw blob, decoded per-row on demand to avoid ~2 GB FP32 peak.
- **Step 5 — ModelWeights & transpose-at-load.** 9 tensors per layer (2 norms, 4 attn projections, 3 FFN projections) + 3 global. All 2D weights transposed at load from [out, in] to [in, out] so matmul is X @ W_stored with no runtime transpose. K/V projections are [4096, 1024] not [4096, 4096] because GQA uses 8 KV heads vs 32 query heads. Two distinct RMSNorm gammas per layer (Part 2 pitfall).
- **Step 6 — Embedding lookup.** Token ID -> row index into 128,256 x 4,096 table. Lazy per-row decode from cached BF16 blob. CPU-side only; resulting [s, 4096] FP32 matrix transferred to GPU. ~0.01% of table used per prompt — keeping full table on GPU wastes ~1 GB VRAM for cold data.
- **Step 7 — Matmul kernel (M1 MANDATORY).** Tiled GEMM with 128x128 output tiles, 16-wide K slabs. Three required optimizations: (1) tiling — 128x reuse per HBM load, (2) shared-memory staging — smA/smB hold tiles on-chip, (3) coalesced access — consecutive threads read consecutive addresses. Double-buffered (ping-pong two smA/smB copies to overlap load+compute). Per-thread 8x8 register accumulation = 64 FMAs from 16 shared reads. BF16-weight variant identical except loads widen 4 BF16 values inline. Three entry points: gpu_matmul (M1 grader, H2D/D2H copies), gpu_matmul_device (FP32 forward), gpu_matmul_device_bf16_weight (BF16 forward).
- **Step 8 — RMSNorm kernel.** One block per row, 256 threads. Two passes: (1) tree-reduce sum-of-squares in shared memory, (2) divide by rms and multiply by gamma. Three TA pitfalls: epsilon INSIDE sqrt, gamma not skipped, two distinct gammas per layer. Runs 65 times per forward pass (2 per layer + 1 final).
- **Step 9 — Q/K/V projections.** Three matmuls on X_norm using pre-transposed weights. Q = [s, 4096] (32 heads), K = [s, 1024] (8 heads), V = [s, 1024] (8 heads). Q computed for whole batch at once; K/V per batch slot because they write directly into KV cache at per-slot addresses. First appearance of KV cache in compute flow.
- **Step 10 — RoPE.** Rotates Q and K (not V) in-place using precomputed cos/sin tables. Pairs (i, i+64) not (2i, 2i+1). Base 500,000 not 10,000. Low-index pairs rotate fast (local position); high-index pairs rotate slowly (long-range). Larger base stretches frequencies for longer context support.
- **Step 11 — GQA attention + softmax.** Per-head loop: gather Q_i/K_g^T/V_g -> matmul S=Q*K^T -> scale(1/sqrt(128)) -> causal mask (col>row -> -1e6) -> stable softmax (subtract row max before exp) -> matmul O=S*V -> scatter. GQA: g = hi/4. Score matrix is O(s^2) per head; at s=16384 the 32 matrices exceed model weight memory. Implementation materializes one head at a time to limit peak memory.
- **Step 12 — O proj + residual #1.** O projection (matmul, [s, 4096]) mixes information across heads. Then X = X_original + attn_out — skip connection using pre-norm X, not X_norm. Residual kernel is simplest in project: a[i] += b[i], one thread/element, bandwidth-bound.
- **Step 13 — SwiGLU FFN + residual #2.** Second RMSNorm (post_attn_layernorm gamma), then expand-activate-compress: gate/up matmuls (4096->14336), SwiGLU activation (SiLU(gate)*up), down matmul (14336->4096), residual #2 (X = X + ffn_out). FFN is ~4.2x more FLOPs than the attention projections — dominant compute cost per layer.
- **Step 14 — 32-layer loop.** `for (layer = 0..31)`: each iteration runs the full decoder block (steps 8-13) with that layer's weights. Scratch buffers allocated once and reused across all iterations. Streaming path: load/upload/unload per layer (~14.4 GB PCIe per pass). Resident path: zero weight traffic (BF16 weights pre-loaded). Multi-token generation impractical on streaming path (8 tokens = ~115 GB PCIe traffic vs zero on resident).
- **Step 15 — Final RMSNorm + last-token + lm_head.** Final norm (65th RMSNorm, third gamma). Extract only last token's [4096] vector. lm_head projects to [128256] logits on CPU. lm_head is NOT the embedding table (tie_word_embeddings=false in Instruct checkpoint, diff up to 0.345). Evidence: config.json flag + verify_reference.py numerical comparison.
- **Step 16 — Argmax + decode.** `std::max_element` over 128,256 logits → token ID. `BPETokenizer::decode` looks up ID in `id2tok`, skips special tokens. For multi-token: feed token back into embed → forward_step(q_seq=1) → lm_head → argmax loop. Stops on EOT (128009) or max_new_tokens.
- **Step 17 — KV cache (bonus).** Pre-allocated GPU buffers: [batch, max_len, 1024] per layer, per K and V. 256 MiB at batch=1, S_MAX=1024. Prefill writes rows [0, s₀), each decode step appends 1 row. Converts O(T²) recomputation to O(T) cached reads. Only new K rows get RoPE. Scales linearly in batch × seq_len; at batch=4, seq=4096, cache is 4 GiB.
- **Step 18 — B>1 batched generation (bonus).** Lockstep multi-prompt: all prompts same token length, stacked into one wider forward pass. Finished slots feed EOT to maintain batch shape. Throughput win: GPU underutilized at B=1 decode (tiny matmuls), batching gives more rows for same wall time. Limitation: equal-length only; variable-length needs per-slot attention masking (hard).

## What is left

None — walkthrough complete.

## Deep-dives requested + TA-style Q&A log

### Step 1
**Q:** Why does the multi-token path need both `ModelWeights` and `DeviceModelWeights`?
**A:** Embedding lookup is a CPU-side table lookup in `ModelWeights`. `DeviceModelWeights` holds the 32 layers' projection/FFN/norm weights on GPU in BF16 for CUDA kernels. Different stages need different objects.

### Step 2
**Q:** What would happen if you accidentally closed the assistant turn with EOT before feeding the sequence to the model?
**A:** The model would see a complete conversation with an empty assistant reply — no incomplete turn to continue. It would likely start a new turn or produce incoherent output instead of answering the question.

### Step 3
**Q:** Why is the O(n^2) BPE merge loop acceptable, and when would it become a problem?
**A:** Prompts are capped at 1,000 tokens, so n^2 is microseconds — trivial vs the seconds-long GPU forward pass. It would matter for very long documents or high-throughput servers; production tokenizers use a priority queue for O(n log n).

### Step 4
**Q:** Why not decode the full embedding table to FP32 up-front? Two reasons.
**A:** (1) Obvious: only ~10 rows needed per prompt, not 128,256. (2) Subtle: full FP32 expansion = 128,256 x 4,096 x 4 = ~2 GB. Combined with the ~1 GB raw blob during loading, that's 3 GB peak just for embeddings — significant on a memory-constrained machine that also needs to load 32 layers of weights.

### Step 5
**Q:** K projection is [1024, 4096] in the dump vs Q's [4096, 4096]. Why, and what is 1024?
**A:** 1024 = 8 KV heads x 128 dim/head. GQA gives Q 32 heads but K/V only 8 (every 4 query heads share one K/V head). K/V projections are 4x smaller than Q — saves weight storage and KV cache memory during autoregressive decoding.

**Deep-dive requested:** Student asked for confirmation that decoder layers and kernels (matmul, etc.) will be covered in detail in later steps. Answer: yes — steps 7-14 walk through each kernel and the decoder block piece by piece. Steps 5-6 only introduce the weight shapes that feed those kernels.

### Step 6
**Q:** What fraction of the embedding table is used per prompt, and is GPU-caching it worthwhile?
**A:** ~15 rows out of 128,256 = ~0.01%. Full table is ~1 GB BF16 / ~2 GB FP32 of mostly cold data. Transferring the ~240 KB of actual embeddings over PCIe takes microseconds — far better than wasting 1+ GB of VRAM.

### Step 7
**Q:** Without shared memory, how many times would each element of A be re-read from HBM? What does this say about arithmetic intensity?
**A:** 128 times (once per output column in the tile). With tiling, each HBM load feeds 128 reuses from shared memory — 128x reduction in HBM traffic. Arithmetic intensity goes from ~2 FLOPs/byte (naive) to ~256 FLOPs/byte (tiled), shifting the kernel from memory-bound to compute-bound.

### Step 8
**Q:** The kernel reads each row from HBM twice (pass 1 + pass 2). How many bytes per row, and why not cache it in shared memory?
**A:** 3 x 4,096 x 4 = 49,152 bytes (two reads + one write). Caching the full row in shared memory (16 KB) would halve read traffic, but it tightens shared-memory budget and can reduce occupancy — fewer blocks per SM, less latency hiding. For this simple two-pass kernel the second read is fast enough that the tradeoff isn't worth it.

### Step 9
**Q:** Why can't K and V be computed as a single batched matmul like Q?
**A:** Q writes to a contiguous temporary buffer, so one matmul fills it. K/V write directly into per-batch-slot KV cache regions at different base pointers — not contiguous memory. A single matmul can only write to one contiguous output, so one call per batch slot is needed to target each slot's cache region.

### Step 10
**Q:** theta values decrease exponentially across pair indices — which pairs rotate fast/slow and why is that useful?
**A:** Low-index pairs (theta near 1.0) rotate fast — encode fine-grained local position. High-index pairs (theta near 0) rotate slowly — encode coarse long-range position. Model gets multi-scale position signals simultaneously: some heads use fast dims for grammar, others use slow dims for long-range context. Base 500,000 (vs 10,000) stretches slow dims even further for longer sequence support.

### Step 11
**Q:** For s=512, how much GPU memory do all 32 score matrices need, and at what s does it exceed model weights (~16 GB)?
**A:** s=512: 512x512x4 = 1 MB per head, x32 = 32 MB. Scales O(s^2). At s≈16,384 the 32 matrices hit ~32 GB, exceeding the ~16 GB model weights. This is why long-context is hard and FlashAttention exists (tiles softmax without materializing full s x s). This implementation only materializes one head's S at a time, so peak is one [s,s] matrix, not 32.

### Step 13
**Q:** How many FLOPs do the 3 FFN matmuls account for relative to the 4 attention projection matmuls, for s=1?
**A:** Attention: Q(33.6M) + K(8.4M) + V(8.4M) + O(33.6M) = 84M FLOPs. FFN: gate(117.4M) + up(117.4M) + down(117.4M) = 352M FLOPs. FFN is ~4.2x more expensive — dominant compute cost per layer.

### Step 14
**Q:** How much total PCIe weight traffic does each path incur for 8-token generation?
**A:** Streaming: ~14.4 GB/pass x 8 = ~115 GB. At PCIe Gen4 ~25 GB/s, that's ~4.6s pure transfer per token. Resident BF16: ~16 GB once at startup, zero per pass. Streaming re-transfers the entire model per token — linear in T, impractical for multi-token generation.

### Step 15
**Q:** The assignment says lm_head shares weights with the embedding table. Why doesn't your implementation do that, and what evidence proves they're different?
**A:** (1) `config.json` has `tie_word_embeddings: false` — the authoritative flag for this Instruct checkpoint. (2) The two matrices differ by up to 0.345 in absolute value, verified by `tools/verify_reference.py`. The base Llama 3 model ties them; the Instruct fine-tune untied and trained lm_head separately.

### Step 16
**Q:** Why might greedy decoding not produce the best overall response?
**A:** Greedy is locally optimal but not globally. Best first token may lead to worse overall sentence. Alternatives: beam search (parallel partial sequences), temperature sampling (soften/sharpen distribution), top-k (sample from top k candidates), top-p/nucleus (adaptive candidate set by cumulative probability). Greedy is correct and sufficient for this project.

### Step 17
**Q:** How does KV cache scale, and when does it become a VRAM bottleneck on L4?
**A:** 256 KiB × batch × seq_len. With ~8 GB free after weights: batch=1 hits wall at ~32K seq_len; batch=4 at ~8K; batch=8 at ~4K. Production systems use PagedAttention (vLLM) to manage KV memory like virtual memory pages.

### Step 18
**Q:** What two problems would variable-length batching require, and which is harder?
**A:** (1) Padding + masking: pad shorter prompts, exclude padding from norms/attention/last-token extract. (2) Per-slot attention windows: causal mask and KV cache reads must be batch-aware with per-slot valid ranges. Attention masking is the harder one — every kernel touching the score matrix needs per-slot length info, vs mechanical `if` checks for padding.

## Next planned step

**Walkthrough complete.**

# Code Review Prep

One-sitting prep doc. Read straight through the morning of the review.

## 1. Demo Runbook (do this first, in this order)

### Pre-review setup (\~5 min before the TA joins)

Two scripts handle the full lifecycle — no raw gcloud commands needed.

```bash
# ~5 min before the TA joins, on your Mac:
./scripts/demo-start.sh           # default --max-tokens 32 (good for full sentences)
# or:
./scripts/demo-start.sh 64        # longer answers if needed

# Wait for: "[interactive] ready. max-tokens per prompt: 32."
# Leave this terminal up. Type prompts at the > marker during the demo.

# After the demo:
./scripts/demo-stop.sh
```

### Live with the TA

```bash
# 1. Show the repo is clean and the test API is untouched. (~5s)
git status                                 # working tree clean
git log --oneline -3                       # recent feature commits
git diff HEAD -- tests/test.cpp tests/test_api.h   # zero output (read-only)

# 2. M1 grading suite (FP32 path). Quick sanity check. (~10s)
for i in 1 2 3 4 5 6 7; do ./bin/tests $i | tail -1; done   # 7/7 PASSED

# 3. Conclusive end-to-end demo via the running REPL. (~3-5s per prompt)
# Switch to the interactive terminal you started earlier, type:
> What is the capital of California?
# Expect: "The capital of California is Sacramento."
> What is the capital of France?
# Expect: a Paris-mentioning continuation
> What is the capital of Japan?
# Expect: a Tokyo-mentioning continuation

# 4. Bonus credit (only if TA asks).
./bin/tests_m2m3 batched_b2_distinct_parity      # TODO #2 (+5%)  ~3.5 min
./tools/test_l4.sh --perf                        # TODO #1 (+5%)  ~4 min
```

**Talking-points while the REPL responds:** each prompt exercises every required milestone in one command — tokenizer → embedding lookup → 32 transformer layers (RMSNorm, Q/K/V matmul, RoPE, GQA, residual, SwiGLU FFN, residual) → final RMSNorm → lm_head argmax → BPE detokenize. The reason responses are fast (\~3s) is that the 14.5 GB BF16 weights are already resident on GPU; we paid that load cost (\~165s) once at REPL startup.

### Fallback if the REPL terminal dies (e.g., spot preemption)

```bash
./scripts/demo-start.sh 32   # full lifecycle restart; pays the ~3-min resident load again
```

---

## 2. Design Decisions (the "why we chose X" register)

For each, the order is **what we did → why → what we rejected**.

### 2a. Two paths: FP32 streaming vs BF16 resident

**Did:** Two inference paths. Single-token / M1 grading goes through layer-streaming **FP32** (load BF16 from disk, expand to FP32, upload to GPU, run, free). Multi-token (`--max-tokens N>1`) goes through **resident BF16** (upload all 14.5 GB once, BF16-input matmul, FP32 accumulate).

**Why:** FP32 weights would need 32 GB; L4 has 24 GB — resident FP32 is physically impossible. BF16 fits with \~6 GB headroom for KV cache + activations. This is the only way to claim TODO #1's +5% bonus on this hardware.

**Rejected:** (a) FP32 everywhere → blocks the bonus; (b) BF16 everywhere → the M1 graded matmul kernel would have BF16-precision drift; (c) FP16 instead of BF16 → FP16 has the same 16-bit width but a smaller exponent range, so large activations risk overflow. BF16 has FP32's exponent range with reduced mantissa.

### 2b. Tiled GEMM with double buffering

**Did:** `kernel/matmul.cu` uses 128×128 block tiles, 8×8 thread tiles, 16-wide K depth, double-buffered shared memory, `float4` vectorized loads for B, +1 padding to avoid shared-memory bank conflicts.

**Why:** Spec mandates tiling + shared memory + coalesced loads. Double buffering hides the global-memory latency of the next K-tile fetch behind the current tile's compute. Vectorized loads quadruple per-thread global-memory throughput. Bank-padding avoids 32-way conflicts on 128-wide rows.

**Rejected:** (a) 64×64 tiles → too few outputs per block, occupancy-bound; (b) Tensor-core WMMA → on the bonus list as TODO #3, not landed; (c) cuBLAS → spec forbids external compute libraries.

### 2c. Per-head attention orchestrated on the host

**Did:** `run_attention_heads` in `src/inference.cu` copies Q/K/V to host once per layer, then loops over 32 heads on the CPU dispatching `gpu_matmul_device` + `gpu_scale` + `gpu_causal_mask` + `gpu_softmax` per head.

**Why:** Simple, correct, and easy to validate against `reference.py` per head. Decouples attention from a custom fused kernel.

**Rejected:** A fused on-device dispatch (planned as TODO #8). Reason rejected for now: extra \~200 LoC and a new kernel for \~300ms/forward saving — worth it for production, not for first-correct.

### 2d. lm_head computed on CPU

**Did:** `compute_lm_head_logits` runs a CPU dot-product over the [128256 × 4096] vocabulary projection.

**Why:** CPU runs once at the end of each generation step — total wall-time impact is small relative to the 32-layer GPU forward. Trivial to verify against PyTorch's `torch.matmul`.

**Rejected:** GPU GEMV (TODO #7). About 525 MFLOPs CPU-side; a GPU GEMV would shave \~50ms off cold inference but adds a kernel and a pre-transpose at load.

### 2e. Transpose weights at load time

**Did:** `src/model_weights.cpp` transposes Q/K/V/O/gate/up/down weights from HuggingFace's `[out, in]` layout to `[in, out]` once, when loading.

**Why:** Llama's matmul is `X @ W^T`. Doing the transpose once at load lets every forward pass use the GEMM in its natural row-major form, avoiding either (a) a runtime transpose each layer or (b) a transpose-aware kernel. Trades load time for forward-pass speed.

**Rejected:** Transposing on the GPU per layer — wastes bandwidth for an operation that's free at load time.

### 2f. Greedy argmax decode (no sampling)

**Did:** `std::max_element` over the logits row. No temperature, no top-k, no top-p.

**Why:** Spec `llm_part2.md:163` mandates argmax; sampling is an explicit bonus (TODO #4). Argmax is deterministic, which makes parity-vs-`reference.py` testing trivial.

**Rejected:** Stochastic sampling — would have required a separate seeded RNG and a non-trivial parity test (compare distribution shape, not exact tokens).

### 2g. Llama 3 chat template applied automatically

**Did:** `apply_chat_template` in `src/inference.cu` wraps the user prompt with `<BOS>`, `<|start_header_id|>user<|end_header_id|>`, the prompt, `<|eot_id|>`, and the assistant header before encoding.

**Why:** Llama 3 *Instruct* was trained with this template; raw prompts produce garbage. The conclusive demo "What is the capital of California?" → "Sacramento." only works because of this.

**Rejected:** Letting users supply their own template — added complexity, no grade benefit.

### 2h. Per-batch K/V slabs in the cache (B>1 path)

**Did:** `KVCache` allocates `[B, max_len, kv_dim]`; per-batch slab base via `k_batch(layer, b)`. K/V projection matmul fans out into B separate launches per layer.

**Why:** Layout makes the host attention loop's per-batch slab `cudaMemcpy` contiguous (one copy per batch slot per layer, no strided gathers). The cost is `B` extra K/V matmul launches per layer — small on B=2.

**Rejected:** Layout `[max_len, B, kv_dim]` — would let one matmul interleave-write all batches' K/V in one launch but force batch-strided gathers in attention. Picked the option that simplifies the more complex code path.

### 2i. RMSNorm/softmax kernels left batch-agnostic

**Did:** When B>1 was added, RMSNorm and softmax were *not* modified. We pass them `rows = B*q_seq` and they work.

**Why:** Both kernels are purely row-local — no cross-row dependency. Treating B as another row dimension is a layout reinterpretation, not a semantic change.

**Rejected:** Adding an explicit `int batch` arg — verbose for zero correctness benefit.

---

## 3. Anticipated TA Questions (with prepared answers)

**Q1: "Walk me through `matmul.cu` line by line."**
A: Block tile constants (line 21–32), shared-memory geometry with +1 padding (line 36+), kernel computes its (block_x, block_y) tile of C, double-buffer index toggling on lines that load tile_n+1 while computing tile_n, register-tile accumulator unrolled over TM×TN. End with the bank-conflict-aware shared-memory write and the coalesced final write to global C.

**Q2: "What's the arithmetic intensity of your matmul on a 4096×4096×4096 problem?"**
A: 2·M·N·K FLOPs / (M·K + K·N + M·N) bytes·4 = roughly 100+ FLOPs/byte for that size. Compute-bound on L4 (compute roof ≈ 60 TFLOP/s FP32, bandwidth roof ≈ 300 GB/s, ridge ≈ 200 FLOP/byte — we're below ridge, so memory-bandwidth-bound at this size, but close).

**Q3: "Why FP32 accumulate inside your BF16 matmul?"**
A: Numerical stability. BF16 mantissa is 7 bits; accumulating thousands of products in BF16 would lose magnitude. FP32 accumulate preserves precision while inputs stay BF16 to halve memory traffic.

**Q4: "Why didn't you use cuBLAS / cuDNN?"**
A: Spec forbids external compute libraries. `llm_part1.md:30` requires C++ controller + custom CUDA kernels.

**Q5: "Why is your attention loop on the CPU?"**
A: Tracked as a known optimization (TODO #8). Correct first, fast second. Per-head dispatch from host is straightforward and verifiable; a fused on-device attention is \~200 LoC of new kernel work for \~300ms/forward.

**Q6: "Walk me through one decoder layer."**
A: Layer body is `src/inference.cu` lines \~280–390. Order: input RMSNorm → Q/K/V matmul (resident BF16 or streamed FP32) → RoPE on Q and K → host-side per-head GQA attention → output projection → residual add → post-attn RMSNorm → gate/up matmul → SwiGLU → down matmul → residual add. Each step has a `Stopwatch` block for telemetry.

**Q7: "What's grouped-query attention and how did you implement it?"**
A: 32 Q heads share 8 K/V heads (4 Q heads per K/V group). In `run_attention_heads`, the head index `hi` maps to `kvg = hi / 4` so four Q heads in a row reuse the same K/V slice. Saves \~75% of K/V memory and bandwidth.

**Q8: "Why do you transpose weights at load?"**
A: HuggingFace stores `W` as `[out, in]`, but Llama's matmul is `X @ W^T`. Transposing once at load avoids a transpose-aware kernel or a per-layer GPU transpose.

**Q9: "Why isn't your softmax `exp(x) / sum(exp(x))`?"**
A: Numerical stability. Subtract row max before exponentiation: `exp(x - max(x)) / sum(exp(x - max(x)))`. Without this, large logits overflow to inf and softmax produces NaN.

**Q10: "RoPE pairs which dimensions?"**
A: First half with second half (i with i+head_dim/2), not even/odd interleaving. Llama 3 uses `rotate_full` from HuggingFace. Base is 500000, not 10000 (a common gotcha).

**Q11: "Why is `lm_head` separate from the embedding table?"**
A: `tie_word_embeddings: false` in Llama 3's config. `lm_head` is its own weight matrix.

**Q12: "How do you know your output is correct?"**
A: Three layers of validation: (1) M1 7/7 grading tests pass; (2) per-operator parity vs `reference.py` (PyTorch) in M2-3 fixtures; (3) end-to-end demo `bin/llm "What is the capital of California?"` returns "Sacramento." with the expected token sequence.

**Q13: "What were the hardest bugs?"**
A: (a) Forgetting RMSNorm's gamma multiply — passes a sanity check but produces wrong outputs once integrated. (b) RoPE base = 10000 instead of 500000 — outputs look numerically reasonable but are silently wrong. (c) Causal mask only applied to the last row instead of every row above the diagonal.

**Q14: "How would you make this 5× faster?"**
A: (a) Move per-head attention on-device (TODO #8) — saves PCIe round-trips. (b) Tensor-core WMMA matmul with BF16 inputs (TODO #3). (c) Fuse RMSNorm + matmul (epilog fusion). (d) GPU GEMV for `lm_head` (TODO #7). (e) FP8 quantization for weights (TODO #5).

**Q15: "What's TODO #2 batching and how did you validate it?"**
A: Added a leading `[B]` dimension end-to-end — KVCache slabs, embedding lookup pads to max length, forward_step row-stacks activations as `[B*q_seq, d]`, RoPE kernel decodes position from row index. Validation is `batched_b2_distinct_parity`: run B=1 with prompt A and prompt B separately, then B=2 with [A, B], and confirm batch positions match the singletons within 1e-3 max-abs-diff on the final hidden state.

**Q16: "What do `Validation:` lines in `docs/todos/TODO.md` mean?"**
A: Internal QA gates we set for ourselves. Not class-spec requirements. The header note in TODO.md documents this. The class spec's only mandated correctness check is `llm_part2.md:174` (argmax token from a forward pass matches `reference.py`).

---

## 4. Things to have open in tabs during the review

- `src/inference.cu` (the forward pass — most likely starting point)
- `kernel/matmul.cu` (will absolutely be asked about)
- `kernel/rope.cu`, `kernel/attention.cu`, `kernel/rmsnorm.cu` (probable follow-ups)
- `tests/test_api.cpp` (M1 implementations)
- `docs/llm_part1.md`, `docs/llm_part2.md` (spec, for citing line numbers)
- `reference.py` (ground truth)
- This file

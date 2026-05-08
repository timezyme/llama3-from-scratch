# `src/inference.cu` Follow-Up Tracker

This note preserves the suggested `src/inference.cu` follow-ups for the next planning agent. It is a tracker, not approval to implement.

## Current Status

- `CUDA_CHECK` was re-verified after the implementation review: the macro currently has its continuation backslashes, `git diff -- src/inference.cu` is empty, and `git diff --check -- src/inference.cu` passes. There is no current `CUDA_CHECK` blocker.
- If `src/inference.cu` is edited again, keep CUDA error checking intact; do not weaken or remove `CUDA_CHECK` while splitting files.

## Non-Negotiable Guardrails

- Preserve the professor-facing mandatory harness:
  - Do not change `tests/test.cpp`, `tests/test_api.h`, or `tests/test_api.cpp`.
  - `make tests && ./bin/tests 1` must keep working on a CPU-only checkout.
  - `make -n tests` must not pull in `main.o`, `inference.o`, `device_weights.o`, `kv_cache.o`, or CUDA-only kernel objects.
- Preserve public inference API compatibility in `include/inference.h`.
- Preserve required B=1 single-token behavior before touching optional batching or performance work.
- Preserve the repo fact that `tie_word_embeddings=false`; logits use the separate `global/lm_head_weight.bin`, not the embedding table.
- Treat local macOS CPU-only limits as verification limits, not CUDA correctness findings. CUDA validation belongs on the L4 lane.

## Candidate Refactor: Split `src/inference.cu`

Goal: reduce file size and make ownership/lifetime boundaries easier to review without changing behavior.

Suggested split:

- `src/inference.cu`
  - Keep only public API wrappers and high-level generation entry points, or make it the thin orchestration file.
- `src/inference_forward.cu`
  - Move `forward_step`, layer loop orchestration, and per-layer tensor flow.
- `src/inference_attention.cu`
  - Move `AttentionScratch` and `run_attention_heads`.
- `src/inference_generation.cu`
  - Move single-prompt and batched generation loops if `src/inference.cu` becomes too busy.
- `src/inference_utils.cu` or a small header
  - Move chat template helpers, RoPE table allocation, length validation, `decode_token`, and `compute_lm_head_logits` if needed.

Planning constraints:

- Keep the split mechanical first. Do not combine file splitting with kernel rewrites, GPU `lm_head`, new attention kernels, or workspace redesign.
- Update the Makefile object lists deliberately. Any new CUDA source used by `bin/llm` or `tests_m2m3` must be added to the correct CUDA-only object list.
- Do not expose internal helpers publicly unless tests or a real caller need the seam.

## Candidate Optimization Backlog

Order these by risk and measurable payoff, not by aesthetics.

1. **Reusable inference workspace**
   - Replace per-`forward_step` `cudaMalloc`/`cudaFree` scratch allocation with an RAII workspace that is sized once per generation session.
   - Lower risk than kernel rewrites, but shape/lifetime validation must be strict.

2. **Profiling-mode synchronization**
   - Keep CUDA error checking.
   - Consider gating some inner-loop `cudaDeviceSynchronize()` calls behind a profiling/debug flag or replacing timing syncs with CUDA events.
   - Do not remove synchronization needed before host reads.

3. **GPU `lm_head` GEMV + argmax**
   - Move final logits and argmax to device to remove the CPU `VOCAB_SIZE * EMBEDDING_DIM` loop per generated token.
   - Medium risk because final token IDs can change with numeric differences; require parity against the current CPU `lm_head` path.

4. **Fused or decode-specialized attention**
   - Highest likely speedup and highest risk.
   - Especially useful for `q_seq == 1` decode, where a specialized GQA kernel can avoid materializing per-head Q/K/V/S/O scratch.
   - Must preserve GQA grouping, RoPE positions, causal masking, KV-cache indexing, and batch indexing.

5. **Batching-aware attention improvements**
   - Bonus-only scope. Keep required B=1 compliance separate.
   - Do not destabilize single-prompt generation while chasing B>1 performance.

## Verification Shape

Use targeted lanes before broad regression:

1. Local CPU mandatory harness: `make clean && make tests && ./bin/tests 1`.
2. L4 build smoke after any split: `make ARCH=sm_89 all tests tests_m2m3`.
3. L4 targeted correctness:
   - `./bin/llm "What is the capital of California?"`
   - `./bin/llm --max-tokens 8 "What is the capital of California?"`
   - `./bin/tests_m2m3 batched_b2_distinct_parity`
4. L4 performance lane only after correctness: `./tools/test_l4.sh --perf`.
5. Quick regression after targeted pass: `./tools/test_l4.sh --quick`.

Avoid using the full L4 regression as the inner loop; reserve it for the final acceptance gate.

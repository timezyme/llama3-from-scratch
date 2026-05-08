# Remaining Work Plan

> **Credit-priority backlog**: `docs/todos/TODO.md` (which bonus item to pick
> next when the user wants to maximize grade points). When the priority
> orderings disagree, the user picks; close out by editing both files.

This plan replaces the old milestone plans. Those plans described required
Milestone 2-3 work that has since landed. This file is now the working plan for
finishing the project from the current `main` branch. Items here are numbered
by execution order (perf/dependency-driven), not by credit value.

## Current Status

The required assignment work for Parts 1 and 2 is implemented.

- Part 1 is complete: tokenizer, binary weight dump/load, embedding lookup,
  tiled CUDA matmul, CPU fallback for M1 tests, shared-memory tiling,
  coalesced memory access, and double buffering.
- Milestone 2 is complete: RMSNorm and Q/K/V projections are implemented and
  covered by the M2-3 fixture tests.
- Milestone 3 is complete: RoPE, grouped-query attention, causal masking,
  stable softmax, output projection, residual add, SwiGLU FFN, 32-layer
  forward pass, final RMSNorm, `lm_head`, argmax, and CLI token generation are
  implemented.
- The current CLI supports single-token generation and multi-token generation
  through `--max-tokens`.

Known verification:

- Local verification on this machine: `make all tests` passes, and all seven
  M1 grading tests pass.
- Fresh L4 rebaseline on 2026-05-02: quick path passed M1 `7/7` and fast
  M2-3 `25/25` in `180.91s`; full path passed M1 `7/7` and M2-3 `28/28` in
  `783.14s`; `cs265-l4` was confirmed `TERMINATED` afterward.
- KV-cache one-token parity was added and verified on L4 on 2026-05-03:
  full path passed M1 `7/7` and M2-3 `29/29`; `cs265-l4` was confirmed
  `TERMINATED` afterward.
- Resident BF16 weight parity was later verified on L4 on 2026-05-03:
  quick path passed M1 `7/7` and fast M2-3 `27/27`; full path passed M1
  `7/7` and M2-3 `32/32`. The L4 VM was stopped afterward.

## TODO #1 Research Notes

Reference projects point to one clean architecture for KV caching:

- `ai-dock/llama.cpp-cuda` is a build/release wrapper, not an inference
  implementation. Its useful clue is that it delegates the real model execution
  to upstream `ggml-org/llama.cpp`.
- `llama.cpp` allocates K/V tensors once per layer in backend buffers, places
  them on the same device as the layer when offload is enabled, and keeps model
  tensors in backend weight buffers. Its decode path feeds only the current
  token batch through the graph while the memory/KV subsystem supplies the
  cached prefix.
- `vLLM` separates KV block management from compute. That is the right idea for
  paged attention later, but it is too much machinery for the current B=1
  contiguous-cache TODO.
- TensorRT-LLM explicitly splits context/prefill from generation. It documents
  contiguous KV cache as the simpler form and paged KV cache as the later
  memory-efficient form. For this project, TODO #1 should finish the contiguous
  cache first; paged attention stays last.

Rule for this project: use these as design references only. Do not copy code.
The implementation stays home-grown and shaped around the existing
`KVCache`, `DeviceModelWeights`, and CUDA kernels.

## Bonus Status

| Item | Status | What remains |
|------|--------|--------------|
| KV cache (+5%) | Implemented | 8-token resident perf gate verified on L4 2026-05-03; warm decode 346 ms/token, no forbidden streaming timers. |
| Batching (+5%) | Not implemented | Add a leading batch dimension end-to-end and prove B>1 parity. |
| Tensor cores | Not implemented | L4 BF16 tensor-core capability is verified, and a BF16-weight matmul primitive exists, but WMMA is not wired into inference. |
| Sampling | Not implemented | Add temperature, top-k, top-p, and fixed-seed behavior. |
| Quantization | Not implemented | Add q8 or q4 weight dumps, loader path, kernel path, and drift tests. |
| Paged attention | Not implemented | Depends on stable KV cache and should come late. |
| GPU `lm_head` | Not implemented | Current `lm_head` projection is still a CPU dot product. |
| On-device attention reshape | Not implemented | Current attention path still copies Q/K/V to host for per-head reshape. |

## Remaining Tasks

1. DONE - Rebaseline the current branch on L4.

   Completed on 2026-05-02.

   Results:
   - `./tools/test_l4.sh --quick`: M1 `7/7`, fast M2-3 `25/25`, `180.91s`
     wall time including VM start and stop.
   - `./tools/test_l4.sh`: M1 `7/7`, M2-3 `28/28`, `783.14s` wall time
     including VM start and stop.
   - Build used CUDA 12.9 at `/usr/local/cuda-12.9` and `ARCH=sm_89`.
   - `cs265-l4` was confirmed `TERMINATED` afterward.

   Pass gate met:
   - M1 grading tests pass unchanged.
   - Full M2-3 suite passes.
   - Binaries compile with `ARCH=sm_89`.
   - No L4 VM is left running.

2. DONE - Add explicit KV-cache parity coverage.

   Completed on 2026-05-03.

   Added `full_forward_kv_cache_one_token_parity` to the M2-3 suite. It proves
   cached one-token generation matches the existing single-token path:

   ```text
   generate_tokens(prompt, 1)[0] == generate_next_token(prompt)
   ```

   Results:
   - `./tools/test_l4.sh`: M1 `7/7`, M2-3 `29/29`.
   - New parity test passed on L4.
   - Existing full-forward tests still passed, including
     `full_forward_medium_prompt`.
   - `cs265-l4` was confirmed `TERMINATED` afterward.

3. DONE - Implement resident BF16 device weights.

   Completed on 2026-05-03:
   - Added validated raw-BF16 tensor loading APIs.
   - Added a separate CUDA-owned `DeviceModelWeights` path that can load a
     layer into resident transposed BF16 projection buffers and FP32 norm
     buffers.
   - Added M2-3 tests for raw loader parity and layer-0 resident-weight smoke.
   - Added resident generation entry points and rewired the multi-token CLI path
     to load resident BF16 layer weights once per process instead of streaming
     layer weights on every decode step.
   - Added `full_forward_resident_one_token_parity` as the L4 gate proving the
     resident BF16 path returns the same first token as the streaming path.
   - L4 quick path passed M1 `7/7` and fast M2-3 `27/27`.
   - L4 full path passed M1 `7/7` and M2-3 `32/32`.
   - The L4 VM was stopped afterward.

   Current file boundary:
   - `include/loader.h` / `src/loader.cpp`: raw-BF16 tensor read helper with
     header and shape validation.
   - `include/device_weights.h` / `src/device_weights.cu`: resident device
     weight owner and load/free lifecycle.
   - `src/inference.cu`: resident-weight forward path that removes
     `load_layer`/`unload_layer` from decode.
   - `main.cpp`: `--max-tokens N` uses the resident BF16 path; single-token CLI
     stays on the streaming path for the existing Paris sanity gate until L4
     parity is proven.
   - `tests/test_m2m3.cpp`: focused resident-weight parity or smoke tests.

   Pass gate met:
   - `cudaMemGetInfo` shows enough free memory after residency.
   - M1 and full M2-3 pass with the resident path tests included.

4. DONE - Rewire inference matmuls to use resident BF16 weights.

   Use the existing `gpu_matmul_device_bf16_weight` primitive for FP32
   activations multiplied by BF16 resident weights. This should remove most
   repeated host-to-device weight copies before introducing WMMA.

   Local code landed on 2026-05-03:
   - `forward_step` now accepts an optional `DeviceModelWeights` pointer.
   - Resident mode uses BF16 device projection buffers directly for Q/K/V, O,
     gate/up/down and resident FP32 norm buffers for both per-layer RMSNorms.
   - Streaming mode is preserved for the existing required tests and fallback.
   - `generate_tokens_resident` loads all resident layers once before prefill
     and reuses them through decode.

   L4 verification completed with M1 `7/7` and M2-3 `32/32`.

5. DONE - Close the TODO #1 KV-cache performance gate.

   Completed on 2026-05-03.

   Ran `./tools/test_l4.sh --perf` on `cs265-l4` (us-east1-c, sm_89, CUDA 12.9)
   after an incremental build. Total wall 196.12s for
   `./bin/llm --max-tokens 8 "The capital of France is"`.

   Results:
   - `PASS kv_cache_perf` printed by the audit lane.
   - Forbidden streaming timers absent across all decode steps:
     `layer.load_disk_to_host`, `layer.h2d_weights`, `layer.unload`.
   - Required summary fields all present.
   - Cached output: `"Paris!"` (token 60704, then EOT_ID 128009 at step 3,
     ending generation early). Matches the validated single-token Paris result.

   Telemetry:
   - `weights.load_all_resident_bf16`: 176.39s one-time cold load of all 32
     layers as transposed BF16.
   - `step.prefill` (15 prompt tokens): 378.21 ms.
   - `step.decode` (per token): 346.11 ms avg over 2 invocations
     (345.25 ms / 346.96 ms).
   - Per-layer breakdown across 96 invocations:
     `layer.attn_pre` 2.12 ms, `layer.attn_heads` 2.97 ms,
     `layer.post_attn_and_ffn` 5.94 ms, `layer.total` 11.14 ms.
   - VRAM: 22.03 GiB total; resident weights used 13.00 GiB; 8.59 GiB free
     after residency.

   Pass gate met:
   - 8-token CLI completes (the "agreed performance budget" was never
     quantified in the plan, so the gate closes on the absence of forbidden
     streaming timers plus functional output, matching the script's own
     pass criteria).
   - Cached output matches reference behavior for the first token.
   - `docs/JOURNAL.md` records timing.

   Remaining ~346 ms warm-decode cost is dominated by per-head attention
   H2D/D2H thrashing in `run_attention_heads`
   (`src/inference.cu:79-151`), not KV-cache or weight streaming. That belongs
   to item 7 (on-device per-head attention reshape).

6. Move `lm_head` to the GPU.

   Replace the CPU vocabulary dot product with a GPU GEMV or GEMM path. Keep
   last-token-only projection.

   Pass gate:
   - GPU `lm_head` logits match the CPU helper within tolerance.
   - `full_forward_medium_prompt` still passes.
   - CLI Paris sanity still returns token `60704`.

7. Move per-head attention reshape onto the device.

   Remove the current Q/K/V host reshape loop. Add a CUDA dispatch/reshape
   kernel that builds per-head Q, K-transposed, and V slices directly on the
   GPU.

   Pass gate:
   - `decoder_block_layer0_fixture` passes.
   - `full_forward_hello` and `full_forward_medium_prompt` pass.
   - Telemetry shows per-layer attention H2D/D2H reshape traffic is gone.

8. Implement BF16 WMMA tensor-core matmul.

   Add a WMMA path for BF16 inputs with FP32 accumulation on sm_89. Keep the
   scalar BF16-weight matmul as the reference path until drift is measured.

   Pass gate:
   - WMMA matmul parity passes against the scalar BF16-weight path.
   - Drift is documented before any tolerance change.
   - WMMA is at least 2x faster than scalar BF16 matmul on a representative
     4096 x 4096 case.
   - Full M2-3 and `tools/verify_reference.py` pass within the accepted BF16
     tolerance.

9. Add batching.

   Add a leading batch dimension through token IDs, embeddings, kernels,
   attention, KV cache ownership, and output handling. Keep B=1 as the default
   path.

   Pass gate:
   - B=1 remains unchanged.
   - B=2 forward output matches two separate B=1 runs.
   - Repeat for B=4 and B=8 if VRAM allows.
   - Throughput improves enough to justify the batch path.

10. Add sampling.

    Add host-side sampling after logits are available: temperature, top-k,
    top-p, and a fixed seed.

    Pass gate:
    - `temp=0` exactly matches argmax.
    - Fixed seed runs are reproducible.
    - Existing greedy tests and CLI sanity still pass.

11. Add quantization only after the resident-weight path is stable.

    Start with per-output-channel INT8 weights before attempting lower bit
    widths.

    Pass gate:
    - Quantized dump files and scales are generated reproducibly.
    - Quantized matmul drift is measured per operator.
    - End-to-end argmax remains unchanged on reference prompts.

12. Add paged attention last.

    Paged attention should depend on a stable KV cache and passing batching
    path. Add page tables only after the simpler contiguous KV cache is fully
    correct and fast.

    Pass gate:
    - Paged and non-paged attention match at sequence lengths 16, 64, 256, and
      1024.
    - No regression in M1, M2-3, CLI sanity, or `verify_reference.py`.

13. Final documentation and submission pass.

    Update `docs/JOURNAL.md`, `docs/todos/TODO.md`, README references, and any
    runbook details after the final implementation state is known.

    Pass gate:
    - The docs no longer describe completed work as pending.
    - All test commands and expected outputs are current.
    - The final state is easy to explain in code review: required work,
      optional bonuses, measured speedups, and known tradeoffs.

## Common Gate After Every Implementation Task

Every task above should end with the same basic verification:

```bash
make tests
for i in 1 2 3 4 5 6 7; do ./bin/tests "$i"; done
./tools/test_l4.sh --quick
```

For any task that changes inference behavior, also run:

```bash
./tools/test_l4.sh
python3 tools/verify_reference.py
./bin/llm "The capital of France is"
```

Do not relax a tolerance silently. Record the measured drift and reason in
`docs/JOURNAL.md` before moving on.

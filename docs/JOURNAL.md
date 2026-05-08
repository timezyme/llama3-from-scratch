# Project Journal

> Chronological log of milestones, design decisions, and review outcomes for the CS265 LLM project. New entries append to the bottom.

## 2026-02-20 — Milestone 1 Complete

**Implementation**: BPE tokenizer (greedy merge loop), weight dumper (256B header + BF16 payload, 291 tensors), weight loader (BF16/FP16->FP32 via milifloat.h), tiled CUDA GEMM kernel (shared memory, coalesced access), CPU matmul fallback, TestAPI wiring (tokenize w/ BOS, get_embeddings, matmul).

**Hardening**: size_t overflow fix, CPU loop reorder (i,k,j), CUDA free_all lambda, TILE_SIZE compile-time config, unique_ptr ownership, singleton tokenizer/loader.

**GCP T4 results**: All tests passed — tokenize, embeddings (3x4096), matmul (up to 128x256x64, max err 2.3e-05).

## 2026-03-22 — Upstream Tests + Double Buffering

- Pulled tests 2-7 + binary fixtures from professor's repo. Kept our test_api.cpp (remote had stubs).
- Added token.model/tokenizer.model from upstream.
- Set up local env: downloaded Llama 3 weights, created .venv, ran dumper locally. All 7 tests pass (CPU).
- Added double-buffered shared memory to CUDA matmul kernel. TILE_SIZE 16->32. All 7 tests pass on GCP T4.
- Audited all llm_part1.md requirements — all 9 mandatory items pass. M2 spec not yet released.

## 2026-05-02 — Spec Compliance Review (Parts 1 & 2)

**Method**: Cross-checked `docs/llm_part1.md` (M1) and `docs/llm_part2.md` (M2/M3) against current code. Local CPU build clean; all 7 M1 tests pass on macOS fallback.

**Findings**:
- All M1 requirements met: tokenizer (BPE greedy merge), dumper/loader (280B header + BF16), embedding lookup, tiled GEMM with shared mem + coalesced + double-buffered.
- All M2 requirements met: RMSNorm (one-block-per-row, eps inside sqrt, gamma applied), Q/K/V projections via existing matmul.
- All M3 requirements met: RoPE (rotate-full pairing i↔i+h_d/2, base 500000), GQA (g=i/(h/h_k)), causal mask via -1e6 add, numerically-stable softmax (max-subtract), output projection + residual, SwiGLU (SiLU(gate)*up + 3 matmuls), 32-layer loop, final RMSNorm + lm_head + argmax.
- Common pitfalls (Part 2 §4) all avoided: stable softmax, correct RoPE pairing, base 500000, two distinct RMSNorm weights/layer, gamma applied, weights transposed at load, last-token-only lm_head, full-triangle causal mask.

**Bonus credit status**:
- Done: float4 vectorization, double-buffered shared memory tiles, bank-conflict avoidance.
- Missing: tensor cores, batching (5%), KV caching (5%), quantization, paged attention, sampling.

**Other observations**:
- Per-layer attention does H2D/D2H of Q/K/V every layer for host-orchestrated per-head reshape (`inference.cu:209-218`). Functionally correct, perf-suboptimal.
- `lm_head` projection runs on CPU (`compute_lm_head_logits`). Trivially movable to GPU GEMV.
- `bin/llm` produces only 1 token; spec says single-step is correct, but multi-token loop is a small UX win.

**Action items captured in** `docs/todos/TODO.md`. First item (KV caching) being addressed next.

## 2026-05-02 — KV cache attempt: correct logic, wrong perf model

**Goal**: TODO #1 — implement KV caching for the 5% spec-bonus credit.

**Built**:
- `include/kv_cache.h` + `src/kv_cache.cu` — RAII `KVCache` with per-layer device K/V buffers sized to `S_MAX=1024`.
- `src/inference.cu` refactored to a unified `forward_step(h_input, q_seq, weights, cache, ...)` handling both prefill (`q_seq=N`, `len_before=0`) and decode (`q_seq=1`, `len_before>0`). Causal mask only when `q_seq==kv_seq`. K/V projections write directly into the cache slot at `len_before`.
- `main.cpp` — `--max-tokens N` flag, decode loop until EOT or limit.
- `tests/test_m2m3.cpp` untouched (its `run_forward_pass` is file-local, so existing 27 tests still pass).

**GCP T4 validation**:
- 7/7 M1 tests PASS, 27/27 M2-3 tests PASS.
- Single-token CLI: `"The capital of France is"` → `Paris` (token 60704). Correct.
- Multi-token CLI (`--max-tokens 8`): cancelled. Far too slow.

**Root cause of multi-token slowness**:
`forward_step` calls `weights.load_layer(layer)` / `unload_layer(layer)` inside the 32-layer loop. Each `load_layer` reads BF16 from disk, converts to FP32, transposes, allocates host memory; `unload_layer` frees it. 8-token generation pays this 32×8=256 times. Single-token only pays 32. KV cache logic is correct; the problem is the layer-streaming cost is NOT amortized across decode steps, defeating the cache's purpose.

**Memory wall blocking the obvious fix**:
- 32 layers FP32 on host = ~48GB → n1-standard-4 has 15GB RAM. ❌
- 32 layers FP32 on GPU = ~24GB → T4 has 16GB. ❌
- 32 layers BF16 on GPU = ~12GB → fits T4, but requires BF16-input matmul kernel (significant rewrite, pairs with TODO #3).

**Status**: implementation correctness verified, perf goal not met. Need a memory plan before this can claim bonus credit. Consulting Codex on the right architecture next; may combine with TODO #3.

**VM**: stopped (TERMINATED) after validation. No further GCP usage until plan is ready.

## 2026-05-02 — Stated goal + GPU choice reconsidered

**User goal**: complete **ALL** bonus features if possible — KV caching (#1), batching (#2), tensor cores (#3), and the spec-mentioned topics where credit is achievable.

**T4-specific limitation that drove the earlier hedge**: T4 (sm_75) has FP16/INT8 tensor cores only, **no BF16 tensor cores** (those require sm_80+). On T4 you cannot keep BF16 numerics AND use tensor cores in the same kernel — forcing a choice between fixture parity and tensor-core credit.

**This is a GPU choice, not a project choice**. Better-fit GCP options:

| GPU | Arch | VRAM | BF16 TC | FP8 TC | Spot $/hr (approx) | Fits all bonuses? |
|-----|------|------|---------|--------|-------------------|-------------------|
| T4  | sm_75 | 16 GB | no | no | $0.10–0.16 | KV+batching only; tensor cores only via FP16 (numerics drift) |
| L4  | sm_89 | 24 GB | **yes** | yes | ~$0.20–0.30 | yes; comfortable margin for batching too |
| A100 40GB | sm_80 | 40 GB | yes | no | ~$1.20+ | yes; overkill, expensive |

**Recommendation**: switch to **L4** (g2-standard-4 + 1× nvidia-l4 in us-central1-a). It removes the BF16-vs-tensor-cores conflict, gives 8 GB more VRAM (room for batching activations + resident embeddings + `lm_head`), and is roughly the same spot cost as T4. Plan before any code: spin a one-shot L4 to confirm spot availability and pricing in our zone, then keep T4 stopped.

**Plan reframed for sm_80+ hardware**:
- Phase A: telemetry on chosen GPU; confirm `cudaMemGetInfo()` margin.
- Phase B: resident BF16 weights + BF16-input matmul → makes **TODO #1 (KV caching)** deliver real speedup.
- Phase C: GPU `lm_head` + on-device per-head attention → drops decode latency further (and TODO #7, #8).
- Phase D: WMMA path with BF16 inputs and FP32 accumulator → **TODO #3 (tensor cores)** without numerical compromise.
- Phase E: batched forward pass → **TODO #2 (batching)** within VRAM budget.
- Phase F: sampling, quantization, paged attention as further increments.

Codex consult and verification of Codex's claims captured in this conversation; key items checked against CUDA WMMA type matrix and per-layer memory math.

## 2026-05-02 — L4 verification burst (Step 1 complete)

**Provisioned**: `g2-standard-4` + 1× `nvidia-l4` SPOT in `us-east1-c` (us-central1-a/b/c had spot stockouts; us-east1-c had capacity). Image: `common-cu129-ubuntu-2204-nvidia-580`.

**Confirmed on L4**:
- `nvidia-smi` reports compute capability **8.9**, 23034 MiB total VRAM, 22564 MiB free at idle, driver 580.126.20.
- CUDA 12.9 toolkit installed at `/usr/local/cuda-12.9`.
- Project builds clean with `NVCCFLAGS='-std=c++17 -O2 -arch=sm_89'` against existing source. Only warning is the deprecated-arch notice for the Makefile's default `sm_75` (cosmetic).
- BF16 WMMA 16×16×16 smoke kernel: max abs diff vs CPU BF16-rounded reference = **5.96e-08** (effectively bit-exact for FP32 accumulator). Confirms BF16 tensor cores work on L4 sm_89 as documented.
- `cudaMemGetInfo` reports **21.84 GiB free / 22.03 GiB total** at runtime — 7+ GiB margin over our ~14.5 GiB resident-weights plan, comfortable for batched activations + workspace.

**Decision**: L4 confirmed adequate for the full bonus matrix (KV cache + tensor cores + batching + tied bonuses). No fallback to A100 needed.

**Cost**: ~$0.10 for the verification burst (provisioned, ran, deleted within ~20 min).

**Operational note**: spot capacity in us-central1-a was exhausted at provision time; us-east1-c had capacity. Future provisions should iterate across zones (us-central1-{a,b,c,f}, us-east1-{c,d}, us-west{1,4}-*) until spot lands. T4 in us-central1-a remains stopped as fallback.

**VM**: deleted. Both project VMs accounted for: T4 (cs265-gpu-test) TERMINATED in us-central1-a; L4 verification VM deleted. No active billing.

## 2026-05-02 — Phase B0 BF16-weight matmul primitive

**Goal**: first implementation slice toward resident BF16 weights on the L4 path. This is not copied from reference projects; it extends this repo's existing tiled FP32 GEMM shape so later phases can switch weight residency without changing call sites again.

**Built**:
- `gpu_matmul_device_bf16_weight(d_A, d_B_bf16, d_C, M, K, N)` in `kernel/matmul.cu` / `kernel/kernels.cuh`.
- BF16 weights are stored as raw `uint16_t` BF16 bits in device memory, loaded into shared memory through aligned `uint2` chunks when possible, expanded to FP32, and accumulated in FP32.
- `bf16_weight_matmul_parity` in `tests/test_m2m3.cpp`: compares the BF16-weight path against the existing FP32 device matmul using BF16-rounded reference weights.

**Self-review refinement**:
- Re-read `docs/llm_part1.md` / `docs/llm_part2.md` and CUDA guidance. The first version was correct but used scalar BF16 weight loads; refined it to vectorized BF16 tile loads with scalar fallback for edge tiles to better match the project's coalesced-memory requirement.
- Expanded `bf16_weight_matmul_parity` to cover both a model-shaped aligned case `(3,4096)x(4096,1024)` and a small ragged case `(5,17)x(17,19)` that exercises edge-tile scalar fallback.
- Searched the changed code for reference-project/source-copy traces. No external project names or source-attribution residue found in code; implementation remains built from this repo's existing tiled GEMM design.

**Local validation**:
- `git diff --check` PASS.
- `make tests` PASS.
- 7/7 M1 grading tests PASS locally on CPU fallback.

**Blocked gate**:
- `make tests_m2m3` cannot run on this Mac because `nvcc` is not installed.
- The configured GCP project currently exposes only the old `cs265-gpu-test` T4 VM. The previous L4 verification VM was deleted, and no active G2/L4 VM name/zone is discoverable from repo notes or local SSH config.

**Next required gate before Phase B1**:
- Run on G2/L4 with CUDA: `make clean && make tests_m2m3 && ./bin/tests_m2m3 bf16_weight_matmul_parity`.
- Then run all 27 M2-3 fixtures, all 7 M1 grading tests, and CLI sanity (`./bin/llm "The capital of France is"` -> token 60704 / Paris) on the same G2/L4 target.

## 2026-05-02 — L4 build/test loop cleanup

Optimized the G2/L4 test environment without changing inference behavior: `Makefile` now takes `ARCH=sm_89`, `tools/test_l4.sh` uses `make -j$(nproc)`, preserves source mtimes on copy, avoids unconditional `make clean`, and has a `--quick` path. Provisioning was rerun idempotently: model shards, dump, `token.model`, and M2-3 fixtures were all present and skipped; `~/.bashrc` now adds CUDA 12.9 to PATH. Results: first quick run with `layer_streaming_smoke` included took 5:07; revised quick path skipped `full_forward_*` plus `layer_streaming_smoke` and passed in 1:11 (M1 7/7, M2-3 fast subset 25/25). Default full path reused the build and passed all tests (M1 7/7, M2-3 28/28) but took 12:20 including VM stop, so the remaining full-loop cost is the slow tests/runtime, not build setup. VM stopped afterward (`cs265-l4` TERMINATED).

## 2026-05-02 — Remaining-work Task 1 L4 rebaseline complete

Executed Task 1 from `docs/plans/remaining-work-plan.md` on the G2/L4 VM. Quick rebaseline used CUDA 12.9 at `/usr/local/cuda-12.9`, built for `ARCH=sm_89`, and passed M1 7/7 plus the fast M2-3 subset 25/25 in 180.91s wall time including VM start/stop. Full rebaseline reused the build and passed M1 7/7 plus all M2-3 tests 28/28 in 783.14s wall time including VM start/stop. `cs265-l4` was checked afterward with `gcloud compute instances list` and confirmed `TERMINATED` in `us-east1-c`. Next task is explicit KV-cache parity coverage before resident BF16 weight work.

## 2026-05-03 — Remaining-work Task 2 KV parity gate complete

Added `full_forward_kv_cache_one_token_parity` to `tests/test_m2m3.cpp`. The test compares `generate_tokens(prompt, 1)[0]` with `generate_next_token(prompt)` for `"The capital of France is"`, giving the KV-cache API an explicit one-token parity gate before resident-weight work. Verified on the G2/L4 VM with `./tools/test_l4.sh`: M1 `7/7`, M2-3 `29/29`, including the new parity test and the existing `full_forward_medium_prompt`. The full run remains slow because the current inference path still reloads/unloads layer weights; this is the main target for the resident BF16 device-weight phase. `cs265-l4` was checked afterward and confirmed `TERMINATED` in `us-east1-c`.

## 2026-05-03 — Task 3 resident BF16 weights local slice

Started the resident-weight phase without rewiring inference yet. Added `load_1d_bf16_raw` and `load_2d_bf16_raw` to the dump loader so BF16 payload bits can be read with dtype, rank, shape, and file-size validation. Added `DeviceModelWeights` in a separate CUDA module (`include/device_weights.h`, `src/device_weights.cu`) so CUDA allocation/copy logic stays out of the CPU `ModelWeights` fallback. The device owner loads layer projections as transposed raw BF16 buffers and norm weights as FP32 device buffers; `tests/test_m2m3.cpp` now has `raw_bf16_loader_parity` and `resident_layer0_weight_smoke`. Local gates: `make all tests` passed, M1 `7/7` passed, `git diff --check` passed. CUDA verification is still pending: `make tests_m2m3` cannot run on this Mac (`nvcc` missing), and sandboxed `gcloud` cannot run `tools/test_l4.sh` without writing to its config directory.

## 2026-05-03 — Task 4 resident inference rewire local slice

Rewired inference to use the new resident BF16 layer buffers without removing the old streaming path. `forward_step` now accepts an optional `DeviceModelWeights` pointer; resident mode uses `gpu_matmul_device_bf16_weight` for Q/K/V, O, gate/up/down, and resident FP32 per-layer norm buffers. Added `generate_next_token_resident` and `generate_tokens_resident`; the multi-token CLI path now loads resident BF16 layer weights once per process, while the single-token CLI path stays on the previous streaming implementation for the existing Paris sanity check until L4 parity is proven. Added `full_forward_resident_one_token_parity` to compare resident first-token output with the streaming path on L4. Local gates passed: `make all tests`, M1 `7/7`, and `git diff --check`. CUDA gates are still pending because this Mac has no `nvcc`; `./tools/test_l4.sh --quick` is also blocked by sandboxed `gcloud` being unable to write `/Users/spasco/.config/gcloud/credentials.db`.

## 2026-05-03 — L4 test framework lane split

Fixed the L4 workflow so full regression is no longer the default development loop. `tools/test_l4.sh` now defaults to the quick lane, adds explicit `--unit`, `--quick`, `--perf`, and `--full` lanes, and reserves `--full` for the final acceptance gate. The new `--perf` lane builds only `bin/llm`, runs the TODO #1 resident 8-token KV-cache audit with a 300s timeout, saves `build/l4-kv-cache-perf.log`, prints load/prefill/decode timers, and fails if old streaming timers (`layer.load_disk_to_host`, `layer.h2d_weights`, `layer.unload`) appear. Updated `docs/RUNBOOK-L4.md`, `README.md`, `docs/development-guide.md`, `CLAUDE.md`, and `docs/learnings.md` to match. Local validation: `bash -n tools/test_l4.sh`, help/invalid-lane parsing, `git diff --check`, `make tests`, and M1 `7/7` all passed. L4 lane runtime still needs live verification on `cs265-l4`.

## 2026-05-03 — TODO #1 KV-cache perf gate verified on L4

Closed the bonus-credit KV-cache perf gate. Ran `./tools/test_l4.sh --perf` on `cs265-l4` (us-east1-c, sm_89, CUDA 12.9). Total wall 196.12s for `./bin/llm --max-tokens 8 "The capital of France is"` after an incremental build. The audit lane reported `PASS kv_cache_perf`; the three forbidden streaming timers (`layer.load_disk_to_host`, `layer.h2d_weights`, `layer.unload`) were absent across all decode steps. Required summary fields all printed: `weights.load_all_resident_bf16` = 176.39s (one-time cold load of all 32 layers as transposed BF16), `step.prefill` = 378.21 ms (15 prompt tokens), `step.decode` = 346.11 ms/token avg over 2 invocations. Per-layer breakdown for the 96 invocations across 3 steps: `layer.attn_pre` 2.12 ms, `layer.attn_heads` 2.97 ms, `layer.post_attn_and_ffn` 5.94 ms, `layer.total` 11.14 ms. Cached output: `"Paris!"` (token 60704, then EOT_ID 128009 at step 3, ending generation early — matches the validated single-token Paris result). VRAM: 22.03 GiB total; resident weights consumed 13.00 GiB, leaving 8.59 GiB free. Plan item 5 flipped from IN PROGRESS to DONE; bonus-status table flipped KV cache from Partially implemented to Implemented. Remaining ~346 ms warm-decode cost is dominated by per-head attention H2D/D2H thrashing in `run_attention_heads` (`src/inference.cu:79-151`), not KV-cache or weight streaming — that's item 7 territory (on-device per-head attention reshape). No "agreed performance budget" was ever quantified in the plan, so the gate closes on the absence of forbidden streaming timers plus functional output, matching the script's own pass criteria.

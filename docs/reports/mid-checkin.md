# Milestone 1 Status

The PDF requires 5 steps. All 5 are implemented and tested.

| Step | Requirement | Status |
|------|-------------|--------|
| 1. Download assets | Model weights + tokenizer in assets/llama3/ | Done (on GCP VM) |
| 2. Tokenize prompts | BPE encode/decode working | Done. "Hello world" -> [128000, 9906, 1917] |
| 3. Dump and load weights | Python dumper + C++ loader | Done. 291 tensors dumped in BF16 |
| 4. Embedding lookup | Token IDs -> FP32 embedding vectors | Done. 3 tokens -> 12288 floats, values match |
| 5. CUDA matmul kernel | Tiled GEMM with shared memory | Done. Max error 2.3e-05 vs CPU reference |

## Test Results

All 5 tests passed on the GCP T4 VM:
- Grading test (./bin/tests 1): PASSED
- Embeddings: PASSED
- Matmul (identity, known values, large random): PASSED

## Code Review (2026-02-20)

Post-implementation review applied 7 fixes:
- Integer overflow protection in matmul size checks (test_api.cpp)
- Cache-friendly loop reorder in CPU matmul (matmul_cpu.cpp)
- Leak-safe CUDA memory cleanup via unified free_all lambda (matmul.cu)
- Configurable TILE_SIZE with static_assert guards (matmul.cu)
- Exception-safe unique_ptr ownership in loader (loader.cpp)
- Shared singleton tokenizer/loader to avoid per-call reconstruction (test_api.cpp)
- Style: size_t by value, removed stale comments (loader.h)

## What's Not Done

- Tests 2+ in the grading harness haven't been revealed yet -- we only know test 1.
  The graders may have additional tests for embeddings and matmul that we haven't seen.
- Model assets and dumped weights only exist on the GCP VM, not locally.
- Milestone 2 (RMSNorm, attention, SWiGLU, full decoder) hasn't been released yet --
  expected early March.

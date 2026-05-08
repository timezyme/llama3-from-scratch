<!-- Generated: 2026-05-02 | Updated: 2026-05-08 | Files scanned: 26 source + 8 kernel + 8 tool | Token estimate: ~800 -->

# Architecture

## Project Type
Single-app C++17/CUDA inference engine for Llama 3 8B Instruct.

## Pipeline
```
Prompt
  -> BPETokenizer.encode (src/tokenizer_bpe.cpp)
  -> apply_chat_template (src/inference_chat.cu)
  -> ModelWeights.get_embeddings (src/model_weights.cpp)
  -> 32x Decoder Layer (resident BF16 weights on GPU):
       RMSNorm -> Q/K/V Proj -> RoPE -> GQA Attention -> O Proj -> Residual
       RMSNorm -> Gate/Up Proj -> SwiGLU -> Down Proj -> Residual
  -> Final RMSNorm
  -> lm_head output projection (NOT embedding table)
  -> Argmax (single-token) OR loop with KV cache (multi-token)
  -> BPETokenizer.decode
```

Four CLI modes (selected in `main.cpp` argv parser):
- `bin/llm "prompt"` -> single token via `generate_next_token` (FP32 streaming path)
- `bin/llm --max-tokens N "prompt"` -> N tokens via `generate_tokens_resident` (resident BF16 + KV cache)
- `bin/llm --prompt p1 --prompt p2 --max-tokens N` -> B>1 batched `generate_tokens_resident` (equal-length tokenizations only)
- `bin/llm --interactive --max-tokens N` -> REPL (N >= 2): weights load once, prompt loop on stdin

Resident BF16 weights load once at REPL/multi-token startup (~165s cold) and stay on
the GPU. Each decode step uses the cached weights + KVCache, so per-token latency is
bounded by attention + matmul, not disk I/O. See `docs/JOURNAL.md` for measured timings.

## Module Map
```
main.cpp                     CLI entry; parses --max-tokens, dispatches to generate_*
config.h                     Model constants (dims, paths, layer count)

include/tokenizer.h          LLMTokenizer (abstract) + BPETokenizer
src/tokenizer_bpe.cpp        BPE encode/decode, special token handling

include/loader.h             LlamaDumpLoader: binary dump -> FP32
src/loader.cpp               280-byte header parse, BF16/FP16/FP32 decode
include/milifloat.h          bf16_to_float(), half_to_float()

include/model_weights.h      LayerWeights, GlobalWeights, ModelWeights
src/model_weights.cpp        Transpose-at-load, layer-by-layer streaming

include/inference.h          generate_next_token(), generate_tokens(),
                              generate_*_resident() (single/multi/batched/debug),
                              decode_token(), GenerateDebugResult
src/inference.cu             Public-API facade: thin delegators to *_impl
src/inference_chat.cu        Llama 3 Instruct chat-template wrapping
src/inference_layer.cu       forward_step + per-head attention + lm_head logits
src/inference_loop.cu        Three orchestrators (single, KV-cached, batched)
src/inference_internal.h     Project-internal cross-TU declarations

include/device_weights.h     DeviceLayerWeights, DeviceModelWeights:
                              GPU-resident BF16 projections + FP32 layernorms
src/device_weights.cu        cudaMalloc upload, transposed BF16 projections,
                              load_layer / load_all_layers / unload_layer

include/kv_cache.h           KVCache: device-side per-layer K/V buffers
src/kv_cache.cu              cudaMalloc/cudaFree of [max_len, kv_dim] per layer

include/instrument.h         Header-only Stopwatch + probe_vram (telemetry)
include/operator.cuh         AbstractOperator base for GPU operators
include/prelude.h            Common type aliases and STL imports
```

## Build Dispatch
```
Makefile: nvcc detected?
  yes -> kernel/*.cu + src/*.cu compiled, CUDA_ENABLED defined,
         model_weights / device_weights / inference{,_chat,_layer,_loop} /
         kv_cache / 6 kernels linked into bin/llm
  no  -> bin/llm refuses to link; bin/tests still builds via kernel/matmul_cpu.cpp

ARCH variable selects target SM (default sm_75 for T4 fallback;
override with `make ARCH=sm_89` for L4).
```

## Entry Points
| Binary           | Source              | Purpose                              |
|------------------|---------------------|--------------------------------------|
| `bin/llm`        | main.cpp            | Single- or multi-token CLI inference |
| `bin/tests`      | tests/test.cpp      | 7 M1 grading tests                   |
| `bin/tests_m2m3` | tests/test_m2m3_main.cpp | 38 M2-3 internal tests (CUDA req.) |

## Test Phases (M2-3, registered in `build_registry()` at test_m2m3_main.cpp)
- Phase 0 (5): matmul device-parity sanity, BF16 loader/weight smoke (`test_m2m3_matmul.cpp`)
- Phase 1 (7): RMSNorm + Q/K/V projection fixtures (`test_m2m3_rmsnorm_proj.cpp`)
- Phase 2 (8): RoPE, GQA, mask, softmax, attention output (`test_m2m3_rope_attn.cpp`)
- Phase 3-4 (7): residual, SwiGLU, decoder block, kernel smoke, full_forward_hello (`test_m2m3_decoder_full.cpp`)
- Phase 5 (11): final norm, lm_head, layer streaming, embedding batched padding, KV cache bounds, full_forward_*, batched_b2_distinct_parity (`test_m2m3_kv_batch.cpp`)

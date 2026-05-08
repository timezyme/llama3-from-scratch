<!-- Generated: 2026-05-02 | Updated: 2026-05-05 | Files scanned: 5 (loader, model_weights, dumper, gen_token_model, kv_cache) | Token estimate: ~550 -->

# Data Format

## Binary Dump Format (280-byte header)
```
[0..255]   tensor name (null-padded ASCII)
[256..259] dtype code (uint32 LE): 0=FP32, 1=FP16, 2=BF16
[260..263] ndims (uint32 LE): 1 or 2
[264..271] shape[0] (uint64 LE)
[272..279] shape[1] (uint64 LE, 0 for 1D)
[280..]    raw payload in declared dtype
```
Produced by `tools/dumper.py` from HuggingFace safetensors.

## Weight Layout on Disk
```
assets/llama3/dump/
  embeddings.bin           [128256, 4096] BF16
  global/
    model_norm_weight.bin  [4096] BF16
    lm_head_weight.bin     [128256, 4096] BF16   <-- NOT the embedding table
  layer_00/ ... layer_31/
    model_layers_N_input_layernorm_weight.bin           [4096]
    model_layers_N_post_attention_layernorm_weight.bin  [4096]
    model_layers_N_self_attn_q_proj_weight.bin          [4096, 4096]
    model_layers_N_self_attn_k_proj_weight.bin          [1024, 4096]
    model_layers_N_self_attn_v_proj_weight.bin          [1024, 4096]
    model_layers_N_self_attn_o_proj_weight.bin          [4096, 4096]
    model_layers_N_mlp_gate_proj_weight.bin             [14336, 4096]
    model_layers_N_mlp_up_proj_weight.bin               [14336, 4096]
    model_layers_N_mlp_down_proj_weight.bin             [4096, 14336]
```
291 tensors total. Dirs use zero-padded index (`layer_00`); filenames use
plain (`model_layers_0_`).

## Tokenizer Artifact
```
assets/llama3/token.model    BPE rank file: "<base64-bytes> <rank>" per line
```
Produced by `tools/gen_token_model.py` from HuggingFace `tokenizer.json`.
Path is set in `config.h::TOKENIZER_PATH`. The C++ BPE tokenizer reads ranks
from this file at startup.

## Loading Strategy
- All weights stored as BF16 on disk; `milifloat.h` exposes `bf16_to_float()`.
- 2D projection weights transposed at load time: `[out, in] -> [in, out]`.
- **Two paths**, selected by call site in `src/inference.cu`:
  - **FP32 streaming** (single-token, M1 grading): `weights.load_layer(N)` reads
    BF16, expands to FP32, uploads to the GPU; `unload_layer(N)` frees afterward.
  - **Resident BF16** (multi-token / `--max-tokens N>1` / `--interactive`):
    all 32 layers loaded once as BF16 device buffers (~13 GiB on L4); the BF16-
    weight matmul kernel (`gpu_matmul_device_bf16_weight`) reads BF16 and
    accumulates in FP32. Required for the KV-cache TODO #1 perf gate.
- Embeddings cached in `LlamaDumpLoader` for row lookups.

## KV Cache Layout (device-side, multi-token decode only)
```
KVCache (allocated once at decode start):
  per layer L in [0, 32):
    d_K[L]: [max_seq_len, NUM_KV_HEADS * HEAD_DIM] FP32  -- 1024*1024 floats default
    d_V[L]: [max_seq_len, NUM_KV_HEADS * HEAD_DIM] FP32

  len_: int -- count of valid rows (advanced by caller after append)
```
Total VRAM at default `max_seq_len=1024`: 32 layers x 2 (K,V) x 1024 x 1024 x 4B
= 256 MiB. See `include/kv_cache.h`.

## Model Constants (config.h)
| Constant       | Value   |
|----------------|---------|
| EMBEDDING_DIM  | 4096    |
| NUM_HEADS      | 32      |
| NUM_KV_HEADS   | 8       |
| HEAD_DIM       | 128     |
| FFN_DIM        | 14336   |
| VOCAB_SIZE     | 128256  |
| NUM_LAYERS     | 32      |
| ROPE_BASE      | 500000  |
| RMS_NORM_EPSILON | 1e-5  |

<!-- Generated: 2026-05-02 | Updated: 2026-05-05 | Files scanned: 7 kernel files | Token estimate: ~600 -->

# CUDA Kernels

All host entry points declared in `kernel/kernels.cuh`.
All device pointers unless noted. 256 threads/block standard.

## Kernel Inventory

| File             | Host Entry                          | Operation                              |
|------------------|-------------------------------------|----------------------------------------|
| matmul.cu        | gpu_matmul()                        | C[M,N] = A[M,K]*B[K,N], host ptrs      |
| matmul.cu        | gpu_matmul_device()                 | Same, device ptrs (no H2D/D2H copies)  |
| matmul.cu        | gpu_matmul_device_bf16_weight()     | A FP32 × B BF16 device ptrs; FP32 accum (resident BF16 weight path) |
| matmul_cpu.cpp   | gpu_matmul()                        | CPU fallback, i-k-j loop order         |
| rmsnorm.cu       | gpu_rmsnorm()                       | Row-wise RMSNorm, 1 block/row          |
| rope.cu          | gpu_rope()                          | Rotary pos embed, rotate_full, base=500k |
| rope.cu          | precompute_rope_table()             | Host-side cos/sin table                |
| attention.cu     | gpu_scale()                         | data[i] *= scale                       |
| attention.cu     | gpu_causal_mask()                   | S[r,c] = -1e6 where c > r              |
| attention.cu     | gpu_softmax()                       | Row-wise, numerically stable           |
| swiglu.cu        | gpu_swiglu()                        | SiLU(gate) * up                        |
| residual.cu      | gpu_residual_add()                  | a[i] += b[i] in-place                  |

`src/kv_cache.cu` is **not** a kernel — it is a host-side allocator that owns
`cudaMalloc`'d per-layer K/V buffers. See `data.md` for layout.

## Matmul Kernel Detail
- Double-buffered tiled GEMM with shared memory
- Tiles: BM=128, BN=128, BK=16; thread tile: TM=8, TN=8
- 256 threads/block (16x16), each computes 64 output elements
- Shared memory +1 padding (bank conflict avoidance)
- Matrix B: float4 vectorized loads for coalesced access
- Scalar FP32 FMA. The resident BF16-weight path (`gpu_matmul_device_bf16_weight`)
  reads BF16 tile-by-tile, expands to FP32 in shared memory, and accumulates in
  FP32 — still scalar, not WMMA. WMMA / BF16 tensor cores not yet wired (TODO #3).

## Forward Pass Kernel Sequence (per layer)
```
gpu_rmsnorm(X, gamma, Xnorm)          # pre-attention norm
gpu_matmul_device(Xnorm, Wq, Q)       # Q projection
gpu_matmul_device(Xnorm, Wk, K)       # K projection
gpu_matmul_device(Xnorm, Wv, V)       # V projection
gpu_rope(Q, cos, sin)                 # rotary embeddings
gpu_rope(K, cos, sin)
  [per-head attention loop, host-orchestrated]:
  gpu_matmul_device(Qi, KgT, S)       # QK^T
  gpu_scale(S, 1/sqrt(hd))            # scale
  gpu_causal_mask(S)                  # mask future
  gpu_softmax(S)                      # attention weights
  gpu_matmul_device(S, Vg, Oi)        # weighted values
gpu_matmul_device(attn, Wo, attn_out) # output projection
gpu_residual_add(X, attn_out)
gpu_rmsnorm(X, gamma, Xnorm)          # post-attention norm
gpu_matmul_device(Xnorm, Wgate, gate) # gate projection
gpu_matmul_device(Xnorm, Wup, up)     # up projection
gpu_swiglu(gate, up, gate)            # activation
gpu_matmul_device(gate, Wdown, ffn)   # down projection
gpu_residual_add(X, ffn)
```

## Decode Mode (multi-token, KV-cached)
During incremental decode, Q is projected for 1 row only; new K/V rows are
appended to `KVCache.k(layer)` / `KVCache.v(layer)` and the per-head matmul
attends over the full prefix in cache. The kernel sequence is unchanged — only
the leading dimension of Q (and the K/V pointer) differs from prefill.

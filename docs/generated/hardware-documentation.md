# CUDA Kernel Documentation

**Generated:** 2026-03-29 | **Updated:** 2026-05-05 | **Scan Level:** Exhaustive

## Overview

The project implements 7 CUDA kernel files for GPU-accelerated Llama 3 8B inference. Kernels operate on row-major FP32 data; the matmul kernel additionally has a BF16-weight variant (`gpu_matmul_device_bf16_weight`) used by the resident-weights path. Production target is the NVIDIA L4 (Ada Lovelace, sm_89); kernels also build for T4 (sm_75) as a fallback. The kernel files are:

| File | Purpose |
|------|---------|
| `kernel/matmul.cu` | Tiled GEMM with double-buffered shared memory |
| `kernel/rmsnorm.cu` | Row-wise RMS normalization |
| `kernel/rope.cu` | Rotary position embeddings |
| `kernel/attention.cu` | Scale, causal mask, and softmax helpers |
| `kernel/swiglu.cu` | SwiGLU activation |
| `kernel/residual.cu` | In-place residual addition |
| `kernel/matmul_cpu.cpp` | CPU fallback for non-CUDA builds |

## Hardware Target

| Property          | NVIDIA L4 (production) | NVIDIA T4 (fallback) |
|-------------------|------------------------|-----------------------|
| Architecture      | Ada Lovelace (sm_89)   | Turing (sm_75)        |
| VRAM              | 24 GB GDDR6            | 16 GB GDDR6           |
| FP32 Performance  | ~30 TFLOPS             | 8.1 TFLOPS            |
| BF16 tensor cores | yes                    | no                    |
| Memory Bandwidth  | 300 GB/s               | 320 GB/s              |
| GCP Instance      | g2-standard-4          | n1-standard-4         |

L4 was selected over T4 because resident BF16 weights (~13 GiB) plus KV cache + activations need >16 GB, and BF16 tensor cores require sm_80+. Reference: `docs/JOURNAL.md` 2026-05-02 entry.

## Kernel Interface

```cpp
// kernel/kernels.cuh

// CUDA kernel (only visible when compiled by nvcc)
#ifdef __CUDACC__
__global__ void matmul_kernel(const float *A, const float *B, float *C,
                              int M, int K, int N);
#endif

// Host entry point — allocates device memory, copies host data, runs kernel, copies back
void gpu_matmul(const float *A, const float *B, float *C, int M, int K, int N);

// Device-pointer entry point — all pointers must already be in device memory
// No host-device copies. Used in the multi-step inference pipeline to avoid redundant cudaMemcpy calls.
void gpu_matmul_device(const float *d_A, const float *d_B, float *d_C, int M, int K, int N);
```

## CUDA Kernel Architecture (`kernel/matmul.cu`)

### Tile Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `BM` | 128 | Block tile rows (rows of C per thread block) |
| `BN` | 128 | Block tile columns (cols of C per thread block) |
| `BK` | 16 | K-dimension tile size (step through K) |
| `TM` | 8 | Thread tile rows (output elements per thread, row) |
| `TN` | 8 | Thread tile columns (output elements per thread, col) |
| `BLOCK_X` | 16 | Threads per block (x dimension) |
| `BLOCK_Y` | 16 | Threads per block (y dimension) |
| Threads/block | 256 | `BLOCK_X * BLOCK_Y` |
| Output/thread | 64 | `TM * TN` = register-level accumulation |

### Algorithm

1. **Double-buffered shared memory:**
   - Two buffers: `smA[2][BM][BK+1]` and `smB[2][BK][BN+1]`
   - While computing on buffer `cur`, the next tile prefetches into buffer `nxt`
   - `+1` padding on inner dimension avoids shared memory bank conflicts

2. **Prefetch first tile:** Load tile 0 into buffer 0 before the main loop

3. **Main loop** (iterate over K-dimension tiles):
   - Swap buffers (`cur` and `nxt`)
   - If not the last tile, prefetch next tile into `nxt` buffer
   - `__syncthreads()` — ensure current tile is fully loaded
   - Compute: each thread accumulates `TM x TN` partial products using register array `acc[TM][TN]`

4. **Store results:** Write `acc[TM][TN]` back to global memory C (with bounds checking)

### Memory Access Optimization

- **Vectorized B loads:** Matrix B loaded via `float4` (128-bit) reads for coalesced global memory access
- **Shared memory:** Tiles reused `BK` times before reload, reducing global memory bandwidth
- **Register blocking:** Each thread holds 64 accumulators in registers, maximizing compute-to-memory ratio
- **Bank conflict avoidance:** `+1` padding on shared memory inner dimension

### Host Wrapper (`gpu_matmul`)

```
1. Validate non-negative dimensions (M, K, N)
2. cudaMalloc → d_A [M*K], d_B [K*N], d_C [M*N]
3. cudaMemcpy host → device (A, B)
4. Compute grid dimensions:
   grid = ((N + BN - 1) / BN, (M + BM - 1) / BM)
   block = (BLOCK_X, BLOCK_Y)
5. Launch matmul_kernel<<<grid, block>>>
6. cudaGetLastError() + cudaDeviceSynchronize()
7. cudaMemcpy device → host (C)
8. cudaFree (d_A, d_B, d_C)
```

### Device-Pointer Wrapper (`gpu_matmul_device`)

```
1. Validate non-negative dimensions (M, K, N)
2. Compute grid dimensions:
   grid = ((N + BN - 1) / BN, (M + BM - 1) / BM)
   block = (BLOCK_X, BLOCK_Y)
3. Launch matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N)
4. cudaGetLastError() + cudaDeviceSynchronize()
```

Unlike `gpu_matmul`, this entry point skips all `cudaMalloc`/`cudaMemcpy`/`cudaFree` calls. All three pointers (`d_A`, `d_B`, `d_C`) must reside in device memory. This is the variant used throughout the multi-layer inference pipeline where tensors stay on the GPU between operations.

---

## RMSNorm Kernel (`kernel/rmsnorm.cu`)

### Formula

```
output[r, c] = input[r, c] / RMS(row_r) * gamma[c]

where RMS(row_r) = sqrt(mean(row_r^2) + epsilon)
```

Note: `epsilon` is inside the `sqrt`, not added after it.

### Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Threads/block | 256 | One block per row |
| Grid | `rows` blocks | Each block processes one row |
| Shared memory | 256 floats | Partial sums for tree reduction |

### Algorithm

Two-pass approach within each block:

1. **Pass 1 -- Sum of squares:** Each thread accumulates `input[r, c]^2` for its assigned columns, then a shared-memory tree reduction computes the total sum. The RMS value is `sqrt(sum / cols + epsilon)`.
2. **Pass 2 -- Normalize and scale:** Each thread computes `output[r, c] = input[r, c] / rms * gamma[c]`.

### Host Entry Point

```cpp
void gpu_rmsnorm(const float *d_input, const float *d_gamma, float *d_output,
                 int rows, int cols, float epsilon);
```

All pointers must be in device memory.

---

## RoPE Kernel (`kernel/rope.cu`)

### Overview

Applies rotary position embeddings in-place to Q and K tensors. Uses the "rotate_full" pairing convention where dimension `i` is paired with dimension `i + head_dim/2`, not consecutive pairs.

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base frequency | 500,000 | Llama 3 value (original RoPE paper uses 10,000) |
| Pairing | `(i, i + head_dim/2)` | "rotate_full" convention |
| Grid | One thread per `(position, head, pair_index)` triple | |

### Algorithm

For each `(position, head, pair_index)` triple:
1. Look up precomputed `cos(theta)` and `sin(theta)` from the frequency table.
2. Let `x0 = x[pos, head, pair_index]` and `x1 = x[pos, head, pair_index + head_dim/2]`.
3. Apply rotation in-place:
   - `x[..., pair_index]         = x0 * cos - x1 * sin`
   - `x[..., pair_index + half]  = x0 * sin + x1 * cos`

### Host Entry Points

```cpp
// Apply RoPE in-place on device memory
void gpu_rope(float *d_x, const float *d_cos, const float *d_sin,
              int seq_len, int num_heads, int head_dim);

// Precompute cosine/sine tables on the host
void precompute_rope_table(std::vector<float> &cos_out, std::vector<float> &sin_out,
                           int seq_len, int head_dim, float base);
```

`precompute_rope_table` runs on the CPU and fills `cos_out` and `sin_out` with shape `[seq_len, head_dim/2]`. The tables are then copied to device memory before calling `gpu_rope`.

---

## Attention Helper Kernels (`kernel/attention.cu`)

Three kernels bundled in one file, used together during the attention computation.

### Scale Kernel

Multiplies every element by a scalar.

```cpp
void gpu_scale(float *d_data, int count, float scale);
```

| Parameter | Value |
|-----------|-------|
| Threads/block | 256 |
| Grid | `(count + 255) / 256` blocks |
| Operation | `d_data[i] *= scale` |

One thread per element.

### Causal Mask Kernel

Applies a causal (lower-triangular) mask to an `s x s` attention score matrix.

```cpp
void gpu_causal_mask(float *d_S, int s);
```

| Parameter | Value |
|-----------|-------|
| Grid | Covers all `s * s` elements |
| Operation | `if (col > row) d_S[row * s + col] = -1e6` |

One thread per element of the `s x s` matrix. Positions where `col > row` (upper triangle) are set to `-1e6` so they become zero after softmax.

### Softmax Kernel

Numerically stable row-wise softmax.

```cpp
void gpu_softmax(float *d_data, int rows, int cols);
```

| Parameter | Value |
|-----------|-------|
| Threads/block | 256 |
| Grid | `rows` blocks (one block per row) |
| Shared memory | 256 floats for reductions |

Three-pass approach within each block:

1. **Find row max:** Shared-memory tree reduction to find `max(row)`.
2. **Exp and sum:** Compute `exp(x - max)` for each element and reduce to get the row sum.
3. **Normalize:** Divide each element by the row sum.

---

## SwiGLU Kernel (`kernel/swiglu.cu`)

### Formula

```
output[i] = SiLU(gate[i]) * up[i]

where SiLU(x) = x / (1 + exp(-x))
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Threads/block | 256 |
| Grid | `(count + 255) / 256` blocks |
| Aliasing | `d_output` may alias `d_gate` (in-place safe) |

One thread per element.

### Host Entry Point

```cpp
void gpu_swiglu(const float *d_gate, const float *d_up, float *d_output, int count);
```

All pointers must be in device memory. `d_output` may point to the same allocation as `d_gate`.

---

## Residual Add Kernel (`kernel/residual.cu`)

### Formula

```
a[i] += b[i]
```

In-place addition.

### Configuration

| Parameter | Value |
|-----------|-------|
| Threads/block | 256 |
| Grid | `(count + 255) / 256` blocks |

One thread per element.

### Host Entry Point

```cpp
void gpu_residual_add(float *d_a, const float *d_b, int count);
```

All pointers must be in device memory. `d_a` is modified in-place.

---

## CPU Fallback (`kernel/matmul_cpu.cpp`)

Used when `nvcc` is not available (e.g., local macOS development without CUDA):

```cpp
void gpu_matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    // i-k-j loop order for cache-friendly row-major access
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            const float a_ik = A[i * K + k];
            const float *b_row = &B[k * N];
            for (int j = 0; j < N; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}
```

The i-k-j loop order ensures sequential access through both `C[i,:]` and `B[k,:]` rows, giving good cache locality on row-major data.

## Build-Time Dispatch

The Makefile auto-detects `nvcc`:

```makefile
ifneq ($(shell command -v $(NVCC) 2>/dev/null),)
  CUDA_ENABLED := 1    # → links matmul.cu
else
  CUDA_ENABLED := 0    # → links matmul_cpu.cpp
endif
```

Both implementations export the same `gpu_matmul()` symbol, so the rest of the codebase links transparently against either version.

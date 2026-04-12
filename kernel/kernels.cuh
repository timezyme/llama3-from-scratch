// Kernel declarations for GPU operators (matmul, RMSNorm, etc.).
// Provides host-callable entry points that launch CUDA kernels internally.

#pragma once

#include "prelude.h"

// CUDA kernel declarations — only visible when compiled by nvcc.
#ifdef __CUDACC__
__global__ void matmul_kernel(const float *A, const float *B, float *C, int M,
                              int K, int N);
__global__ void rmsnorm_kernel(const float *input, const float *gamma,
                               float *output, int rows, int cols,
                               float epsilon);
#endif

// Host entry point: C[M,N] = A[M,K] * B[K,N], all row-major FP32.
// Dispatches to GPU kernel or CPU fallback depending on build configuration.
// Manages device memory internally (allocate, copy in, compute, copy out, free).
void gpu_matmul(const float *A, const float *B, float *C, int M, int K, int N);

// Device-pointer entry point: C[M,N] = A[M,K] * B[K,N], all row-major FP32.
// All pointers must already reside in device (GPU) memory.
// No host-device transfers are performed. Caller manages device memory.
// Exists to avoid redundant cudaMemcpy in multi-step pipelines (Part 2+).
void gpu_matmul_device(const float *d_A, const float *d_B, float *d_C,
                       int M, int K, int N);

// -----------------------------------------------------------------------
// RMSNorm: output[r,c] = input[r,c] / RMS(row_r) * gamma[c]
// RMS(row) = sqrt( mean(row^2) + epsilon )
// All pointers must be in device memory. One block per row.
void gpu_rmsnorm(const float *d_input, const float *d_gamma,
                 float *d_output, int rows, int cols, float epsilon);

// -----------------------------------------------------------------------
// RoPE: apply rotary position embeddings in-place.
// x: flat projected tensor [seq_len, num_heads * head_dim], device memory.
// cos_table, sin_table: precomputed [seq_len, head_dim/2], device memory.
// Pairs dimension i with i + head_dim/2 (rotate_full convention).
void gpu_rope(float *d_x, const float *d_cos, const float *d_sin,
              int seq_len, int num_heads, int head_dim);

// Precompute RoPE cos/sin tables on the host.
// cos_out, sin_out: [seq_len, head_dim/2], host memory.
void precompute_rope_table(float *cos_out, float *sin_out,
                           int seq_len, int head_dim, float base);

// -----------------------------------------------------------------------
// Attention helpers (all device pointers)

// Scale every element: data[i] *= scale.
void gpu_scale(float *d_data, int count, float scale);

// Causal mask: for S[s,s], set S[row,col] = -1e6 where col > row.
void gpu_causal_mask(float *d_S, int s);

// Row-wise numerically stable softmax (in-place).
// data: [rows, cols]. Subtracts row max before exp, then normalizes.
void gpu_softmax(float *d_data, int rows, int cols);

// -----------------------------------------------------------------------
// SwiGLU: output[i] = SiLU(gate[i]) * up[i]
// SiLU(x) = x / (1 + exp(-x))
// All pointers must be in device memory. d_output may alias d_gate.
void gpu_swiglu(const float *d_gate, const float *d_up,
                float *d_output, int count);

// -----------------------------------------------------------------------
// Residual add: a[i] += b[i], in-place.
// All pointers must be in device memory.
void gpu_residual_add(float *d_a, const float *d_b, int count);

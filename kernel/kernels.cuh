// Public host API for every GPU operator used by the inference pipeline.
//
// The pattern is: each kernel lives in its own .cu file with a launch
// helper named `gpu_<op>` that the C++ controller calls. The helper
// takes care of grid/block sizing, error checking, and (for the
// host-pointer variant) any host<->device copies. The controller in
// inference.cu therefore stays in plain C++ and never has to write a
// triple-chevron launch directly.
//
// Two variants per matmul:
//   gpu_matmul        — host pointers (used by the M1 grading test)
//   gpu_matmul_device — device pointers (used inside the forward pass)
// Plus gpu_matmul_device_bf16_weight, which keeps weights as BF16 in
// VRAM (video RAM) and widens to FP32 inside the kernel.

#pragma once

#include "prelude.h"

#include <cstdint>

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

// Device-pointer entry point: C[M,N] = A[M,K] * B_bf16[K,N].
// A and C are FP32 device pointers. B stores raw BF16 bits in device memory.
// Accumulation stays FP32, matching a BF16-rounded FP32 reference weight.
void gpu_matmul_device_bf16_weight(const float *d_A,
                                   const uint16_t *d_B_bf16,
                                   float *d_C, int M, int K, int N);

// -----------------------------------------------------------------------
// RMSNorm (root-mean-square layer normalization):
// output[r,c] = input[r,c] / RMS(row_r) * gamma[c]
// RMS(row) = sqrt( mean(row^2) + epsilon )
// All pointers must be in device memory. One block per row.
void gpu_rmsnorm(const float *d_input, const float *d_gamma,
                 float *d_output, int rows, int cols, float epsilon);

// -----------------------------------------------------------------------
// RoPE (rotary position embedding): apply rotations in-place.
// x: flat projected tensor [seq_len, num_heads * head_dim], device memory.
// For batched tensors, seq_len is B*q_seq and q_seq is the per-batch length.
// cos_table, sin_table: precomputed [q_seq, head_dim/2], device memory.
// Pairs dimension i with i + head_dim/2, matching Llama 3 rotate_half.
void gpu_rope(float *d_x, const float *d_cos, const float *d_sin,
              int seq_len, int num_heads, int head_dim, int q_seq = -1);

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

// Strided per-head gather/scatter, used to slice/place one head's [rows, head_dim]
// inside a packed [rows, stride] tensor on the device. No host transfers.
void gpu_gather_head(const float *d_src, float *d_dst, int rows, int head_dim,
                     int src_stride, int head_offset);
void gpu_gather_head_transpose(const float *d_src, float *d_dst, int rows,
                               int head_dim, int src_stride, int head_offset);
void gpu_scatter_head(const float *d_src, float *d_dst, int rows, int head_dim,
                      int dst_stride, int head_offset);

// -----------------------------------------------------------------------
// SwiGLU (Swish-Gated Linear Unit): output[i] = SiLU(gate[i]) * up[i]
// SiLU(x) = x / (1 + exp(-x))
// All pointers must be in device memory. d_output may alias d_gate.
void gpu_swiglu(const float *d_gate, const float *d_up,
                float *d_output, int count);

// -----------------------------------------------------------------------
// Residual add: a[i] += b[i], in-place.
// All pointers must be in device memory.
void gpu_residual_add(float *d_a, const float *d_b, int count);

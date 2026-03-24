// Kernel declarations for matrix multiplication.
// Provides a unified gpu_matmul() interface that links against either
// the CUDA tiled GEMM (matmul.cu) or the CPU fallback (matmul_cpu.cpp),
// depending on whether nvcc is available at build time.

#pragma once

#include "prelude.h"

// CUDA kernel declaration — only visible when compiled by nvcc.
#ifdef __CUDACC__
__global__ void matmul_kernel(const float *A, const float *B, float *C, int M,
                              int K, int N);
#endif

// Host entry point: C[M,N] = A[M,K] * B[K,N], all row-major FP32.
// Dispatches to GPU kernel or CPU fallback depending on build configuration.
void gpu_matmul(const float *A, const float *B, float *C, int M, int K, int N);

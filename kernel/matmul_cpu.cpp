// CPU fallback for the matmul kernel, used when the project is built
// without nvcc. It exists so headers/tests still link without CUDA — the
// inference binary itself refuses to run on a CPU-only build (see
// main.cpp). The signature matches gpu_matmul exactly so callers don't
// have to know which backend is compiled in.

#include "kernel/kernels.cuh"

#include <stdexcept>

// Device pointers are meaningless on a CPU build. Throw rather than
// silently dereferencing them.
void gpu_matmul_device(const float *, const float *, float *,
                       int, int, int) {
    throw std::runtime_error("gpu_matmul_device requires CUDA (nvcc build)");
}

// CPU GEMM (general matrix multiply): C[M,N] = A[M,K] * B[K,N], row-major.
void gpu_matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    if (M < 0 || K < 0 || N < 0) {
        throw std::runtime_error("gpu_matmul expects non-negative dimensions");
    }

    // i-k-j loop order (not the textbook i-j-k). Why: the innermost loop
    // walks `j` consecutively, which streams sequentially through both
    // C[i, :] and B[k, :] — both row-major contiguous. The naive i-j-k
    // would walk B by column, missing the cache on every inner iteration.
    for (int i = 0; i < M; ++i) {
        float *c_row = &C[i * N]; // pointer to output row i
        for (int j = 0; j < N; ++j) {
            c_row[j] = 0.0f; // zero the output row before accumulation
        }

        // Accumulate: C[i,j] += A[i,k] * B[k,j] for all k.
        // Hoisting a_ik outside the j-loop avoids redundant loads of A.
        for (int k = 0; k < K; ++k) {
            const float a_ik = A[i * K + k];
            const float *b_row = &B[k * N]; // pointer to B row k
            for (int j = 0; j < N; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

// CPU fallback for matrix multiply C[M,N] = A[M,K] * B[K,N].
// Used when nvcc is not available (no GPU build).
// Implements the same gpu_matmul interface so the rest of the codebase
// links transparently against either the CUDA or CPU version.

#include "kernel/kernels.cuh"

#include <stdexcept>

void gpu_matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    if (M < 0 || K < 0 || N < 0) {
        throw std::runtime_error("gpu_matmul expects non-negative dimensions");
    }

    // Uses i-k-j loop order (instead of the naive i-j-k) so the innermost
    // loop streams sequentially through both C[i,:] and B[k,:], giving
    // good cache locality on row-major data.
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

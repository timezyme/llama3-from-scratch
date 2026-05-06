// ============================================================================
// rmsnorm.cu — Root-Mean-Square layer normalization for Llama 3.
// ============================================================================
//
// What it does: normalizes each row of `input` by its root-mean-square, then
// scales by per-feature learned weights `gamma`. Replaces LayerNorm in
// Llama 3 (and most modern LLMs) — same idea, but with no mean subtraction
// and no bias, which is cheaper and works just as well in practice.
//
// Per-row formula:
//   rms(x) = sqrt(mean(x_i^2) + epsilon)
//   y_i    = (x_i / rms(x)) * gamma_i
//
// Where this kernel runs in the forward pass — three call sites:
//   (a) Pre-attention norm in every decoder block: normalize x before
//       the Q/K/V projections.
//   (b) Pre-FFN norm in every decoder block: normalize the attention
//       output before the gate/up projections.
//   (c) Final norm at the very end, just before lm_head.
//
// Read the file top-to-bottom — the layout matches execution order:
//   Section 1: Small helper (CUDA error wrap).
//   Section 2: rmsnorm_kernel — two passes: reduce sum-of-squares, then scale.
//   Section 3: gpu_rmsnorm    — host entry point.
//
// Credit:
//   - Algorithm: Zhang & Sennrich, "Root Mean Square Layer Normalization"
//     (arXiv:1910.07467, 2019).
//   - Tree reduction in shared memory (halve active threads each step):
//     Mark Harris, "Optimizing Parallel Reduction in CUDA" (NVIDIA, 2007);
//     equivalent treatment in Kirk & Hwu, Programming Massively Parallel
//     Processors (PMPP), Chapter 9.
//
// Common pitfalls (silent correctness bugs to watch for):
//   - Epsilon goes INSIDE the sqrt: sqrt(mean(x^2) + eps). The variant
//     sqrt(mean(x^2)) + eps fails on all-zero rows, dividing by ~eps and
//     blowing the activation up.
//   - The gamma multiply must NOT be skipped. Each call site has its own
//     learned gamma; Llama 3 has 2 vectors per decoder layer (pre-attn,
//     pre-FFN) plus 1 final-norm gamma — easy to miss-wire if all three
//     are treated as one.
//
// Glossary:
//   epsilon — small constant (1e-5 in this project) for numerical safety.
//   gamma   — per-feature learned scale, shape [cols]. One vector per
//             call site: 2 vectors per decoder layer (pre-attn, pre-FFN)
//             plus 1 final-norm vector before lm_head.
//   rows    — number of tokens being normalized (B*S in batched mode).
//   cols    — feature dimension (4096 for Llama 3 8B).
// ============================================================================

#include "kernel/kernels.cuh"

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

namespace {

// ----------------------------------------------------------------------------
// Section 1 — Small helper. Turns a CUDA status code into a thrown
// runtime_error that includes the source location and the failing call.
// ----------------------------------------------------------------------------
void throw_cuda_error(cudaError_t err, const char *expr, const char *file,
                      int line) {
    if (err == cudaSuccess) return;
    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << " for " << expr << ": "
        << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
}

} // namespace

#define CUDA_CHECK(expr) throw_cuda_error((expr), #expr, __FILE__, __LINE__)

// ============================================================================
// Section 2 — rmsnorm_kernel: one block normalizes one row.
// ============================================================================
//
// Block / thread layout:
//   One thread block per row of the input. Threads inside the block stride
//   across the row in steps of blockDim.x so consecutive lanes touch
//   consecutive columns (coalesced HBM reads and writes).
//
// Pass outline — labels are repeated inline below:
//   PASS 1 — Each thread accumulates a partial sum-of-squares for the
//            columns it owns. A tree reduction in shared memory collapses
//            the partial sums into a single row total.
//   COMPUTE — One scalar division + sqrt + epsilon-add gives the row's
//             RMS divisor. Every thread computes this independently
//             (cheaper than a dedicated broadcast step).
//   PASS 2 — Each thread writes y[i] = (x[i] / rms) * gamma[i] for the
//            columns it owns. No more reductions or syncs needed.
// ============================================================================
__global__ void rmsnorm_kernel(const float *__restrict__ input,
                               const float *__restrict__ gamma,
                               float *__restrict__ output,
                               int rows, int cols, float epsilon) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float *x = input + row * cols;
    float *y = output + row * cols;

    extern __shared__ float sdata[];

    // ---- PASS 1: per-thread partial sum-of-squares ---------------------
    // Each thread sums squares for the columns it owns. blockDim.x partial
    // sums end up in sdata[], one per thread, ready for the tree reduction.
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = x[i];
        thread_sum += val * val;
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Tree reduction: each step halves the active threads and adds the
    // upper half into the lower half until sdata[0] holds the row total.
    // log2(blockDim.x) sync barriers, contiguous-thread predication keeps
    // warp divergence to a minimum.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // ---- COMPUTE: per-row RMS divisor ----------------------------------
    // Epsilon is inside the sqrt so a row with sum_sq = 0 still produces
    // a finite (small) divisor instead of dividing by zero. Every thread
    // computes this scalar independently — sdata[0], cols, and epsilon
    // are already visible to all threads after the reduction barrier,
    // so the dedicated "one-thread-broadcasts" pattern is skipped.
    // Equivalent result, simpler code, one extra div+sqrt per thread
    // (negligible).
    float rms = sqrtf(sdata[0] / static_cast<float>(cols) + epsilon);

    // ---- PASS 2: scale and write back ----------------------------------
    // y[i] = (x[i] / rms) * gamma[i]. The gamma multiply must be applied
    // here — skipping it is a silent correctness bug because every
    // RMSNorm site has its own learned gamma weights.
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        y[i] = x[i] / rms * gamma[i];
    }
}

// ============================================================================
// Section 3 — gpu_rmsnorm: host entry point.
// ============================================================================
//
// Thin wrapper: pick block / grid shape, dispatch the kernel, surface CUDA
// errors via CUDA_CHECK. All pointers must already live in device memory.
// ============================================================================
void gpu_rmsnorm(const float *d_input, const float *d_gamma,
                 float *d_output, int rows, int cols, float epsilon) {
    if (rows <= 0 || cols <= 0) return;

    // 256 threads per block keeps the tree reduction power-of-two.
    // For cols=4096, each thread handles 16 elements per pass.
    const int threads = 256;
    const int shared_bytes = threads * sizeof(float);

    rmsnorm_kernel<<<rows, threads, shared_bytes>>>(
        d_input, d_gamma, d_output, rows, cols, epsilon);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

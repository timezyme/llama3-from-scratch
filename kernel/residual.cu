// ============================================================================
// residual.cu — Residual (skip) connection: a[i] += b[i], in place.
// ============================================================================
//
// What it does: elementwise add b into a. Implements the two skip
// connections inside every decoder block — the trick that lets gradients
// (during training) and information (during inference) flow around each
// sub-block instead of having to pass through it. Without these adds,
// 32 layers of attention + FFN would amplify or attenuate signal until
// the activations collapse or explode; the skips keep magnitudes stable.
//
// Where this kernel runs in the forward pass — twice per decoder block:
//   (1) After attention:   X = X + attn_out
//   (2) After FFN:         X = X + ffn_out
//
// One kernel handles both call sites: the work shape and arithmetic are
// identical, so duplicating the code would gain nothing. Fusing this
// into the matmul output stage would shave one launch + one HBM round
// trip but obscures the per-layer skip structure.
//
// Read the file top-to-bottom — the layout matches execution order:
//   Section 1: Small helper (CUDA error wrap).
//   Section 2: residual_add_kernel — one thread per element, no reductions.
//   Section 3: gpu_residual_add    — host entry point.
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
// Section 2 — residual_add_kernel: one thread per element.
// ============================================================================
//
// a[i] += b[i] for i in [0, count). Fully data-parallel, no reductions or
// shared memory. The kernel is bandwidth-bound, not compute-bound.
// ============================================================================
__global__ void residual_add_kernel(float *__restrict__ a,
                                    const float *__restrict__ b,
                                    int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    a[i] += b[i];
}

// ============================================================================
// Section 3 — gpu_residual_add: host entry point.
// ============================================================================
//
// Thin wrapper: pick block / grid shape, dispatch the kernel, surface CUDA
// errors. Both buffers live in device memory; `count` is their flat
// element count (B * S * d for a layer's residual add).
// ============================================================================
void gpu_residual_add(float *d_a, const float *d_b, int count) {
    if (count <= 0) return;

    constexpr int threads = 256;
    int blocks = (count + threads - 1) / threads;

    residual_add_kernel<<<blocks, threads>>>(d_a, d_b, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

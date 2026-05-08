// ============================================================================
// swiglu.cu — SwiGLU activation for the Llama 3 FFN (feed-forward network).
// ============================================================================
//
// What it does: combines two parallel projections of the layer input —
// `gate` and `up` — into one elementwise product, with `gate` first
// passed through SiLU (Sigmoid Linear Unit, also called Swish):
//
//   SiLU(x)              = x * sigmoid(x) = x / (1 + exp(-x))
//   SwiGLU(gate, up)[i]  = SiLU(gate[i]) * up[i]
//
// Where this kernel runs in the forward pass — the full FFN sub-block:
//   gate    = X_norm @ W_gate^T                (matmul, [s, d_ff])
//   up      = X_norm @ W_up^T                  (matmul, [s, d_ff])
//   H       = SwiGLU(gate, up)                 <-- this kernel
//   ffn_out = H @ W_down^T                     (matmul, [s, d])
//
// Read the file top-to-bottom — the layout matches execution order:
//   Section 1: Small helper (CUDA error wrap).
//   Section 2: swiglu_kernel — one thread per output element, no reductions.
//   Section 3: gpu_swiglu    — host entry point.
//
// Credit:
//   - SwiGLU activation in transformer FFNs: Shazeer, "GLU Variants
//     Improve Transformer" (arXiv:2002.05202, 2020).
//   - SiLU / Swish activation: Ramachandran, Zoph, Le, "Searching for
//     Activation Functions" (arXiv:1710.05941, 2017); independently
//     introduced as SiL by Elfwing, Uchibe, Doya (arXiv:1702.03118, 2017).
//
// Implementation notes:
//   - One thread per output element. Fully data-parallel — no shared
//     memory, no reductions, no syncs. The kernel is bandwidth-bound.
//   - d_output is allowed to alias d_gate, so the caller can fuse this
//     in place over the gate buffer and skip one VRAM allocation per
//     layer's forward pass.
//
// Glossary:
//   d_ff  — FFN hidden dimension (14336 for Llama 3 8B).
//   gate  — left projection, fed through SiLU; controls "how much" of up.
//   up    — right projection; the "what" being gated by SiLU(gate).
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
// Section 2 — swiglu_kernel: one thread per output element.
// ============================================================================
//
// For each i in [0, count): silu(gate[i]) * up[i]. SiLU is computed inline
// as g / (1 + exp(-g)) — one expf + one division + one multiply. Output
// may alias `gate` so the caller can fuse this in place.
// ============================================================================
__global__ void swiglu_kernel(const float *__restrict__ gate,
                              const float *__restrict__ up,
                              float *__restrict__ output,
                              int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    // the kernal is these three lines
    float g = gate[i];
    float silu = g / (1.0f + expf(-g));
    output[i] = silu * up[i];
}

// ============================================================================
// Section 3 — gpu_swiglu: host entry point.
// ============================================================================
//
// Thin wrapper: pick block / grid shape, dispatch the kernel, surface CUDA
// errors. All three buffers (gate, up, output) live in device memory and
// share the same shape; `count` is their flat element count (s * d_ff
// per layer). `output` may alias `gate` to fuse the activation in place.
// ============================================================================
void gpu_swiglu(const float *d_gate, const float *d_up,
                float *d_output, int count) {
    if (count <= 0) return;

    constexpr int threads = 256;
    int blocks = (count + threads - 1) / threads;

    swiglu_kernel<<<blocks, threads>>>(d_gate, d_up, d_output, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

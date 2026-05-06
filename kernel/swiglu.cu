// SwiGLU (Swish-Gated Linear Unit) activation for the Llama 3
// FFN (feed-forward network).
//
// The full FFN is:
//   gate = X_norm @ W_gate^T            (matmul; in [s, d_ff])
//   up   = X_norm @ W_up^T              (matmul; in [s, d_ff])
//   H    = SiLU(gate) * up              <-- this kernel, elementwise
//   ffn_out = H @ W_down^T              (matmul; in [s, d])
//
// SiLU (Sigmoid Linear Unit, also called swish): SiLU(x) = x * sigmoid(x)
//   = x / (1 + exp(-x)). llm_part2 §3.2 requires this elementwise
// activation between the gate/up projections and W_down.
//
// One thread per output element — fully data-parallel, no reductions or
// shared memory needed. d_output is allowed to alias d_gate so the
// caller can fuse this into the gate buffer in place and avoid one VRAM
// (video RAM) allocation in the forward pass.

#include "kernel/kernels.cuh"

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

namespace {

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

__global__ void swiglu_kernel(const float *__restrict__ gate,
                              const float *__restrict__ up,
                              float *__restrict__ output,
                              int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float g = gate[i];
    float silu = g / (1.0f + expf(-g));
    output[i] = silu * up[i];
}

void gpu_swiglu(const float *d_gate, const float *d_up,
                float *d_output, int count) {
    if (count <= 0) return;

    constexpr int threads = 256;
    int blocks = (count + threads - 1) / threads;

    swiglu_kernel<<<blocks, threads>>>(d_gate, d_up, d_output, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

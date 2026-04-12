// SwiGLU CUDA kernel for Llama 3 FFN.
//
// Formula: output[i] = SiLU(gate[i]) * up[i]
//          SiLU(x) = x / (1 + exp(-x))
//
// One thread per element. d_output may alias d_gate.

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

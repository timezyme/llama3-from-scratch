// Residual addition kernel: a[i] += b[i], in place. One thread per element.
//
// This single kernel implements both residual connections inside every
// decoder block:
//   X = X + attn_out          (after the attention sub-block)
//   X = X + ffn_out           (after the FFN sub-block)
// llm_part2 §3.1 requires both residuals; reusing one elementwise
// kernel for both is the obvious move — the work shape is identical.

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

__global__ void residual_add_kernel(float *__restrict__ a,
                                    const float *__restrict__ b,
                                    int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    a[i] += b[i];
}

void gpu_residual_add(float *d_a, const float *d_b, int count) {
    if (count <= 0) return;

    constexpr int threads = 256;
    int blocks = (count + threads - 1) / threads;

    residual_add_kernel<<<blocks, threads>>>(d_a, d_b, count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

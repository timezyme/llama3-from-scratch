// RMSNorm CUDA kernel for Llama 3 inference.
//
// Formula: RMSNorm(x) = (x / RMS(x)) * gamma
//          RMS(x) = sqrt( (1/d) * sum(x_i^2) + epsilon )
//
// Note: epsilon is INSIDE the sqrt, not added after.
// Each row is normalized independently. One block per row.

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

// One block per row. Each thread handles a strided slice of the row.
// Uses shared memory for parallel reduction of the sum-of-squares.
__global__ void rmsnorm_kernel(const float *__restrict__ input,
                               const float *__restrict__ gamma,
                               float *__restrict__ output,
                               int rows, int cols, float epsilon) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float *x = input + row * cols;
    float *y = output + row * cols;

    extern __shared__ float sdata[];

    // Pass 1: compute sum of squares over this row.
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = x[i];
        thread_sum += val * val;
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Tree reduction in shared memory.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // RMS = sqrt(mean_sq + epsilon).  epsilon is INSIDE sqrt.
    float rms = sqrtf(sdata[0] / static_cast<float>(cols) + epsilon);

    // Pass 2: normalize and apply gamma scaling.
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        y[i] = x[i] / rms * gamma[i];
    }
}

void gpu_rmsnorm(const float *d_input, const float *d_gamma,
                 float *d_output, int rows, int cols, float epsilon) {
    if (rows <= 0 || cols <= 0) return;

    // Use 256 threads per block (power of 2 for reduction).
    // Each thread loops over ceil(cols / 256) elements.
    const int threads = 256;
    const int shared_bytes = threads * sizeof(float);

    rmsnorm_kernel<<<rows, threads, shared_bytes>>>(
        d_input, d_gamma, d_output, rows, cols, epsilon);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

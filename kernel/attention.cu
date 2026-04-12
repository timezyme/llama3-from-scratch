// Attention helper kernels for Llama 3 GQA attention.
//
// Provides:
// - Causal mask application (add -1e6 to upper triangle)
// - Numerically stable softmax (max subtraction before exp)
// - Scale kernel (multiply by 1/sqrt(h_d))

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

// Scale every element of a matrix by a constant factor.
__global__ void scale_kernel(float *__restrict__ data, int count,
                             float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] *= scale;
    }
}

// Apply causal mask to a score matrix S[s, s].
// For each (row, col) where col > row, set S[row, col] = -1e6.
__global__ void causal_mask_kernel(float *__restrict__ S, int s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= s * s) return;
    int row = idx / s;
    int col = idx % s;
    if (col > row) {
        S[idx] = -1e6f;
    }
}

// Numerically stable softmax over each row of a matrix.
// For each row: subtract max, exponentiate, normalize.
// One block per row, shared memory for reductions.
__global__ void softmax_kernel(float *__restrict__ data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float *row_data = data + row * cols;
    extern __shared__ float sdata[];

    // Pass 1: find row maximum
    float thread_max = -1e30f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = row_data[i];
        if (val > thread_max) thread_max = val;
    }
    sdata[threadIdx.x] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (sdata[threadIdx.x + stride] > sdata[threadIdx.x]) {
                sdata[threadIdx.x] = sdata[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    float row_max = sdata[0];

    // Pass 2: exponentiate (with max subtraction) and sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = expf(row_data[i] - row_max);
        row_data[i] = val;
        thread_sum += val;
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];

    // Pass 3: normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_data[i] /= row_sum;
    }
}

// --- Host entry points ---

void gpu_scale(float *d_data, int count, float scale) {
    if (count <= 0) return;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(d_data, count, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void gpu_causal_mask(float *d_S, int s) {
    if (s <= 0) return;
    int total = s * s;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    causal_mask_kernel<<<blocks, threads>>>(d_S, s);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void gpu_softmax(float *d_data, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    int threads = 256;
    int shared_bytes = threads * sizeof(float);
    softmax_kernel<<<rows, threads, shared_bytes>>>(d_data, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

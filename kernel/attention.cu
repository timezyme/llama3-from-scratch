// Attention helper kernels for Llama 3 GQA attention.
//
// Provides:
// - Causal mask application (add -1e6 to upper triangle)
// - Numerically stable softmax (max subtraction before exp)
// - Scale kernel (multiply by 1/sqrt(h_d))
// - Per-head strided gather/scatter (used to slice a single head out of
//   the row-major [seq, num_heads * head_dim] layout without leaving the GPU)

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

// Strided gather of a single head's [rows, head_dim] slice out of a packed
// [rows, src_stride] tensor. dst[r, c] = src[r, head_offset + c].
__global__ void gather_head_kernel(const float *__restrict__ src,
                                   float *__restrict__ dst, int rows,
                                   int head_dim, int src_stride,
                                   int head_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * head_dim;
    if (idx >= total) return;
    int r = idx / head_dim;
    int c = idx % head_dim;
    dst[r * head_dim + c] = src[r * src_stride + head_offset + c];
}

// Same gather but transposes from [rows, head_dim] to [head_dim, rows], so
// downstream matmul can consume a row-major K^T with shape [HEAD_DIM, kv_seq].
// Tiled with shared memory so both src loads (consecutive c) and dst stores
// (consecutive r) are coalesced; the +1 padding on the inner tile dim avoids
// shared-memory bank conflicts during the transpose.
constexpr int GATHER_T_TILE = 32;
__global__ void gather_head_transpose_kernel(const float *__restrict__ src,
                                             float *__restrict__ dst, int rows,
                                             int head_dim, int src_stride,
                                             int head_offset) {
    __shared__ float tile[GATHER_T_TILE][GATHER_T_TILE + 1];

    int r_in = blockIdx.y * GATHER_T_TILE + threadIdx.y;
    int c_in = blockIdx.x * GATHER_T_TILE + threadIdx.x;
    if (r_in < rows && c_in < head_dim) {
        tile[threadIdx.y][threadIdx.x] =
            src[r_in * src_stride + head_offset + c_in];
    }
    __syncthreads();

    int c_out = blockIdx.x * GATHER_T_TILE + threadIdx.y;
    int r_out = blockIdx.y * GATHER_T_TILE + threadIdx.x;
    if (c_out < head_dim && r_out < rows) {
        dst[c_out * rows + r_out] = tile[threadIdx.x][threadIdx.y];
    }
}

// Inverse of gather_head_kernel: write a [rows, head_dim] head back into
// dst[r, head_offset + c] inside a [rows, dst_stride] packed tensor.
__global__ void scatter_head_kernel(const float *__restrict__ src,
                                    float *__restrict__ dst, int rows,
                                    int head_dim, int dst_stride,
                                    int head_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * head_dim;
    if (idx >= total) return;
    int r = idx / head_dim;
    int c = idx % head_dim;
    dst[r * dst_stride + head_offset + c] = src[r * head_dim + c];
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

void gpu_gather_head(const float *d_src, float *d_dst, int rows, int head_dim,
                     int src_stride, int head_offset) {
    if (rows <= 0 || head_dim <= 0) return;
    int total = rows * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    gather_head_kernel<<<blocks, threads>>>(d_src, d_dst, rows, head_dim,
                                            src_stride, head_offset);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_gather_head_transpose(const float *d_src, float *d_dst, int rows,
                               int head_dim, int src_stride, int head_offset) {
    if (rows <= 0 || head_dim <= 0) return;
    dim3 block(GATHER_T_TILE, GATHER_T_TILE);
    dim3 grid((head_dim + GATHER_T_TILE - 1) / GATHER_T_TILE,
              (rows + GATHER_T_TILE - 1) / GATHER_T_TILE);
    gather_head_transpose_kernel<<<grid, block>>>(d_src, d_dst, rows, head_dim,
                                                  src_stride, head_offset);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_scatter_head(const float *d_src, float *d_dst, int rows, int head_dim,
                      int dst_stride, int head_offset) {
    if (rows <= 0 || head_dim <= 0) return;
    int total = rows * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    scatter_head_kernel<<<blocks, threads>>>(d_src, d_dst, rows, head_dim,
                                             dst_stride, head_offset);
    CUDA_CHECK(cudaGetLastError());
}

// Attention building-block kernels for Llama 3's Grouped Query Attention
// (GQA). GEMM means general matrix multiply. Each piece is separate so
// inference.cu can compose the llm_part2 §3.1 attention flow per head:
//
//   gather Q_i (and K_g transposed, V_g) -> matmul(Q_i, K_g^T) -> scale
//   -> causal_mask -> softmax -> matmul(softmax, V_g) -> scatter back
//
// The four numerically/architecturally important pieces — and the
// llm_part2 §4 pitfalls they avoid:
//
//   - SCALE by 1/sqrt(h_d) before softmax. Without scaling, dot products
//     grow as O(sqrt(h_d)) and push softmax into the saturated regime.
//   - CAUSAL MASK on every (p,q) with q>p across the full s*s score
//     matrix, not just the diagonal or last row. We use -1e6, the finite
//     sentinel suggested in llm_part2 §3.2, instead of literal -inf.
//   - NUMERICALLY STABLE SOFTMAX: subtract the per-row max before
//     exp(). Skipping this step makes exp() overflow on even modestly
//     large scores; the assignment marks this as non-optional.
//   - GATHER/SCATTER kernels stay on the GPU — they slice and write
//     individual heads out of the packed [s, h * h_d] layout without
//     ever copying to host. The transposed gather variant produces a
//     row-major K^T so the score-matrix matmul Q_i * K_g^T can use the
//     standard tiled GEMM kernel directly.

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

// Scale every element by a constant. Used to apply the 1/sqrt(h_d)
// factor to the raw QK^T scores before the causal mask and softmax.
__global__ void scale_kernel(float *__restrict__ data, int count,
                             float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] *= scale;
    }
}

// Causal mask for a square score matrix S[s, s]: for every (row, col)
// with col > row, write -1e6. After softmax those positions become
// effectively zero, so each query at position p only attends to itself
// and earlier tokens (the causal autoregressive constraint from
// llm_part2 §3.1).
//
// Why -1e6 instead of -inf: llm_part2 §3.2 explicitly suggests "add a
// large negative value such as -10^6 (acting as -inf) to all positions
// where q>p, then proceed with softmax normally." A finite sentinel
// also avoids literal infinities in the score matrix before softmax.
__global__ void causal_mask_kernel(float *__restrict__ S, int s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= s * s) return;
    int row = idx / s;
    int col = idx % s;
    if (col > row) {
        S[idx] = -1e6f;
    }
}

// Numerically stable softmax, in place, one block per row.
//
// Three-pass design (matches the order required by llm_part2 §4):
//   1. Find the row maximum via tree reduction in shared memory.
//   2. Exponentiate (val - row_max) — max-subtraction prevents the
//      overflow that the assignment explicitly warns about. Without it,
//      exp(score) is `inf` for moderately large scores and the kernel
//      silently produces NaN logits.
//   3. Reduce the exponentiated row to a sum, then normalize.
//
// Mathematically equivalent to the naive exp(S)/sum(exp(S)) because
// subtracting a constant from every element of softmax leaves the
// result unchanged.
__global__ void softmax_kernel(float *__restrict__ data, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float *row_data = data + row * cols;
    extern __shared__ float sdata[];

    // Pass 1: find row maximum.
    float thread_max = -1e30f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = row_data[i];
        if (val > thread_max) thread_max = val;
    }
    sdata[threadIdx.x] = thread_max;
    __syncthreads();

    // Tree reduction: pairwise max-merge halves the active threads each step.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (sdata[threadIdx.x + stride] > sdata[threadIdx.x]) {
                sdata[threadIdx.x] = sdata[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    float row_max = sdata[0];

    // Pass 2: exponentiate after subtracting row_max. Each thread also
    // accumulates a partial sum to feed the next reduction.
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = expf(row_data[i] - row_max);
        row_data[i] = val;
        thread_sum += val;
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Tree reduction: pairwise sum-merge to produce the row total.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];

    // Pass 3: divide every element by the row sum. After this pass each
    // row is a valid probability distribution that sums to 1.
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_data[i] /= row_sum;
    }
}

// Slice one head out of a packed Q (or V) tensor. The full Q is laid out
// row-major as [rows, num_heads * head_dim]; this gather copies the
// head_dim-wide column slab starting at `head_offset` for every row,
// producing a contiguous [rows, head_dim] tensor that the per-head
// matmul can consume directly without any pointer arithmetic gymnastics.
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

// Same gather but materializes K^T (transpose) on the way out: it reads
// K's [rows, head_dim] slice and writes [head_dim, rows]. This gives the
// score GEMM Q_i * K_g^T contiguous row-major inputs without a separate
// transposed-B matmul kernel.
//
// Tiled with shared memory: both the source loads (consecutive `c`) and
// the destination stores (consecutive `r` on the transposed side) are
// coalesced. The +1 column padding staggers shared-memory addresses so
// the transpose store pattern avoids same-bank columns.
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

// Inverse of gather_head_kernel: place a per-head [rows, head_dim]
// result back into the packed [rows, dst_stride] output tensor at
// `head_offset`. After all 32 heads are scattered, the packed tensor
// holds the concatenated O = concat(O_0, ..., O_{h-1}) ready for the
// output projection W_O.
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

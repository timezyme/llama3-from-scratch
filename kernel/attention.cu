// ============================================================================
// attention.cu — Building-block kernels for Llama 3 Grouped Query Attention.
// ============================================================================
//
// What it does: provides the small kernels that compose one head of GQA
// (Grouped Query Attention) inside the forward pass. The big matmuls in
// the middle (Q * K^T and softmax(...) * V) are not here — those go
// through the GEMM (general matrix multiply) kernels in matmul.cu.
//
// Per-head attention flow that src/inference_layer.cu composes from
// these pieces (see run_attention_heads in that file):
//
//   gather Q_i,  gather V_g,  gather K_g^T   (kernels in this file)
//          |            |             |
//          v            v             v
//   matmul Q_i * K_g^T          -> S       (matmul.cu)
//   scale S by 1/sqrt(h_d)                 (this file)
//   causal_mask S                          (this file)
//   softmax S row-wise                     (this file)
//   matmul S * V_g              -> O_i     (matmul.cu)
//   scatter O_i into packed output         (this file)
//
// Read the file top-to-bottom — the layout matches execution order:
//   Section 1: Small helper (CUDA error wrap).
//   Section 2: Score-matrix transforms — scale, causal_mask, softmax.
//              Always applied to S in this order before the S * V matmul.
//   Section 3: Per-head data movement — gather (Q/V), gather-transpose
//              (produces K^T directly), scatter (per-head O back into O).
//   Section 4: Host entry points (one per kernel above).
//
// Credit:
//   - Scaled dot-product attention: Vaswani et al., "Attention Is All
//     You Need" (arXiv:1706.03762, 2017).
//   - Grouped Query Attention (GQA): Ainslie, Lee-Thorp, de Jong,
//     Zemlyanskiy, Lebrón, Sanghai, "GQA: Training Generalized
//     Multi-Query Transformer Models from Multi-Head Checkpoints"
//     (arXiv:2305.13245, 2023).
//   - Numerically stable softmax (max-subtract trick): standard
//     numerical-computing technique; modern parallel reference is
//     Milakov & Gimelshein, "Online normalizer calculation for softmax"
//     (arXiv:1805.02867, NVIDIA, 2018). Also covered in PMPP Chapter 11.
//   - Tree reduction inside softmax (halve active threads each step):
//     Mark Harris, "Optimizing Parallel Reduction in CUDA" (NVIDIA, 2007).
//   - Tiled transpose with +1 shared-memory padding for bank-conflict
//     avoidance (gather_head_transpose_kernel): Mark Harris, "An Efficient
//     Matrix Transpose in CUDA C/C++" (NVIDIA developer blog, 2013);
//     reproduced in the cuda-samples `transpose` example.
//
// Common pitfalls (design choices below that prevent them):
//   - SCALE by 1/sqrt(h_d) before softmax. Without scaling, raw dot
//     products grow as O(sqrt(h_d)) and push softmax into saturation.
//   - CAUSAL MASK on every (p, q) with q > p across the FULL s * s score
//     matrix — not just the diagonal or the last row. We use -1e6 as the
//     finite "negative-infinity" sentinel (safer than literal -inf for
//     downstream arithmetic).
//   - NUMERICALLY STABLE SOFTMAX: subtract the per-row max BEFORE exp().
//     Without this, exp(score) overflows to +inf for moderately large
//     scores and the kernel silently produces NaN logits.
//   - GATHER/SCATTER stay on the GPU. They slice and reassemble the
//     [rows, h * h_d] packed layout per-head without ever round-tripping
//     through host memory. The transposed gather variant materializes K^T
//     directly so the score GEMM can use the standard tiled kernel.
//
// Glossary:
//   GQA — Grouped Query Attention: more Q heads than K/V heads. Llama 3
//         8B has 32 Q heads and 8 K/V heads (group size 4).
//   h_d — head dimension. 4096 / 32 = 128 for Llama 3 8B.
//   s   — sequence length (rows of Q, K, V in this layer's input).
//   S   — score matrix Q * K^T (also gets scaled, masked, softmaxed).
//   O_i — per-head attention output; concatenated across heads into O.
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
// Section 2 — Score-matrix transforms. Applied to S = Q * K^T in this
// fixed order before the S * V matmul:
//   (2a) scale_kernel       — multiply by 1/sqrt(h_d)
//   (2b) causal_mask_kernel — write -1e6 above the diagonal
//   (2c) softmax_kernel     — numerically stable, row-wise, in place
// ============================================================================

// ---- (2a) scale_kernel ------------------------------------------------------
// Scale every element of S by a constant. Used to apply the 1/sqrt(h_d)
// factor to the raw Q * K^T scores so softmax doesn't saturate.
__global__ void scale_kernel(float *__restrict__ data, int count,
                             float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] *= scale;
    }
}

// ---- (2b) causal_mask_kernel -----------------------------------------------
// Causal mask for a square score matrix S[s, s]: for every (row, col)
// with col > row, write -1e6. After softmax those positions become
// effectively zero, so each query at position p only attends to itself
// and earlier tokens — the autoregressive constraint that makes the
// model unable to "see the future" while training and decoding.
//
// Why -1e6 instead of literal -inf: this is the conventional choice in
// Llama-family reference implementations. exp(-1e6) underflows to
// exactly 0.0 in FP32 (well below FLT_MIN ~ 1.18e-38), so the masked
// positions still vanish from the softmax output — and finite
// arithmetic dodges any compiler fast-math edge cases that mis-handle
// inf operands.
__global__ void causal_mask_kernel(float *__restrict__ S, int s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= s * s) return;
    int row = idx / s;
    int col = idx % s;
    if (col > row) {
        S[idx] = -1e6f;
    }
}

// ---- (2c) softmax_kernel ---------------------------------------------------
// Numerically stable softmax, in place, one block per row.
//
// Three-pass design — the order is forced by the math:
//   Pass 1: find the row maximum via tree reduction in shared memory.
//   Pass 2: exponentiate (val - row_max). The max-subtraction is the
//           "stable" part: without it, exp(score) overflows to +inf for
//           moderately large scores and the kernel silently produces NaN
//           logits. Skipping this is a classic correctness bug.
//   Pass 3: reduce the exponentiated row to a sum, then divide every
//           element by that sum.
//
// Mathematically equivalent to the naive exp(S) / sum(exp(S)) because
// subtracting a constant from every element of softmax leaves the
// result unchanged: e^(x-c) / sum(e^(x-c)) = e^x / sum(e^x).
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

// ============================================================================
// Section 3 — Per-head data movement. Three kernels move per-head slabs
// in and out of the packed [rows, h * h_d] tensors so the per-head
// matmuls in matmul.cu always see contiguous, simply-shaped inputs:
//   (3a) gather_head_kernel             — slice Q_i or V_g out of pack
//   (3b) gather_head_transpose_kernel   — slice K_g AND transpose to K^T
//   (3c) scatter_head_kernel            — write per-head O_i back into pack
// All three stay on the GPU; no host round-trips.
// ============================================================================

// ---- (3a) gather_head_kernel ------------------------------------------------
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

// ---- (3b) gather_head_transpose_kernel -------------------------------------
// Same as (3a), but materializes K^T (transpose) on the way out: it reads
// K's [rows, head_dim] slice and writes [head_dim, rows]. This gives the
// score GEMM Q_i * K_g^T contiguous row-major inputs without needing a
// separate "transposed-B" matmul kernel.
//
// Tiled with shared memory: both the source loads (consecutive `c`) and
// the destination stores (consecutive `r` on the transposed side) are
// coalesced. The +1 column padding staggers shared-memory addresses so
// the transpose-store pattern avoids same-bank columns (32-way bank
// conflicts would otherwise serialize the writes).
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

// ---- (3c) scatter_head_kernel ----------------------------------------------
// Inverse of gather_head_kernel: place a per-head [rows, head_dim]
// result back into the packed [rows, dst_stride] output tensor at
// `head_offset`. After all 32 heads have been scattered, the packed
// tensor holds the concatenated O = concat(O_0, ..., O_{h-1}) ready
// for the output projection W_O in matmul.cu.
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

// ============================================================================
// Section 4 — Host entry points. One thin wrapper per kernel above. Each
// computes the launch grid, dispatches the kernel, and surfaces CUDA
// errors via CUDA_CHECK. Order in this section matches Sections 2-3:
// score-matrix transforms first, then per-head data movement.
// ============================================================================

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

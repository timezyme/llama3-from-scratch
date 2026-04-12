// Rotary Position Embedding (RoPE) CUDA kernel for Llama 3.
//
// Applies positional encodings in-place to Q and K tensors.
// Uses the "rotate_full" convention: pairs dimension i with i + h_d/2.
// Base frequency: 500000 (Llama 3), NOT 10000 (original RoPE paper).
//
// For head vector q of dimension h_d:
//   theta_i = 1 / (base ^ (2*i / h_d))  for i in [0, h_d/2)
//   For position p:
//     q'[i]         = q[i] * cos(p * theta_i) - q[i + h_d/2] * sin(p * theta_i)
//     q'[i + h_d/2] = q[i] * sin(p * theta_i) + q[i + h_d/2] * cos(p * theta_i)

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

// RoPE kernel: applies rotary embeddings in-place.
// x: flat projected tensor [s, num_heads * head_dim], row-major.
// cos_table, sin_table: precomputed [s, head_dim/2].
// Each thread handles one (position, head, pair_index) triple.
__global__ void rope_kernel(float *__restrict__ x,
                            const float *__restrict__ cos_table,
                            const float *__restrict__ sin_table,
                            int seq_len, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_hd = head_dim / 2;
    int total = seq_len * num_heads * half_hd;
    if (idx >= total) return;

    // Decompose flat index -> (position, head, pair_index)
    int pair_idx = idx % half_hd;
    int tmp = idx / half_hd;
    int head = tmp % num_heads;
    int pos = tmp / num_heads;

    // Row stride in the flat tensor
    int row_stride = num_heads * head_dim;
    int base = pos * row_stride + head * head_dim;

    int i_first = base + pair_idx;
    int i_second = base + pair_idx + half_hd;

    // Cos/sin table indexed by [position, pair_index]
    int cs_idx = pos * half_hd + pair_idx;
    float c = cos_table[cs_idx];
    float s = sin_table[cs_idx];

    float q_first = x[i_first];
    float q_second = x[i_second];

    x[i_first]  = q_first * c - q_second * s;
    x[i_second] = q_first * s + q_second * c;
}

void gpu_rope(float *d_x, const float *d_cos, const float *d_sin,
              int seq_len, int num_heads, int head_dim) {
    int half_hd = head_dim / 2;
    int total = seq_len * num_heads * half_hd;
    if (total <= 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    rope_kernel<<<blocks, threads>>>(d_x, d_cos, d_sin,
                                     seq_len, num_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Precompute cos/sin tables on the host.
// cos_table[p * half_hd + i] = cos(p * theta_i)
// sin_table[p * half_hd + i] = sin(p * theta_i)
// where theta_i = 1.0 / pow(base, 2*i / head_dim)
void precompute_rope_table(float *cos_out, float *sin_out,
                           int seq_len, int head_dim, float base) {
    int half_hd = head_dim / 2;
    for (int p = 0; p < seq_len; ++p) {
        for (int i = 0; i < half_hd; ++i) {
            float theta = 1.0f / std::pow(base, 2.0f * i / head_dim);
            float angle = static_cast<float>(p) * theta;
            cos_out[p * half_hd + i] = std::cos(angle);
            sin_out[p * half_hd + i] = std::sin(angle);
        }
    }
}

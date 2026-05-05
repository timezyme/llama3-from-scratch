// Rotary Position Embedding (RoPE) for Llama 3 Q/K heads.
// llm_part2 §4 has two required details: base is 500000, and pair i
// rotates with i + h_d/2 rather than even/odd neighbors. Cos/sin are
// precomputed as [seq_len, h_d/2] tables before the forward pass.

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

// rope_kernel — apply rotary embeddings in-place to x.
//
// Layout: x is row-major [seq_len, num_heads, head_dim]. For batched
// generation seq_len = batch * q_seq and the RoPE position resets at the
// start of each batch slot (row % q_seq). One thread handles exactly one
// (row, head, pair_index) triple — fully parallel, no atomics, no syncs.
//
// Why we rotate the (i, i+h_d/2) pair instead of (2i, 2i+1): this matches
// the HuggingFace `rotate_half` convention used by the Llama 3 reference
// implementation. Read x[i] and x[i+h_d/2] into a 2-vector, multiply by
// the 2x2 rotation matrix, write back. In place is safe because both
// stores depend only on values already loaded into local registers.
__global__ void rope_kernel(float *__restrict__ x,
                            const float *__restrict__ cos_table,
                            const float *__restrict__ sin_table,
                            int seq_len, int q_seq, int num_heads,
                            int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_hd = head_dim / 2;
    int total = seq_len * num_heads * half_hd;
    if (idx >= total) return;

    // Unpack the flat thread index into (row, head, pair_idx).
    // pos = row % q_seq lets one launch handle multiple batch slots in
    // a single tensor — the per-batch position counter resets every
    // q_seq rows so each prompt sees positions 0..q_seq-1.
    int pair_idx = idx % half_hd;
    int tmp = idx / half_hd;
    int head = tmp % num_heads;
    int row = tmp / num_heads;
    int pos = row % q_seq;

    int row_stride = num_heads * head_dim;
    int base = row * row_stride + head * head_dim;

    int i_first = base + pair_idx;             // q[i]
    int i_second = base + pair_idx + half_hd;  // q[i + h_d/2] — rotate-half pair

    // cos_table/sin_table are laid out [position, pair_index].
    int cs_idx = pos * half_hd + pair_idx;
    float c = cos_table[cs_idx];
    float s = sin_table[cs_idx];

    float q_first = x[i_first];
    float q_second = x[i_second];

    // 2x2 rotation: [c -s; s c] * [q_first; q_second].
    x[i_first]  = q_first * c - q_second * s;
    x[i_second] = q_first * s + q_second * c;
}

// Host launcher for rope_kernel. q_seq < 0 is shorthand for "non-batched",
// i.e. q_seq == seq_len so the position counter never wraps.
void gpu_rope(float *d_x, const float *d_cos, const float *d_sin,
              int seq_len, int num_heads, int head_dim, int q_seq) {
    int half_hd = head_dim / 2;
    int total = seq_len * num_heads * half_hd;
    if (total <= 0) return;
    int actual_q_seq = (q_seq < 0) ? seq_len : q_seq;
    if (actual_q_seq <= 0) {
        throw std::runtime_error("gpu_rope: q_seq must be positive");
    }

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    rope_kernel<<<blocks, threads>>>(d_x, d_cos, d_sin,
                                     seq_len, actual_q_seq, num_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Build cos/sin tables on the host. Sized [seq_len, h_d/2]; `base` is
// passed in (always ROPE_BASE = 500000 in this project) so this stays
// reusable for any RoPE-style model. Called once per inference pass and
// uploaded to VRAM, after which the device kernel is a pure table read.
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

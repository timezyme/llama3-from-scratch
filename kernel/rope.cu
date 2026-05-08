// ============================================================================
// rope.cu — Rotary Position Embedding (RoPE) for Llama 3 Q/K heads.
// ============================================================================
//
// What it does: rotates each Q (or K) vector by an angle that depends on
// the token's position in the sequence. Encodes "where in the prompt is
// this token" directly into the activations, so attention sees position
// naturally — no separate learned position embeddings required.
//
// Per-pair formula (i ranges over [0, h_d/2)):
//   theta = 1 / base^(2i / h_d)
//   angle = position * theta
//   x_new[i]        =  x[i] * cos(angle) - x[i + h_d/2] * sin(angle)
//   x_new[i+h_d/2]  =  x[i] * sin(angle) + x[i + h_d/2] * cos(angle)
//
// This is a 2x2 rotation [c -s; s c] applied to every (i, i + h_d/2) pair.
//
// Where this kernel runs in the forward pass:
//   Twice per decoder block — once on Q, once on K — immediately after
//   the per-head gather and before the score-matrix matmul Q * K^T.
//   V is NOT rotated (positional information flows through Q and K only).
//
// Read the file top-to-bottom — the layout matches execution order:
//   Section 1: Small helper (CUDA error wrap).
//   Section 2: rope_kernel              — apply rotation in place on device.
//   Section 3: gpu_rope                 — host entry for the device kernel.
//   Section 4: precompute_rope_table    — build cos/sin tables on the host
//                                         (called once per inference pass,
//                                          uploaded to VRAM after).
//
// Credit:
//   - Algorithm: Su, Lu, Pan, Murtadha, Wen, Liu, "RoFormer: Enhanced
//     Transformer with Rotary Position Embedding" (arXiv:2104.09864, 2021).
//   - Llama 3 specifics (base = 500000 instead of 10000; rotate-half
//     pairing of (i, i + h_d/2) instead of (2i, 2i + 1)): Meta AI Llama 3
//     release; HuggingFace transformers reference implementation in
//     `modeling_llama.py`.
//
// Common pitfalls (failure modes the Credit-cited conventions prevent):
//   - Wrong base (10000 from the paper instead of 500000): works fine on
//     short prompts — the bug only surfaces on long-context eval.
//   - Wrong pairing (interleaved (2i, 2i+1) instead of rotate-half):
//     outputs look plausible but token logits are subtly wrong.
//   Both bugs are silent at first glance, which is why this kernel ships
//   them as project-wide constants instead of leaving them at call sites.
//
// Glossary:
//   h_d      — head dimension (128 for Llama 3 8B).
//   half_hd  — h_d / 2 = number of rotation pairs per head.
//   q_seq    — per-batch-slot sequence length. Lets one kernel launch
//              handle multiple prompts whose positions reset every q_seq
//              rows (so each prompt sees positions 0..q_seq-1).
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
// Section 2 — rope_kernel: apply rotation in place on x.
// ============================================================================
//
// Layout: x is row-major [seq_len, num_heads, head_dim]. In batched
// generation seq_len = batch * q_seq and the RoPE position resets at the
// start of each batch slot — that's what the `row % q_seq` line below
// handles, so one launch can process multiple prompts whose positions
// each restart at 0.
//
// Parallelism: one thread handles exactly one (row, head, pair_index)
// triple. Fully data-parallel — no atomics, no shared memory, no
// __syncthreads. The kernel is bandwidth-bound, not compute-bound.
//
// In-place safety: both stores below depend only on values already loaded
// into local registers (q_first, q_second), so writing x[i_first] and
// x[i_second] in place can't race with the reads.
//
// (Pairing and base conventions are documented in the Credit and Common
// pitfalls blocks at the top of this file — not repeated here.)
// ============================================================================
__global__ void rope_kernel(float *__restrict__ x,
                            const float *__restrict__ cos_table,
                            const float *__restrict__ sin_table,
                            int seq_len, int q_seq, int num_heads,
                            int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_hd = head_dim / 2;
    int total = seq_len * num_heads * half_hd;
    if (idx >= total) return;

    // How the kernel works:
    // Unpack the flat thread index into (row, head, pair_idx). The
    // `row % q_seq` step is what resets the position counter at each
    // batch-slot boundary (see Section 2 banner above for why).
    int pair_idx = idx % half_hd;
    int tmp = idx / half_hd;
    int head = tmp % num_heads;
    int row = tmp / num_heads;
    // computes position
    int pos = row % q_seq;

    int row_stride = num_heads * head_dim;
    int base = row * row_stride + head * head_dim;

    int i_first = base + pair_idx;             // q[i]
    int i_second = base + pair_idx + half_hd;  // q[i + h_d/2] — rotate-half pair

    // Looks up precomputed cos/sin from the table
    int cs_idx = pos * half_hd + pair_idx;
    float c = cos_table[cs_idx];
    float s = sin_table[cs_idx];

    float q_first = x[i_first];
    float q_second = x[i_second];

    // applies the 2D rotation
    x[i_first]  = q_first * c - q_second * s;
    x[i_second] = q_first * s + q_second * c;
}

// ============================================================================
// Section 3 — gpu_rope: host entry for the device kernel.
// ============================================================================
//
// Thin wrapper: pick block / grid shape, dispatch the kernel, surface
// CUDA errors. q_seq < 0 is shorthand for "non-batched" — set q_seq to
// seq_len so the position counter never wraps.
// ============================================================================
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

// ============================================================================
// Section 4 — precompute_rope_table: host-side table builder.
// ============================================================================
//
// Build cos/sin tables on the host, sized [seq_len, h_d/2]. `base` is a
// parameter (always ROPE_BASE = 500000 for this project) so the function
// stays reusable for any RoPE-style model. Called once per inference
// pass and uploaded to VRAM (video RAM); after upload, rope_kernel just
// reads from the precomputed tables instead of recomputing trig per call.
//
// Why precompute on the host instead of on-device: the table is small
// (seq_len * h_d/2 floats), built once, and reused thousands of times by
// the kernel. Host-side trig is plenty fast for a one-shot setup cost,
// and keeping it off-device avoids a special-case kernel launch.
// ============================================================================
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

// Per-step decoder block: runs all 32 Llama 3 layers for one
// prefill or decode pass and returns the final-RMSNormed hidden state
// for the last token in each batch slot.
//
// This file owns the hot-path kernel-launch helpers
// (`run_attention_heads`, `compute_lm_head_logits`) and the giant
// `forward_step` orchestrator that strings them together with
// streaming/resident weight uploads, RoPE rotation, GQA attention, and
// the SwiGLU FFN.

#include "config.h"
#include "device_weights.h"
#include "inference_internal.h"
#include "instrument.h"
#include "kernel/kernels.cuh"
#include "kv_cache.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

// Wrap a CUDA runtime call and turn failures into C++ exceptions. This keeps
// allocation/copy call sites readable while still surfacing the CUDA error
// string immediately. Keep the trailing backslashes: they make the do/while
// block part of the macro expansion.
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error: ") +          \
                                     cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)

namespace {

// Per-head scratch device buffers. Sized once per forward_step call
// (q_seq and kv_seq are constant across all 32 layers and the entire
// per-head loop) and reused for every (layer, batch, head) iteration.
// Without this reuse, every layer/head/batch iteration would allocate and
// free five tiny buffers before doing useful attention work.
struct AttentionScratch {
    float *d_Qi = nullptr;  // [q_seq, HEAD_DIM]
    float *d_KgT = nullptr; // [HEAD_DIM, kv_seq]  (transposed gather)
    float *d_Vg = nullptr;  // [kv_seq, HEAD_DIM]
    float *d_S = nullptr;   // [q_seq, kv_seq]
    float *d_Oi = nullptr;  // [q_seq, HEAD_DIM]
};

// Run GQA (Grouped Query Attention) for one batch slot, all 32 heads.
//
// This implements the "loop over all h=32 query heads in your C++
// controller, launching CUDA kernels for each head's score
// computation, mask application, softmax, and weighted sum" approach
// suggested in llm_part2 §3.2.
//
// Shapes (per batch slot):
//   d_Q_b:    [q_seq, NUM_HEADS * HEAD_DIM]      = [q_seq, 32 * 128]
//   d_K_b:    [kv_seq, NUM_KV_HEADS * HEAD_DIM]  = [kv_seq, 8 * 128] (cache)
//   d_V_b:    [kv_seq, NUM_KV_HEADS * HEAD_DIM]  = [kv_seq, 8 * 128] (cache)
//   d_attn_b: [q_seq, NUM_HEADS * HEAD_DIM]      = output, head-stitched
//
// GQA detail (llm_part2 §3.1): there are only 8 distinct K/V heads
// shared across the 32 query heads. Each KV head services NUM_HEADS /
// NUM_KV_HEADS = 4 consecutive query heads, so query head `hi` reads
// from KV head `g = hi / 4` (matches the assignment formula
// g = floor(i / (h/h_k))). Compared to standard MHA (Multi-Head
// Attention) this reduces K/V memory by 4x while keeping query
// expressiveness intact.
//
// llm_part2 §3.1 describes Q/K/V as logical per-head views. This code
// materializes each head into contiguous buffers so the existing GEMM
// (general matrix multiply) path can consume row-major operands. The
// tradeoff is extra HBM (high-bandwidth memory) traffic instead of a
// separate strided-input matmul.
//
// Causal-mask gating: we skip the mask whenever q_seq == 1 (decode
// step) because the single new query attends to every cached position
// up to and including itself, none of which is "in the future".
// During prefill (q_seq == kv_seq > 1) the mask is required so each
// position only attends to itself and earlier tokens. We also skip
// when q_seq == 1 == kv_seq (the very first prefill of a 1-token prompt
// — there is nothing to mask).
void run_attention_heads(const float *d_Q_b, const float *d_K_b,
                         const float *d_V_b, float *d_attn_b,
                         const AttentionScratch &scratch, int q_seq,
                         int kv_seq) {
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const int q_stride = EMBEDDING_DIM; // NUM_HEADS * HEAD_DIM
    const int heads_per_group = NUM_HEADS / NUM_KV_HEADS;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    const bool apply_mask = (q_seq == kv_seq && q_seq > 1);

    for (int hi = 0; hi < NUM_HEADS; ++hi) {
        // GQA mapping: this query head pulls from KV head `kvg`.
        int kvg = hi / heads_per_group;

        // Pull per-head slices straight out of device memory. The K
        // gather also transposes (gives us K_g^T) so the next matmul is
        // a straight row-major GEMM.
        gpu_gather_head(d_Q_b, scratch.d_Qi, q_seq, HEAD_DIM, q_stride,
                        hi * HEAD_DIM);
        gpu_gather_head_transpose(d_K_b, scratch.d_KgT, kv_seq, HEAD_DIM,
                                  kv_dim, kvg * HEAD_DIM);
        gpu_gather_head(d_V_b, scratch.d_Vg, kv_seq, HEAD_DIM, kv_dim,
                        kvg * HEAD_DIM);

        // Score matrix S = Q_i * K_g^T, then scale by 1/sqrt(h_d).
        gpu_matmul_device(scratch.d_Qi, scratch.d_KgT, scratch.d_S, q_seq,
                          HEAD_DIM, kv_seq);
        gpu_scale(scratch.d_S, q_seq * kv_seq, scale);
        if (apply_mask) {
            gpu_causal_mask(scratch.d_S, q_seq);
        }
        // Numerically stable softmax (kernel does max-subtraction).
        gpu_softmax(scratch.d_S, q_seq, kv_seq);
        // Attended output O_i = softmax(S) * V_g.
        gpu_matmul_device(scratch.d_S, scratch.d_Vg, scratch.d_Oi, q_seq,
                          kv_seq, HEAD_DIM);

        // Stitch this head's [q_seq, HEAD_DIM] output back into the
        // packed [q_seq, NUM_HEADS * HEAD_DIM] tensor at hi * HEAD_DIM.
        gpu_scatter_head(scratch.d_Oi, d_attn_b, q_seq, HEAD_DIM, q_stride,
                         hi * HEAD_DIM);
    }
}

} // namespace

// Project the last hidden vector through the language model head to get
// logits over the full vocabulary (V = 128256).
//
// Math: logits = lm_head @ x_last, where lm_head is [V, d] (loaded
// row-major as the HuggingFace checkpoint stores it) and x_last is the
// last token's [d]-dim hidden state after the final RMSNorm. This is
// effectively a (V x d) * (d x 1) matrix-vector product.
//
// Why CPU and not GPU: this project keeps lm_head as a host tensor and
// only needs one [d] vector per batch slot. A dedicated GEMV (general
// matrix-vector multiply) kernel is an optional optimization, not a
// requirement in llm_part2 §3.2.
std::vector<float> compute_lm_head_logits(const float *lm_head,
                                          const float *h_x_last) {
    std::vector<float> logits(VOCAB_SIZE);
    for (int v = 0; v < VOCAB_SIZE; ++v) {
        float dot = 0.0f;
        const float *row = lm_head + (size_t)v * EMBEDDING_DIM;
        for (int j = 0; j < EMBEDDING_DIM; ++j)
            dot += h_x_last[j] * row[j];
        logits[v] = dot;
    }
    return logits;
}

// One forward step through all 32 decoder blocks. Used both for prefill
// (q_seq = prompt length) and for each decode iteration (q_seq = 1).
//
// Inputs:
//   h_input    [batch, q_seq, EMBEDDING_DIM] — host-side embeddings to push.
//   q_seq      number of new tokens being processed this step.
//   weights    streaming-from-disk model weights (used when resident is null).
//   cache      device-side per-layer K/V tensors. cache.len() is the number
//              of tokens already cached; advances by q_seq at the end.
//   d_cos_full,d_sin_full  full S_MAX-sized RoPE tables on the device. We
//              advance into them by `len_before * (h_d/2)` so this step
//              sees positions [len_before, len_before+q_seq).
//   resident_weights  if non-null, bypass the H2D upload and use BF16
//              tensors that already live in VRAM (video RAM).
//   batch      number of independent prompts processed in lockstep.
//
// Returns: the final-RMSNormed hidden state for the LAST token in each
// batch slot, [batch, EMBEDDING_DIM]. The caller projects this through
// lm_head to get logits.
std::vector<float> forward_step(const float *h_input, int q_seq,
                                ModelWeights &weights, KVCache &cache,
                                const float *d_cos_full,
                                const float *d_sin_full,
                                DeviceModelWeights *resident_weights,
                                int batch) {
    Stopwatch sw_step(q_seq == 1 ? "step.decode" : "step.prefill");
    const int d = EMBEDDING_DIM;
    const int kv_dim = KVCache::kv_dim();
    const int half_hd = HEAD_DIM / 2;
    const int rows = batch * q_seq;

    const int len_before = cache.len();
    const int kv_seq = len_before + q_seq;

    if (q_seq <= 0) {
        throw std::runtime_error("forward_step: q_seq must be positive");
    }
    if (batch <= 0 || batch > cache.batch()) {
        throw std::runtime_error("forward_step: invalid batch size");
    }
    if (kv_seq > cache.max_len()) {
        throw std::runtime_error("forward_step: kv_seq exceeds cache capacity");
    }

    const size_t bytes_X = static_cast<size_t>(rows) * d * sizeof(float);
    const size_t bytes_ffn = static_cast<size_t>(rows) * FFN_DIM *
                              sizeof(float);

    // Per-step device buffers. Allocated once at the top of forward_step
    // and reused across all 32 decoder blocks. The weight pointers
    // (d_w*) are only allocated on the streaming-from-disk path; on the
    // resident path the layer's BF16 tensors already live in VRAM and
    // we just take pointers into them.
    float *d_X = nullptr, *d_Xnorm = nullptr;
    float *d_Q = nullptr;
    float *d_gamma = nullptr;
    float *d_wq = nullptr, *d_wk = nullptr, *d_wv = nullptr, *d_wo = nullptr;
    float *d_wgate = nullptr, *d_wup = nullptr, *d_wdown = nullptr;
    float *d_attn = nullptr, *d_attn_out = nullptr;
    float *d_gate = nullptr, *d_up = nullptr, *d_ffn = nullptr;

    CUDA_CHECK(cudaMalloc(&d_X, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_Xnorm, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_Q, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_gamma, d * sizeof(float)));
    if (resident_weights == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_wq, (size_t)d * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_wk, (size_t)d * kv_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_wv, (size_t)d * kv_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_wo, (size_t)d * d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_wgate, (size_t)d * FFN_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_wup, (size_t)d * FFN_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_wdown, (size_t)FFN_DIM * d * sizeof(float)));
    }
    CUDA_CHECK(cudaMalloc(&d_attn, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_attn_out, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_gate, bytes_ffn));
    CUDA_CHECK(cudaMalloc(&d_up, bytes_ffn));
    CUDA_CHECK(cudaMalloc(&d_ffn, bytes_X));

    // Per-head attention scratch. Allocated once with q_seq and kv_seq
    // sizes, then reused 32 (layers) * batch * 32 (heads) times this step.
    AttentionScratch scratch;
    CUDA_CHECK(cudaMalloc(&scratch.d_Qi,
                          static_cast<size_t>(q_seq) * HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scratch.d_KgT,
                          static_cast<size_t>(HEAD_DIM) * kv_seq * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scratch.d_Vg,
                          static_cast<size_t>(kv_seq) * HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scratch.d_S,
                          static_cast<size_t>(q_seq) * kv_seq * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&scratch.d_Oi,
                          static_cast<size_t>(q_seq) * HEAD_DIM * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, h_input, bytes_X, cudaMemcpyHostToDevice));

    // Slice into the precomputed RoPE table so this step sees the
    // correct positions. After cache.len() = len_before tokens, the
    // new tokens occupy positions [len_before, len_before+q_seq).
    const float *d_cos_step = d_cos_full + (size_t)len_before * half_hd;
    const float *d_sin_step = d_sin_full + (size_t)len_before * half_hd;

    // ----- 32x decoder block loop -----
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        Stopwatch sw_layer("layer.total");
        const LayerWeights *lw = nullptr;
        const DeviceLayerWeights *resident_lw = nullptr;

        // Pick the weight source for this layer. Resident path: BF16
        // tensors already in VRAM (pay once per process). Streaming
        // path: read FP32 weights from disk into host RAM, then upload.
        if (resident_weights != nullptr) {
            resident_lw = &resident_weights->load_layer(layer);
        } else {
            Stopwatch sw_load("layer.load_disk_to_host");
            lw = &weights.load_layer(layer);
        }

        // Streaming path only: copy this layer's host weights to VRAM.
        // Resident path skips this — its weights are already there.
        if (resident_weights == nullptr) {
            Stopwatch sw_h2d("layer.h2d_weights");
            CUDA_CHECK(cudaMemcpy(d_gamma, lw->input_layernorm,
                                   d * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_wq, lw->q_proj,
                                   (size_t)d * d * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_wk, lw->k_proj,
                                   (size_t)d * kv_dim * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_wv, lw->v_proj,
                                   (size_t)d * kv_dim * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_wo, lw->o_proj,
                                   (size_t)d * d * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_wgate, lw->gate_proj,
                                   (size_t)d * FFN_DIM * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_wup, lw->up_proj,
                                   (size_t)d * FFN_DIM * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_wdown, lw->down_proj,
                                   (size_t)FFN_DIM * d * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // ===== Attention sub-block: pre-attention norm, QKV, RoPE =====
        {
            Stopwatch sw("layer.attn_pre");
            // RMSNorm with this layer's `input_layernorm.weight` (gamma).
            // Note: each decoder block has TWO distinct gamma vectors —
            // input_layernorm before attention and post_attention_layernorm
            // before FFN. Mixing them up is a pitfall flagged in llm_part2.
            const float *input_norm =
                resident_lw != nullptr ? resident_lw->input_layernorm : d_gamma;
            gpu_rmsnorm(d_X, input_norm, d_Xnorm, rows, d,
                        RMS_NORM_EPSILON);
            // Q is computed for the whole batch at once (rows = batch*q_seq)
            // because every row independently produces an output Q row.
            // K and V instead must be written into per-batch cache slices
            // at [b, len_before:len_before+q_seq, :], so we issue one
            // matmul per batch slot. This batched v1 layout is less efficient
            // than a strided 3D matmul.
            if (resident_lw != nullptr) {
                gpu_matmul_device_bf16_weight(d_Xnorm, resident_lw->q_proj,
                                              d_Q, rows, d, d);
                for (int b = 0; b < batch; ++b) {
                    const float *d_Xnorm_b =
                        d_Xnorm + static_cast<size_t>(b) * q_seq * d;
                    gpu_matmul_device_bf16_weight(
                        d_Xnorm_b, resident_lw->k_proj,
                        cache.k_at(layer, len_before, b), q_seq, d, kv_dim);
                    gpu_matmul_device_bf16_weight(
                        d_Xnorm_b, resident_lw->v_proj,
                        cache.v_at(layer, len_before, b), q_seq, d, kv_dim);
                }
            } else {
                gpu_matmul_device(d_Xnorm, d_wq, d_Q, rows, d, d);
                for (int b = 0; b < batch; ++b) {
                    const float *d_Xnorm_b =
                        d_Xnorm + static_cast<size_t>(b) * q_seq * d;
                    gpu_matmul_device(d_Xnorm_b, d_wk,
                                      cache.k_at(layer, len_before, b),
                                      q_seq, d, kv_dim);
                    gpu_matmul_device(d_Xnorm_b, d_wv,
                                      cache.v_at(layer, len_before, b),
                                      q_seq, d, kv_dim);
                }
            }
            // RoPE rotates Q and the new K rows in place. V is NOT
            // rotated (RoPE encodes the query/key dot-product geometry,
            // not the attended values). We must apply RoPE only to the
            // newly written K rows — older cached K already had RoPE
            // applied in earlier steps. K rotation runs per batch slot
            // because each slot's new K rows live in a different cache
            // region.
            gpu_rope(d_Q, d_cos_step, d_sin_step, rows, NUM_HEADS, HEAD_DIM,
                     q_seq);
            for (int b = 0; b < batch; ++b) {
                gpu_rope(cache.k_at(layer, len_before, b), d_cos_step,
                         d_sin_step, q_seq, NUM_KV_HEADS, HEAD_DIM, q_seq);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // ===== Attention heads: scaled dot-product per head, GQA-grouped =====
        // Q lives in d_Q; the K/V buffers come from the KV cache so we
        // see the full prefix [0, kv_seq) without recomputing it.
        // Everything stays on the GPU — the gather/scatter kernels slice
        // per-head views without crossing PCIe.
        {
            Stopwatch sw("layer.attn_heads");
            for (int b = 0; b < batch; ++b) {
                run_attention_heads(
                    d_Q + static_cast<size_t>(b) * q_seq * d,
                    cache.k_batch(layer, b), cache.v_batch(layer, b),
                    d_attn + static_cast<size_t>(b) * q_seq * d, scratch,
                    q_seq, kv_seq);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // ===== Post-attention: O proj, residual #1, RMSNorm, FFN, residual #2 =====
        {
            Stopwatch sw("layer.post_attn_and_ffn");
            // Output projection: attn_out = stitched_heads @ W_O^T, [s, d].
            if (resident_lw != nullptr) {
                gpu_matmul_device_bf16_weight(d_attn, resident_lw->o_proj,
                                              d_attn_out, rows, d, d);
            } else {
                gpu_matmul_device(d_attn, d_wo, d_attn_out, rows, d, d);
            }
            // Residual #1: X <- X + attn_out (in place on d_X).
            gpu_residual_add(d_X, d_attn_out, rows * d);
            if (resident_lw == nullptr) {
                CUDA_CHECK(cudaMemcpy(d_gamma, lw->post_attn_layernorm,
                                      d * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
            // Second RMSNorm uses the layer's `post_attention_layernorm`
            // gamma — distinct from the input_layernorm gamma above.
            const float *post_attn_norm =
                resident_lw != nullptr ? resident_lw->post_attn_layernorm
                                       : d_gamma;
            gpu_rmsnorm(d_X, post_attn_norm, d_Xnorm, rows, d,
                        RMS_NORM_EPSILON);
            if (resident_lw != nullptr) {
                gpu_matmul_device_bf16_weight(d_Xnorm, resident_lw->gate_proj,
                                              d_gate, rows, d, FFN_DIM);
                gpu_matmul_device_bf16_weight(d_Xnorm, resident_lw->up_proj,
                                              d_up, rows, d, FFN_DIM);
            } else {
                gpu_matmul_device(d_Xnorm, d_wgate, d_gate, rows, d, FFN_DIM);
                gpu_matmul_device(d_Xnorm, d_wup, d_up, rows, d, FFN_DIM);
            }
            // Fused SiLU(gate) * up, written back into d_gate (alias).
            // d_gate now holds H = SiLU(gate) * up, [s, FFN_DIM].
            gpu_swiglu(d_gate, d_up, d_gate, rows * FFN_DIM);
            // Down projection: ffn_out = H @ W_down^T, [s, d].
            if (resident_lw != nullptr) {
                gpu_matmul_device_bf16_weight(d_gate, resident_lw->down_proj,
                                              d_ffn, rows, FFN_DIM, d);
            } else {
                gpu_matmul_device(d_gate, d_wdown, d_ffn, rows, FFN_DIM, d);
            }
            // Residual #2: X <- X + ffn_out. End of decoder block.
            gpu_residual_add(d_X, d_ffn, rows * d);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        if (resident_weights == nullptr) {
            Stopwatch sw("layer.unload");
            weights.unload_layer(layer);
        }
    }

    // KV cache now records len_before + q_seq tokens for every layer.
    cache.advance(q_seq);

    // Final RMSNorm using `model.norm.weight` (gamma_final). Applied
    // once after all 32 blocks, before lm_head.
    CUDA_CHECK(cudaMemcpy(d_gamma, weights.global().final_norm,
                           d * sizeof(float), cudaMemcpyHostToDevice));
    gpu_rmsnorm(d_X, d_gamma, d_Xnorm, rows, d, RMS_NORM_EPSILON);

    // Greedy decoding only needs logits for the LAST row, so we copy
    // just that row back to host. Projecting the entire [s, d] matrix
    // through lm_head ([V=128256, d]) would be wasteful and risks VRAM
    // pressure for long sequences. (Pitfall: llm_part2 §4.)
    std::vector<float> last_hidden(static_cast<size_t>(batch) * d);
    for (int b = 0; b < batch; ++b) {
        const size_t last_row_offset =
            (static_cast<size_t>(b) * q_seq + (q_seq - 1)) * d;
        CUDA_CHECK(cudaMemcpy(last_hidden.data() + static_cast<size_t>(b) * d,
                              d_Xnorm + last_row_offset, d * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    cudaFree(d_X); cudaFree(d_Xnorm); cudaFree(d_Q);
    cudaFree(d_gamma);
    cudaFree(d_wq); cudaFree(d_wk); cudaFree(d_wv); cudaFree(d_wo);
    cudaFree(d_wgate); cudaFree(d_wup); cudaFree(d_wdown);
    cudaFree(d_attn); cudaFree(d_attn_out);
    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_ffn);
    cudaFree(scratch.d_Qi); cudaFree(scratch.d_KgT); cudaFree(scratch.d_Vg);
    cudaFree(scratch.d_S); cudaFree(scratch.d_Oi);

    return last_hidden;
}

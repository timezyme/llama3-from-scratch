// Inference pipeline for Llama 3 8B with optional KV cache for incremental
// autoregressive decoding.
//
// Two entry points:
//   - generate_next_token: single forward pass over the prompt; argmax token.
//   - generate_tokens:     prefill + decode loop using KVCache. Each decode
//                          step projects Q for one new token only, appends
//                          one K/V row to the cache, and attends over the
//                          full cached prefix.

#include "config.h"
#include "device_weights.h"
#include "inference.h"
#include "instrument.h"
#include "kernel/kernels.cuh"
#include "kv_cache.h"
#include "tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess)                                                \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(err));                  \
    } while (0)

namespace {

// Maximum sequence length the cache (and RoPE table) is sized for.
// Spec bounds the prompt at 1000 tokens; keep headroom for generation.
constexpr int S_MAX = 1024;

// Llama 3 Instruct chat-template special tokens (vocab IDs).
constexpr int BEGIN_OF_TEXT   = 128000;
constexpr int START_HEADER    = 128006;
constexpr int END_HEADER      = 128007;
constexpr int EOT_ID          = 128009;
constexpr int NEWLINE_NEWLINE = 271;   // "\n\n"
constexpr int USER_TOKEN      = 882;   // "user"
constexpr int ASSISTANT_TOKEN = 78191; // "assistant"

// Wrap a prompt in the Llama 3 Instruct chat template.
std::vector<int> apply_chat_template(const BPETokenizer &tok,
                                     const std::string &prompt) {
    auto encoded = tok.encode(prompt);
    std::vector<int> ids;
    ids.reserve(encoded.size() + 10);
    ids.push_back(BEGIN_OF_TEXT);
    ids.push_back(START_HEADER);
    ids.push_back(USER_TOKEN);
    ids.push_back(END_HEADER);
    ids.push_back(NEWLINE_NEWLINE);
    ids.insert(ids.end(), encoded.begin(), encoded.end());
    ids.push_back(EOT_ID);
    ids.push_back(START_HEADER);
    ids.push_back(ASSISTANT_TOKEN);
    ids.push_back(END_HEADER);
    ids.push_back(NEWLINE_NEWLINE);
    return ids;
}

// Per-head scratch buffers. Sized once per forward_step (q_seq and kv_seq are
// fixed across the layer loop) and reused for every layer * batch * head.
struct AttentionScratch {
    float *d_Qi = nullptr;  // [q_seq, HEAD_DIM]
    float *d_KgT = nullptr; // [HEAD_DIM, kv_seq]  (transposed gather)
    float *d_Vg = nullptr;  // [kv_seq, HEAD_DIM]
    float *d_S = nullptr;   // [q_seq, kv_seq]
    float *d_Oi = nullptr;  // [q_seq, HEAD_DIM]
};

// GQA attention across all heads, fully on-device.
// d_Q_b: [q_seq, NUM_HEADS * HEAD_DIM] device pointer for one batch slot.
// d_K_b, d_V_b: [kv_seq, NUM_KV_HEADS * HEAD_DIM] device pointers (typically
//               into the KV cache; only the first kv_seq rows are read).
// d_attn_b:    [q_seq, NUM_HEADS * HEAD_DIM] device output, written per-head.
// scratch:     caller-owned per-head scratch buffers (see AttentionScratch).
// Causal mask is applied only when q_seq == kv_seq (full prefill); decode
// (q_seq=1) skips it because every cached position is in-bounds.
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
        int kvg = hi / heads_per_group;

        // Slice per-head Q, K (transposed), V directly from device buffers.
        gpu_gather_head(d_Q_b, scratch.d_Qi, q_seq, HEAD_DIM, q_stride,
                        hi * HEAD_DIM);
        gpu_gather_head_transpose(d_K_b, scratch.d_KgT, kv_seq, HEAD_DIM,
                                  kv_dim, kvg * HEAD_DIM);
        gpu_gather_head(d_V_b, scratch.d_Vg, kv_seq, HEAD_DIM, kv_dim,
                        kvg * HEAD_DIM);

        gpu_matmul_device(scratch.d_Qi, scratch.d_KgT, scratch.d_S, q_seq,
                          HEAD_DIM, kv_seq);
        gpu_scale(scratch.d_S, q_seq * kv_seq, scale);
        if (apply_mask) {
            gpu_causal_mask(scratch.d_S, q_seq);
        }
        gpu_softmax(scratch.d_S, q_seq, kv_seq);
        gpu_matmul_device(scratch.d_S, scratch.d_Vg, scratch.d_Oi, q_seq,
                          kv_seq, HEAD_DIM);

        // Write this head's output directly into the stitched attention buffer.
        gpu_scatter_head(scratch.d_Oi, d_attn_b, q_seq, HEAD_DIM, q_stride,
                         hi * HEAD_DIM);
    }
}

// CPU dot-product over [VOCAB_SIZE, EMBEDDING_DIM] @ [EMBEDDING_DIM].
// TODO(perf): move to GPU GEMV (see docs/todos/TODO.md item 7).
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

// Run one forward step over `q_seq` new tokens, advancing the cache.
// h_input:    [batch, q_seq, EMBEDDING_DIM] embeddings (host).
// d_cos_full: [S_MAX, HEAD_DIM/2] (device), full RoPE cos table.
// d_sin_full: [S_MAX, HEAD_DIM/2] (device), full RoPE sin table.
// Returns final-RMSNormed last-token hidden states [batch, EMBEDDING_DIM].
std::vector<float> forward_step(const float *h_input, int q_seq,
                                ModelWeights &weights, KVCache &cache,
                                const float *d_cos_full,
                                const float *d_sin_full,
                                DeviceModelWeights *resident_weights,
                                int batch = 1) {
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

    // Persistent device buffers for this step (sized to q_seq).
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

    // Per-head attention scratch — allocated once, reused across every
    // (layer, batch, head). q_seq and kv_seq are fixed for this forward_step.
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

    // RoPE table base offsets for this step (positions [len_before, len_before+q_seq)).
    const float *d_cos_step = d_cos_full + (size_t)len_before * half_hd;
    const float *d_sin_step = d_sin_full + (size_t)len_before * half_hd;

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        Stopwatch sw_layer("layer.total");
        const LayerWeights *lw = nullptr;
        const DeviceLayerWeights *resident_lw = nullptr;

        if (resident_weights != nullptr) {
            resident_lw = &resident_weights->load_layer(layer);
        } else {
            Stopwatch sw_load("layer.load_disk_to_host");
            lw = &weights.load_layer(layer);
        }

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

        {
            Stopwatch sw("layer.attn_pre");
            const float *input_norm =
                resident_lw != nullptr ? resident_lw->input_layernorm : d_gamma;
            gpu_rmsnorm(d_X, input_norm, d_Xnorm, rows, d,
                        RMS_NORM_EPSILON);
            // Q is row-stacked across the batch, but K/V must land in
            // contiguous [batch, s_max, kv_dim] cache slices. The per-batch
            // K/V launches are a deliberate v1 layout tradeoff.
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
            gpu_rope(d_Q, d_cos_step, d_sin_step, rows, NUM_HEADS, HEAD_DIM,
                     q_seq);
            for (int b = 0; b < batch; ++b) {
                gpu_rope(cache.k_at(layer, len_before, b), d_cos_step,
                         d_sin_step, q_seq, NUM_KV_HEADS, HEAD_DIM, q_seq);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Per-head attention, fully on-device: each batch's Q/K/V already
        // live on the GPU (Q in d_Q, K/V in the KV cache). The gather/scatter
        // kernels slice per-head views without ever crossing PCIe.
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

        {
            Stopwatch sw("layer.post_attn_and_ffn");
            if (resident_lw != nullptr) {
                gpu_matmul_device_bf16_weight(d_attn, resident_lw->o_proj,
                                              d_attn_out, rows, d, d);
            } else {
                gpu_matmul_device(d_attn, d_wo, d_attn_out, rows, d, d);
            }
            gpu_residual_add(d_X, d_attn_out, rows * d);
            if (resident_lw == nullptr) {
                CUDA_CHECK(cudaMemcpy(d_gamma, lw->post_attn_layernorm,
                                      d * sizeof(float),
                                      cudaMemcpyHostToDevice));
            }
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
            gpu_swiglu(d_gate, d_up, d_gate, rows * FFN_DIM);
            if (resident_lw != nullptr) {
                gpu_matmul_device_bf16_weight(d_gate, resident_lw->down_proj,
                                              d_ffn, rows, FFN_DIM, d);
            } else {
                gpu_matmul_device(d_gate, d_wdown, d_ffn, rows, FFN_DIM, d);
            }
            gpu_residual_add(d_X, d_ffn, rows * d);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        if (resident_weights == nullptr) {
            Stopwatch sw("layer.unload");
            weights.unload_layer(layer);
        }
    }

    // Cache now holds len_before + q_seq tokens.
    cache.advance(q_seq);

    // Final RMSNorm.
    CUDA_CHECK(cudaMemcpy(d_gamma, weights.global().final_norm,
                           d * sizeof(float), cudaMemcpyHostToDevice));
    gpu_rmsnorm(d_X, d_gamma, d_Xnorm, rows, d, RMS_NORM_EPSILON);

    // Extract each batch element's last row to host.
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

// Allocate and upload the full RoPE cos/sin tables sized to S_MAX positions.
// Caller owns d_cos and d_sin (cudaFree).
void alloc_rope_tables(float **d_cos_out, float **d_sin_out) {
    const int half_hd = HEAD_DIM / 2;
    std::vector<float> h_cos((size_t)S_MAX * half_hd);
    std::vector<float> h_sin((size_t)S_MAX * half_hd);
    precompute_rope_table(h_cos.data(), h_sin.data(), S_MAX, HEAD_DIM,
                          ROPE_BASE);
    CUDA_CHECK(cudaMalloc(d_cos_out, h_cos.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(d_sin_out, h_sin.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(*d_cos_out, h_cos.data(),
                           h_cos.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_sin_out, h_sin.data(),
                           h_sin.size() * sizeof(float),
                           cudaMemcpyHostToDevice));
}

void load_resident_layers(DeviceModelWeights *resident_weights) {
    if (resident_weights == nullptr) {
        return;
    }

    {
        Stopwatch sw("weights.load_all_resident_bf16");
        resident_weights->load_all_layers();
    }

    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    std::printf("  resident BF16 layer weights: %.2f GiB\n",
                resident_weights->total_device_bytes() / gib);
    probe_vram("after_resident_weights");
}

int validate_equal_lengths(const std::vector<std::vector<int>> &batched_ids,
                           const char *context) {
    if (batched_ids.empty()) {
        throw std::runtime_error(std::string(context) + ": empty batch");
    }
    const int s = static_cast<int>(batched_ids[0].size());
    for (size_t b = 1; b < batched_ids.size(); ++b) {
        if (static_cast<int>(batched_ids[b].size()) != s) {
            throw std::runtime_error(
                "batched inference requires equal tokenized prompt lengths "
                "(mixed-length batching is out of scope for TODO #2)");
        }
    }
    return s;
}

GenerateDebugResult generate_tokens_resident_batched_impl(
    ModelWeights &weights, DeviceModelWeights &resident_weights,
    const std::vector<std::string> &prompts, int max_new_tokens) {
    if (prompts.empty()) {
        throw std::runtime_error("generate_tokens_resident: empty prompt batch");
    }

    GenerateDebugResult result;
    result.tokens.resize(prompts.size());
    if (max_new_tokens <= 0) {
        return result;
    }

    Stopwatch::reset();
    probe_vram("startup");

    BPETokenizer tok(TOKENIZER_PATH);
    std::vector<std::vector<int>> prompt_ids;
    prompt_ids.reserve(prompts.size());
    for (const auto &prompt : prompts) {
        prompt_ids.push_back(apply_chat_template(tok, prompt));
    }

    const int batch = static_cast<int>(prompts.size());
    const int prompt_len =
        validate_equal_lengths(prompt_ids, "generate_tokens_resident");

    if (prompt_len + max_new_tokens > S_MAX) {
        throw std::runtime_error(
            "generate_tokens_resident: prompt + max_new_tokens exceeds S_MAX");
    }

    std::printf("  prompt batch: %d prompts, tokens each %d (s=%d)\n", batch,
                prompt_len, prompt_len);

    weights.load_global();
    KVCache cache(S_MAX, batch);
    probe_vram("after_kvcache_alloc");
    load_resident_layers(&resident_weights);

    float *d_cos = nullptr, *d_sin = nullptr;
    alloc_rope_tables(&d_cos, &d_sin);

    std::vector<int> lens;
    int smax = 0;
    std::unique_ptr<float[]> h_emb_prefill(
        weights.get_embeddings_batched(prompt_ids, lens, smax));
    if (smax != prompt_len) {
        throw std::runtime_error("generate_tokens_resident: bad prefill shape");
    }
    auto last_hidden =
        forward_step(h_emb_prefill.get(), prompt_len, weights, cache, d_cos,
                     d_sin, &resident_weights, batch);

    std::vector<int> next_ids(batch, EOT_ID);
    std::vector<bool> done(batch, false);
    for (int b = 0; b < batch; ++b) {
        Stopwatch sw_lm("lm_head.cpu");
        auto logits = compute_lm_head_logits(
            weights.global().lm_head,
            last_hidden.data() + (size_t)b * EMBEDDING_DIM);
        next_ids[b] = static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        result.tokens[b].push_back(next_ids[b]);
        done[b] = (next_ids[b] == EOT_ID);
        std::printf("  [prefill b=%d] -> token %d\n", b, next_ids[b]);
    }

    for (int step = 1; step < max_new_tokens; ++step) {
        if (std::all_of(done.begin(), done.end(), [](bool v) { return v; })) {
            break;
        }

        std::vector<std::vector<int>> one_ids(batch);
        for (int b = 0; b < batch; ++b) {
            // Finished slots stay in the lockstep batch as EOT rows; this wastes
            // compute but keeps the v1 cache and attention layout simple.
            one_ids[b] = {done[b] ? EOT_ID : next_ids[b]};
        }

        std::unique_ptr<float[]> h_emb_one(
            weights.get_embeddings_batched(one_ids, lens, smax));
        if (smax != 1) {
            throw std::runtime_error("generate_tokens_resident: bad decode shape");
        }
        last_hidden = forward_step(h_emb_one.get(), 1, weights, cache, d_cos,
                                   d_sin, &resident_weights, batch);

        for (int b = 0; b < batch; ++b) {
            if (done[b]) {
                continue;
            }
            Stopwatch sw_lm("lm_head.cpu");
            auto logits = compute_lm_head_logits(
                weights.global().lm_head,
                last_hidden.data() + (size_t)b * EMBEDDING_DIM);
            next_ids[b] = static_cast<int>(
                std::max_element(logits.begin(), logits.end()) - logits.begin());
            result.tokens[b].push_back(next_ids[b]);
            done[b] = (next_ids[b] == EOT_ID);
            std::printf("  [decode %d b=%d] -> token %d\n", step, b,
                        next_ids[b]);
        }
    }

    cudaFree(d_cos);
    cudaFree(d_sin);

    result.last_hidden = std::move(last_hidden);
    Stopwatch::print_summary();
    return result;
}

int generate_next_token_impl(ModelWeights &weights,
                             DeviceModelWeights *resident_weights,
                             const std::string &prompt) {
    Stopwatch::reset();
    probe_vram("startup");

    BPETokenizer tok(TOKENIZER_PATH);
    auto token_ids = apply_chat_template(tok, prompt);
    int seq_len = static_cast<int>(token_ids.size());

    std::printf("  tokens: [");
    for (int i = 0; i < seq_len; ++i)
        std::printf("%s%d", i ? ", " : "", token_ids[i]);
    std::printf("] (s=%d)\n", seq_len);

    weights.load_global();
    std::unique_ptr<float[]> h_emb(weights.get_embeddings(token_ids));

    KVCache cache(S_MAX);
    probe_vram("after_kvcache_alloc");
    load_resident_layers(resident_weights);

    float *d_cos = nullptr, *d_sin = nullptr;
    alloc_rope_tables(&d_cos, &d_sin);

    int argmax;
    {
        Stopwatch sw_total("generate.total");
        auto last_hidden = forward_step(h_emb.get(), seq_len, weights, cache,
                                        d_cos, d_sin, resident_weights);
        {
            Stopwatch sw_lm("lm_head.cpu");
            auto logits = compute_lm_head_logits(weights.global().lm_head,
                                                  last_hidden.data());
            argmax = static_cast<int>(
                std::max_element(logits.begin(), logits.end()) - logits.begin());
        }
    }

    cudaFree(d_cos);
    cudaFree(d_sin);

    Stopwatch::print_summary();
    return argmax;
}

std::vector<int> generate_tokens_impl(ModelWeights &weights,
                                      DeviceModelWeights *resident_weights,
                                      const std::string &prompt,
                                      int max_new_tokens) {
    if (max_new_tokens <= 0) {
        return {};
    }

    Stopwatch::reset();
    probe_vram("startup");

    BPETokenizer tok(TOKENIZER_PATH);
    auto prompt_ids = apply_chat_template(tok, prompt);
    int prompt_len = static_cast<int>(prompt_ids.size());

    if (prompt_len + max_new_tokens > S_MAX) {
        throw std::runtime_error(
            "generate_tokens: prompt + max_new_tokens exceeds S_MAX");
    }

    std::printf("  prompt tokens: %d (s=%d)\n", prompt_len, prompt_len);

    weights.load_global();
    KVCache cache(S_MAX);
    probe_vram("after_kvcache_alloc");
    load_resident_layers(resident_weights);

    float *d_cos = nullptr, *d_sin = nullptr;
    alloc_rope_tables(&d_cos, &d_sin);

    // --- Prefill: encode the full prompt, get the first generated token. ---
    std::unique_ptr<float[]> h_emb_prefill(
        weights.get_embeddings(prompt_ids));
    auto last_hidden = forward_step(h_emb_prefill.get(), prompt_len, weights,
                                    cache, d_cos, d_sin, resident_weights);
    auto logits = compute_lm_head_logits(weights.global().lm_head,
                                          last_hidden.data());
    int next_id = static_cast<int>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());

    std::vector<int> generated;
    generated.push_back(next_id);
    std::printf("  [prefill] -> token %d\n", next_id);

    // --- Decode loop: one new token per step until EOT or limit. ---
    for (int step = 1; step < max_new_tokens; ++step) {
        if (next_id == EOT_ID) break;

        std::vector<int> one = {next_id};
        std::unique_ptr<float[]> h_emb_one(weights.get_embeddings(one));
        last_hidden = forward_step(h_emb_one.get(), 1, weights, cache, d_cos,
                                    d_sin, resident_weights);
        logits = compute_lm_head_logits(weights.global().lm_head,
                                         last_hidden.data());
        next_id = static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        generated.push_back(next_id);
        std::printf("  [decode %d] -> token %d\n", step, next_id);
    }

    cudaFree(d_cos);
    cudaFree(d_sin);

    Stopwatch::print_summary();
    return generated;
}

} // namespace

int generate_next_token(ModelWeights &weights, const std::string &prompt) {
    return generate_next_token_impl(weights, nullptr, prompt);
}

std::vector<int> generate_tokens(ModelWeights &weights,
                                 const std::string &prompt,
                                 int max_new_tokens) {
    return generate_tokens_impl(weights, nullptr, prompt, max_new_tokens);
}

int generate_next_token_resident(ModelWeights &weights,
                                 DeviceModelWeights &resident_weights,
                                 const std::string &prompt) {
    return generate_next_token_impl(weights, &resident_weights, prompt);
}

std::vector<int> generate_tokens_resident(ModelWeights &weights,
                                          DeviceModelWeights &resident_weights,
                                          const std::string &prompt,
                                          int max_new_tokens) {
    return generate_tokens_impl(weights, &resident_weights, prompt,
                                max_new_tokens);
}

std::vector<std::vector<int>> generate_tokens_resident(
    ModelWeights &weights, DeviceModelWeights &resident_weights,
    const std::vector<std::string> &prompts, int max_new_tokens) {
    return generate_tokens_resident_batched_impl(weights, resident_weights,
                                                 prompts, max_new_tokens)
        .tokens;
}

GenerateDebugResult generate_tokens_resident_debug(
    ModelWeights &weights, DeviceModelWeights &resident_weights,
    const std::vector<std::string> &prompts, int max_new_tokens) {
    return generate_tokens_resident_batched_impl(weights, resident_weights,
                                                 prompts, max_new_tokens);
}

std::string decode_token(int token_id) {
    static BPETokenizer tok(TOKENIZER_PATH);
    return tok.decode({token_id});
}

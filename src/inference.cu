// Inference pipeline for Llama 3 8B with optional KV cache for incremental
// autoregressive decoding.
//
// Two entry points:
//   - generate_next_token: single forward pass over the prompt; argmax token.
//   - generate_tokens:     prefill + decode loop using KVCache. Each decode
//                          step projects Q for one new token only, appends
//                          one K/V row to the cache, and attends over the
//                          full cached prefix.

#include "inference.h"
#include "config.h"
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

// GQA attention across all heads.
// Q has shape [q_seq, NUM_HEADS * HEAD_DIM] in row-major host memory.
// K, V have shape [kv_seq, NUM_KV_HEADS * HEAD_DIM] in row-major host memory.
// q_offset is the position of Q's first row in the global sequence
// (0 for prefill, cache.len() for decode), used to mask only when needed.
// Causal mask is applied only when q_seq == kv_seq (full prefill).
// For decode (q_seq=1, kv_seq=len+1) all kv positions are valid by
// construction; mask is skipped.
void run_attention_heads(const std::vector<float> &h_Q,
                         const std::vector<float> &h_K,
                         const std::vector<float> &h_V,
                         std::vector<float> &attn_concat,
                         int q_seq, int kv_seq) {
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const int heads_per_group = NUM_HEADS / NUM_KV_HEADS;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    const bool apply_mask = (q_seq == kv_seq && q_seq > 1);

    float *d_Qi = nullptr, *d_KgT = nullptr, *d_Vg = nullptr;
    float *d_S = nullptr, *d_Oi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Qi, q_seq * HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_KgT, HEAD_DIM * kv_seq * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Vg, kv_seq * HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, q_seq * kv_seq * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Oi, q_seq * HEAD_DIM * sizeof(float)));

    for (int hi = 0; hi < NUM_HEADS; ++hi) {
        int kvg = hi / heads_per_group;
        std::vector<float> hQi(q_seq * HEAD_DIM);
        std::vector<float> hKgT(HEAD_DIM * kv_seq);
        std::vector<float> hVg(kv_seq * HEAD_DIM);

        for (int p = 0; p < q_seq; ++p) {
            for (int d2 = 0; d2 < HEAD_DIM; ++d2) {
                hQi[p * HEAD_DIM + d2] =
                    h_Q[p * EMBEDDING_DIM + hi * HEAD_DIM + d2];
            }
        }
        for (int p = 0; p < kv_seq; ++p) {
            for (int d2 = 0; d2 < HEAD_DIM; ++d2) {
                hKgT[d2 * kv_seq + p] =
                    h_K[p * kv_dim + kvg * HEAD_DIM + d2];
                hVg[p * HEAD_DIM + d2] =
                    h_V[p * kv_dim + kvg * HEAD_DIM + d2];
            }
        }

        CUDA_CHECK(cudaMemcpy(d_Qi, hQi.data(),
                               q_seq * HEAD_DIM * sizeof(float),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_KgT, hKgT.data(),
                               HEAD_DIM * kv_seq * sizeof(float),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Vg, hVg.data(),
                               kv_seq * HEAD_DIM * sizeof(float),
                               cudaMemcpyHostToDevice));

        gpu_matmul_device(d_Qi, d_KgT, d_S, q_seq, HEAD_DIM, kv_seq);
        gpu_scale(d_S, q_seq * kv_seq, scale);
        if (apply_mask) {
            gpu_causal_mask(d_S, q_seq);
        }
        gpu_softmax(d_S, q_seq, kv_seq);
        gpu_matmul_device(d_S, d_Vg, d_Oi, q_seq, kv_seq, HEAD_DIM);

        std::vector<float> hOi(q_seq * HEAD_DIM);
        CUDA_CHECK(cudaMemcpy(hOi.data(), d_Oi,
                               q_seq * HEAD_DIM * sizeof(float),
                               cudaMemcpyDeviceToHost));
        for (int p = 0; p < q_seq; ++p)
            for (int d2 = 0; d2 < HEAD_DIM; ++d2)
                attn_concat[p * EMBEDDING_DIM + hi * HEAD_DIM + d2] =
                    hOi[p * HEAD_DIM + d2];
    }

    CUDA_CHECK(cudaFree(d_Qi));
    CUDA_CHECK(cudaFree(d_KgT));
    CUDA_CHECK(cudaFree(d_Vg));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_Oi));
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
// h_input:    [q_seq, EMBEDDING_DIM] embeddings (host).
// d_cos_full: [S_MAX, HEAD_DIM/2] (device), full RoPE cos table.
// d_sin_full: [S_MAX, HEAD_DIM/2] (device), full RoPE sin table.
// Returns the final-RMSNormed last-token hidden state (size EMBEDDING_DIM).
std::vector<float> forward_step(const float *h_input, int q_seq,
                                ModelWeights &weights, KVCache &cache,
                                const float *d_cos_full,
                                const float *d_sin_full) {
    Stopwatch sw_step(q_seq == 1 ? "step.decode" : "step.prefill");
    const int d = EMBEDDING_DIM;
    const int kv_dim = KVCache::kv_dim();
    const int half_hd = HEAD_DIM / 2;

    const int len_before = cache.len();
    const int kv_seq = len_before + q_seq;

    if (kv_seq > cache.max_len()) {
        throw std::runtime_error("forward_step: kv_seq exceeds cache capacity");
    }

    const size_t bytes_X = static_cast<size_t>(q_seq) * d * sizeof(float);
    const size_t bytes_Xkv_full = static_cast<size_t>(kv_seq) * kv_dim *
                                   sizeof(float);
    const size_t bytes_ffn = static_cast<size_t>(q_seq) * FFN_DIM *
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
    CUDA_CHECK(cudaMalloc(&d_wq, (size_t)d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wk, (size_t)d * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wv, (size_t)d * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wo, (size_t)d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wgate, (size_t)d * FFN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wup, (size_t)d * FFN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wdown, (size_t)FFN_DIM * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_attn_out, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_gate, bytes_ffn));
    CUDA_CHECK(cudaMalloc(&d_up, bytes_ffn));
    CUDA_CHECK(cudaMalloc(&d_ffn, bytes_X));

    CUDA_CHECK(cudaMemcpy(d_X, h_input, bytes_X, cudaMemcpyHostToDevice));

    // RoPE table base offsets for this step (positions [len_before, len_before+q_seq)).
    const float *d_cos_step = d_cos_full + (size_t)len_before * half_hd;
    const float *d_sin_step = d_sin_full + (size_t)len_before * half_hd;

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        Stopwatch sw_layer("layer.total");
        const LayerWeights *lw_ptr = nullptr;
        {
            Stopwatch sw_load("layer.load_disk_to_host");
            lw_ptr = &weights.load_layer(layer);
        }
        const LayerWeights &lw = *lw_ptr;

        {
            Stopwatch sw_h2d("layer.h2d_weights");
            CUDA_CHECK(cudaMemcpy(d_gamma, lw.input_layernorm,
                                   d * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_wq, lw.q_proj,
                                   (size_t)d * d * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_wk, lw.k_proj,
                                   (size_t)d * kv_dim * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_wv, lw.v_proj,
                                   (size_t)d * kv_dim * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_wo, lw.o_proj,
                                   (size_t)d * d * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_wgate, lw.gate_proj,
                                   (size_t)d * FFN_DIM * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_wup, lw.up_proj,
                                   (size_t)d * FFN_DIM * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_wdown, lw.down_proj,
                                   (size_t)FFN_DIM * d * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        {
            Stopwatch sw("layer.attn_pre");
            gpu_rmsnorm(d_X, d_gamma, d_Xnorm, q_seq, d, RMS_NORM_EPSILON);
            gpu_matmul_device(d_Xnorm, d_wq, d_Q, q_seq, d, d);
            float *d_K_slot = cache.k_at(layer, len_before);
            float *d_V_slot = cache.v_at(layer, len_before);
            gpu_matmul_device(d_Xnorm, d_wk, d_K_slot, q_seq, d, kv_dim);
            gpu_matmul_device(d_Xnorm, d_wv, d_V_slot, q_seq, d, kv_dim);
            gpu_rope(d_Q, d_cos_step, d_sin_step, q_seq, NUM_HEADS, HEAD_DIM);
            gpu_rope(d_K_slot, d_cos_step, d_sin_step, q_seq, NUM_KV_HEADS,
                      HEAD_DIM);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Copy Q (q_seq rows) and full cached K/V (kv_seq rows) to host for the
        // host-orchestrated per-head attention loop. (See TODO #8: move this on-device.)
        std::vector<float> h_Q(q_seq * d);
        std::vector<float> h_K(kv_seq * kv_dim);
        std::vector<float> h_V(kv_seq * kv_dim);
        CUDA_CHECK(cudaMemcpy(h_Q.data(), d_Q, bytes_X,
                               cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_K.data(), cache.k(layer), bytes_Xkv_full,
                               cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_V.data(), cache.v(layer), bytes_Xkv_full,
                               cudaMemcpyDeviceToHost));

        std::vector<float> attn_concat((size_t)q_seq * d, 0.0f);
        {
            Stopwatch sw("layer.attn_heads");
            run_attention_heads(h_Q, h_K, h_V, attn_concat, q_seq, kv_seq);
        }

        {
            Stopwatch sw("layer.post_attn_and_ffn");
            CUDA_CHECK(cudaMemcpy(d_attn, attn_concat.data(), bytes_X,
                                   cudaMemcpyHostToDevice));
            gpu_matmul_device(d_attn, d_wo, d_attn_out, q_seq, d, d);
            gpu_residual_add(d_X, d_attn_out, q_seq * d);
            CUDA_CHECK(cudaMemcpy(d_gamma, lw.post_attn_layernorm,
                                   d * sizeof(float), cudaMemcpyHostToDevice));
            gpu_rmsnorm(d_X, d_gamma, d_Xnorm, q_seq, d, RMS_NORM_EPSILON);
            gpu_matmul_device(d_Xnorm, d_wgate, d_gate, q_seq, d, FFN_DIM);
            gpu_matmul_device(d_Xnorm, d_wup, d_up, q_seq, d, FFN_DIM);
            gpu_swiglu(d_gate, d_up, d_gate, q_seq * FFN_DIM);
            gpu_matmul_device(d_gate, d_wdown, d_ffn, q_seq, FFN_DIM, d);
            gpu_residual_add(d_X, d_ffn, q_seq * d);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        {
            Stopwatch sw("layer.unload");
            weights.unload_layer(layer);
        }
    }

    // Cache now holds len_before + q_seq tokens.
    cache.advance(q_seq);

    // Final RMSNorm.
    CUDA_CHECK(cudaMemcpy(d_gamma, weights.global().final_norm,
                           d * sizeof(float), cudaMemcpyHostToDevice));
    gpu_rmsnorm(d_X, d_gamma, d_Xnorm, q_seq, d, RMS_NORM_EPSILON);

    // Extract last row to host.
    std::vector<float> last_hidden(d);
    const size_t last_row_offset = (size_t)(q_seq - 1) * d;
    CUDA_CHECK(cudaMemcpy(last_hidden.data(), d_Xnorm + last_row_offset,
                           d * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_X); cudaFree(d_Xnorm); cudaFree(d_Q);
    cudaFree(d_gamma);
    cudaFree(d_wq); cudaFree(d_wk); cudaFree(d_wv); cudaFree(d_wo);
    cudaFree(d_wgate); cudaFree(d_wup); cudaFree(d_wdown);
    cudaFree(d_attn); cudaFree(d_attn_out);
    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_ffn);

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

} // namespace

int generate_next_token(ModelWeights &weights, const std::string &prompt) {
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

    float *d_cos = nullptr, *d_sin = nullptr;
    alloc_rope_tables(&d_cos, &d_sin);

    int argmax;
    {
        Stopwatch sw_total("generate.total");
        auto last_hidden = forward_step(h_emb.get(), seq_len, weights, cache,
                                         d_cos, d_sin);
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

std::vector<int> generate_tokens(ModelWeights &weights,
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

    float *d_cos = nullptr, *d_sin = nullptr;
    alloc_rope_tables(&d_cos, &d_sin);

    // --- Prefill: encode the full prompt, get the first generated token. ---
    std::unique_ptr<float[]> h_emb_prefill(
        weights.get_embeddings(prompt_ids));
    auto last_hidden = forward_step(h_emb_prefill.get(), prompt_len, weights,
                                     cache, d_cos, d_sin);
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
                                    d_sin);
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

std::string decode_token(int token_id) {
    static BPETokenizer tok(TOKENIZER_PATH);
    return tok.decode({token_id});
}

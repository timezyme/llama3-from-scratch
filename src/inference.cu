// Inference pipeline for Llama 3 8B.
// Implements generate_next_token: tokenize -> embed -> 32-layer decode -> logits -> argmax.

#include "inference.h"
#include "config.h"
#include "kernel/kernels.cuh"
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

// Per-head attention: Q*K^T, scale, mask, softmax, *V -> concat.
// Runs on GPU per-head with host-side orchestration.
static void run_attention_heads(const std::vector<float> &h_Q_rope,
                                const std::vector<float> &h_K_rope,
                                const std::vector<float> &h_V,
                                std::vector<float> &attn_concat, int s) {
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const int heads_per_group = NUM_HEADS / NUM_KV_HEADS;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    float *d_Qi = nullptr, *d_KgT = nullptr, *d_Vg = nullptr;
    float *d_S = nullptr, *d_Oi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Qi, s * HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_KgT, HEAD_DIM * s * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Vg, s * HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, s * s * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Oi, s * HEAD_DIM * sizeof(float)));

    for (int hi = 0; hi < NUM_HEADS; ++hi) {
        int kvg = hi / heads_per_group;
        std::vector<float> hQi(s * HEAD_DIM), hKgT(HEAD_DIM * s),
            hVg(s * HEAD_DIM);

        for (int p = 0; p < s; ++p) {
            for (int d2 = 0; d2 < HEAD_DIM; ++d2) {
                hQi[p * HEAD_DIM + d2] =
                    h_Q_rope[p * EMBEDDING_DIM + hi * HEAD_DIM + d2];
                hKgT[d2 * s + p] =
                    h_K_rope[p * kv_dim + kvg * HEAD_DIM + d2];
                hVg[p * HEAD_DIM + d2] =
                    h_V[p * kv_dim + kvg * HEAD_DIM + d2];
            }
        }

        CUDA_CHECK(cudaMemcpy(d_Qi, hQi.data(), s * HEAD_DIM * sizeof(float),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_KgT, hKgT.data(),
                               HEAD_DIM * s * sizeof(float),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Vg, hVg.data(), s * HEAD_DIM * sizeof(float),
                               cudaMemcpyHostToDevice));

        gpu_matmul_device(d_Qi, d_KgT, d_S, s, HEAD_DIM, s);
        gpu_scale(d_S, s * s, scale);
        gpu_causal_mask(d_S, s);
        gpu_softmax(d_S, s, s);
        gpu_matmul_device(d_S, d_Vg, d_Oi, s, s, HEAD_DIM);

        std::vector<float> hOi(s * HEAD_DIM);
        CUDA_CHECK(cudaMemcpy(hOi.data(), d_Oi,
                               s * HEAD_DIM * sizeof(float),
                               cudaMemcpyDeviceToHost));
        for (int p = 0; p < s; ++p)
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

// Compute logits via lm_head weight matrix.
// lm_head is [VOCAB_SIZE, EMBEDDING_DIM], x_last is [EMBEDDING_DIM].
// logits[v] = dot(lm_head[v, :], x_last).
static std::vector<float> compute_lm_head_logits(const float *lm_head,
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

// Run the full 32-layer forward pass on device.
// h_embeddings: [seq_len, EMBEDDING_DIM] on host.
// Returns argmax token ID.
static int run_forward_pass(const float *h_embeddings, int seq_len,
                            ModelWeights &weights) {
    const int d = EMBEDDING_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const size_t total = static_cast<size_t>(seq_len) * d;
    const int half_hd = HEAD_DIM / 2;
    const int table_sz = seq_len * half_hd;

    // Precompute RoPE tables
    std::vector<float> h_cos(table_sz), h_sin(table_sz);
    precompute_rope_table(h_cos.data(), h_sin.data(), seq_len, HEAD_DIM,
                          ROPE_BASE);

    // Persistent device buffers
    float *d_X = nullptr, *d_Xnorm = nullptr;
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
    float *d_cos = nullptr, *d_sin = nullptr;
    float *d_gamma = nullptr;
    float *d_wq = nullptr, *d_wk = nullptr, *d_wv = nullptr, *d_wo = nullptr;
    float *d_wgate = nullptr, *d_wup = nullptr, *d_wdown = nullptr;
    float *d_attn = nullptr, *d_attn_out = nullptr;
    float *d_gate = nullptr, *d_up = nullptr, *d_ffn = nullptr;

    size_t bytes_X = total * sizeof(float);
    size_t bytes_kv = (size_t)seq_len * kv_dim * sizeof(float);
    size_t bytes_ffn = (size_t)seq_len * FFN_DIM * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_X, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_Xnorm, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_Q, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_K, bytes_kv));
    CUDA_CHECK(cudaMalloc(&d_V, bytes_kv));
    CUDA_CHECK(cudaMalloc(&d_cos, table_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, table_sz * sizeof(float)));
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

    // Upload initial data
    CUDA_CHECK(cudaMemcpy(d_X, h_embeddings, bytes_X, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), table_sz * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), table_sz * sizeof(float),
                           cudaMemcpyHostToDevice));

    // 32-layer decoder loop
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        const LayerWeights &lw = weights.load_layer(layer);

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

        gpu_rmsnorm(d_X, d_gamma, d_Xnorm, seq_len, d, RMS_NORM_EPSILON);
        gpu_matmul_device(d_Xnorm, d_wq, d_Q, seq_len, d, d);
        gpu_matmul_device(d_Xnorm, d_wk, d_K, seq_len, d, kv_dim);
        gpu_matmul_device(d_Xnorm, d_wv, d_V, seq_len, d, kv_dim);
        gpu_rope(d_Q, d_cos, d_sin, seq_len, NUM_HEADS, HEAD_DIM);
        gpu_rope(d_K, d_cos, d_sin, seq_len, NUM_KV_HEADS, HEAD_DIM);

        // Attention (per-head loop via host)
        std::vector<float> h_Qr(seq_len * d), h_Kr(seq_len * kv_dim),
            h_Vp(seq_len * kv_dim);
        CUDA_CHECK(cudaMemcpy(h_Qr.data(), d_Q, bytes_X,
                               cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Kr.data(), d_K, bytes_kv,
                               cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Vp.data(), d_V, bytes_kv,
                               cudaMemcpyDeviceToHost));
        std::vector<float> attn_concat(total, 0.0f);
        run_attention_heads(h_Qr, h_Kr, h_Vp, attn_concat, seq_len);

        CUDA_CHECK(cudaMemcpy(d_attn, attn_concat.data(), bytes_X,
                               cudaMemcpyHostToDevice));
        gpu_matmul_device(d_attn, d_wo, d_attn_out, seq_len, d, d);
        gpu_residual_add(d_X, d_attn_out, static_cast<int>(total));

        CUDA_CHECK(cudaMemcpy(d_gamma, lw.post_attn_layernorm,
                               d * sizeof(float), cudaMemcpyHostToDevice));
        gpu_rmsnorm(d_X, d_gamma, d_Xnorm, seq_len, d, RMS_NORM_EPSILON);

        gpu_matmul_device(d_Xnorm, d_wgate, d_gate, seq_len, d, FFN_DIM);
        gpu_matmul_device(d_Xnorm, d_wup, d_up, seq_len, d, FFN_DIM);
        gpu_swiglu(d_gate, d_up, d_gate, seq_len * FFN_DIM);
        gpu_matmul_device(d_gate, d_wdown, d_ffn, seq_len, FFN_DIM, d);
        gpu_residual_add(d_X, d_ffn, static_cast<int>(total));

        weights.unload_layer(layer);
        if ((layer + 1) % 8 == 0)
            std::printf("  layer %d done\n", layer);
    }

    // Final RMSNorm
    CUDA_CHECK(cudaMemcpy(d_gamma, weights.global().final_norm,
                           d * sizeof(float), cudaMemcpyHostToDevice));
    gpu_rmsnorm(d_X, d_gamma, d_Xnorm, seq_len, d, RMS_NORM_EPSILON);

    // Extract last row -> logits via shared lm_head (embedding table dot product)
    std::vector<float> h_Xfinal(total);
    CUDA_CHECK(cudaMemcpy(h_Xfinal.data(), d_Xnorm, bytes_X,
                           cudaMemcpyDeviceToHost));
    const float *x_last = h_Xfinal.data() + (seq_len - 1) * d;
    auto logits = compute_lm_head_logits(weights.global().lm_head, x_last);
    int argmax = static_cast<int>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());

    // Cleanup
    cudaFree(d_X); cudaFree(d_Xnorm);
    cudaFree(d_Q); cudaFree(d_K);
    cudaFree(d_V); cudaFree(d_cos);
    cudaFree(d_sin); cudaFree(d_gamma);
    cudaFree(d_wq); cudaFree(d_wk);
    cudaFree(d_wv); cudaFree(d_wo);
    cudaFree(d_wgate); cudaFree(d_wup);
    cudaFree(d_wdown); cudaFree(d_attn);
    cudaFree(d_attn_out); cudaFree(d_gate);
    cudaFree(d_up); cudaFree(d_ffn);

    return argmax;
}

int generate_next_token(ModelWeights &weights, const std::string &prompt) {
    // Wrap prompt in Llama 3 Instruct chat template.
    // Format: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
    //         {prompt}
    //         <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
    static constexpr int BEGIN_OF_TEXT = 128000;
    static constexpr int START_HEADER  = 128006;
    static constexpr int END_HEADER    = 128007;
    static constexpr int EOT_ID        = 128009;
    static constexpr int NEWLINE_NEWLINE = 271;  // "\n\n"
    static constexpr int USER_TOKEN    = 882;    // "user"
    static constexpr int ASSISTANT_TOKEN = 78191; // "assistant"

    BPETokenizer tok(TOKENIZER_PATH);
    auto encoded = tok.encode(prompt);

    std::vector<int> token_ids;
    // Header: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
    token_ids.push_back(BEGIN_OF_TEXT);
    token_ids.push_back(START_HEADER);
    token_ids.push_back(USER_TOKEN);
    token_ids.push_back(END_HEADER);
    token_ids.push_back(NEWLINE_NEWLINE);
    // Prompt tokens
    token_ids.insert(token_ids.end(), encoded.begin(), encoded.end());
    // Suffix: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
    token_ids.push_back(EOT_ID);
    token_ids.push_back(START_HEADER);
    token_ids.push_back(ASSISTANT_TOKEN);
    token_ids.push_back(END_HEADER);
    token_ids.push_back(NEWLINE_NEWLINE);

    int seq_len = static_cast<int>(token_ids.size());

    std::printf("  tokens: [");
    for (int i = 0; i < seq_len; ++i)
        std::printf("%s%d", i ? ", " : "", token_ids[i]);
    std::printf("] (s=%d)\n", seq_len);

    // Get embeddings
    weights.load_global();
    std::unique_ptr<float[]> h_emb(weights.get_embeddings(token_ids));

    return run_forward_pass(h_emb.get(), seq_len, weights);
}

std::string decode_token(int token_id) {
    static BPETokenizer tok(TOKENIZER_PATH);
    return tok.decode({token_id});
}

// Shared helper implementations for the M2-3 test binary.
//
// Defines all symbols declared in tests/test_m2m3_helpers.h with external
// linkage. Helpers used by only one group's TU stay `static` and travel
// with that group's file.

#include "tests/test_m2m3_helpers.h"

void check_cuda(cudaError_t err, const char *expr, const char *file, int line) {
    if (err == cudaSuccess) return;
    std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line, expr,
                 cudaGetErrorString(err));
    std::exit(FAIL);
}

void fill_deterministic(float *buf, int count, int seed) {
    for (int i = 0; i < count; ++i) {
        buf[i] = static_cast<float>((i + seed) % 13) * 0.1f - 0.6f;
    }
}

uint16_t float_to_bf16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_bits_to_float_host(uint16_t bits) {
    uint32_t fp32_bits = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &fp32_bits, sizeof(value));
    return value;
}

bool compare(const float *a, const float *b, int count, float eps,
             int max_prints) {
    bool ok = true;
    int printed = 0;
    for (int i = 0; i < count; ++i) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > eps) {
            if (printed < max_prints) {
                std::printf("  mismatch [%d]: %.6f vs %.6f (diff=%.6f)\n", i,
                            a[i], b[i], diff);
                ++printed;
            }
            ok = false;
        }
    }
    return ok;
}

float max_abs_diff(const float *a, const float *b, int count) {
    float max_diff = 0.0f;
    for (int i = 0; i < count; ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

std::vector<float> load_fixture(const std::string &path, size_t count) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "  cannot open fixture: %s\n", path.c_str());
        return {};
    }
    std::vector<float> buf(count);
    size_t read = std::fread(buf.data(), sizeof(float), count, f);
    std::fclose(f);
    if (read != count) {
        std::fprintf(stderr, "  fixture size mismatch: expected %zu got %zu\n",
                     count, read);
        return {};
    }
    return buf;
}

// Run attention for all heads on pre-RoPE'd Q, K and original V.
// Q_rope: [s, EMBEDDING_DIM], K_rope: [s, kv_dim], V: [s, kv_dim] on host.
// Writes concatenated result to attn_concat [s, EMBEDDING_DIM] on host.
void run_attention_heads(const std::vector<float> &h_Q_rope,
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

    CUDA_CHECK(cudaFree(d_Qi)); CUDA_CHECK(cudaFree(d_KgT));
    CUDA_CHECK(cudaFree(d_Vg)); CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_Oi));
}

// Forward pass: 32-layer decoder + final norm + logits -> argmax token.
// h_embeddings: [seq_len, EMBEDDING_DIM] host memory.
// weights: ModelWeights with load_global() already called.
// Returns argmax token ID.
int run_forward_pass(const float *h_embeddings, int seq_len,
                     ModelWeights &weights) {
    const int d = EMBEDDING_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const size_t total = static_cast<size_t>(seq_len) * d;
    const int half_hd = HEAD_DIM / 2;
    const int table_sz = seq_len * half_hd;

    std::vector<float> h_cos(table_sz), h_sin(table_sz);
    precompute_rope_table(h_cos.data(), h_sin.data(), seq_len, HEAD_DIM,
                          ROPE_BASE);

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

    CUDA_CHECK(cudaMemcpy(d_X, h_embeddings, bytes_X, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), table_sz * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), table_sz * sizeof(float),
                           cudaMemcpyHostToDevice));

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

    CUDA_CHECK(cudaMemcpy(d_gamma, weights.global().final_norm,
                           d * sizeof(float), cudaMemcpyHostToDevice));
    gpu_rmsnorm(d_X, d_gamma, d_Xnorm, seq_len, d, RMS_NORM_EPSILON);

    std::vector<float> h_Xfinal(total);
    CUDA_CHECK(cudaMemcpy(h_Xfinal.data(), d_Xnorm, bytes_X,
                           cudaMemcpyDeviceToHost));
    const float *x_last = h_Xfinal.data() + (seq_len - 1) * d;
    auto logits = compute_lm_head_logits(weights.global().lm_head, x_last);
    int argmax = static_cast<int>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());

    CUDA_CHECK(cudaFree(d_X)); CUDA_CHECK(cudaFree(d_Xnorm));
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin)); CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_wq)); CUDA_CHECK(cudaFree(d_wk));
    CUDA_CHECK(cudaFree(d_wv)); CUDA_CHECK(cudaFree(d_wo));
    CUDA_CHECK(cudaFree(d_wgate)); CUDA_CHECK(cudaFree(d_wup));
    CUDA_CHECK(cudaFree(d_wdown)); CUDA_CHECK(cudaFree(d_attn));
    CUDA_CHECK(cudaFree(d_attn_out)); CUDA_CHECK(cudaFree(d_gate));
    CUDA_CHECK(cudaFree(d_up)); CUDA_CHECK(cudaFree(d_ffn));

    return argmax;
}

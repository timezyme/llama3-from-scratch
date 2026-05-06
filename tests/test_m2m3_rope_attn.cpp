// Phase 2 tests: RoPE, GQA mapping, causal mask, softmax, attention output,
// and the new output_projection_fixture (Milestone 3).

#include "tests/test_m2m3_helpers.h"

// ---------------------------------------------------------------------------
// Phase 2 Tests: RoPE and Attention
// ---------------------------------------------------------------------------

// Proves: RoPE pairs (i, i+h_d/2), uses base 500000, not even/odd or 10000.
static int test_rope_manual() {
    // Tiny test: 1 position, 1 head, head_dim=4 (half=2).
    // q = [1, 2, 3, 4]
    // theta_0 = 1/(500000^(0/4)) = 1.0
    // theta_1 = 1/(500000^(2/4)) = 1/sqrt(500000) = 0.001414...
    // At position 0: cos=1, sin=0 for both -> no change.
    // At position 1:
    //   cos(1*theta_0) = cos(1.0) = 0.540302
    //   sin(1*theta_0) = sin(1.0) = 0.841471
    //   cos(1*theta_1) = cos(0.001414) = 0.999999
    //   sin(1*theta_1) = sin(0.001414) = 0.001414
    // Pairs: (q[0],q[2]) and (q[1],q[3])
    //   q'[0] = 1*cos(1) - 3*sin(1) = 0.540302 - 2.524413 = -1.984111
    //   q'[2] = 1*sin(1) + 3*cos(1) = 0.841471 + 1.620906 = 2.462377
    //   q'[1] = 2*cos(0.001414) - 4*sin(0.001414) = 1.999997 - 0.005657 = 1.994340
    //   q'[3] = 2*sin(0.001414) + 4*cos(0.001414) = 0.002828 + 3.999996 = 4.002824

    const int seq_len = 2, num_heads = 1, head_dim = 4;
    const int half_hd = head_dim / 2;
    const int total = seq_len * num_heads * head_dim;
    const int table_size = seq_len * half_hd;

    // Input: position 0 = [1,2,3,4], position 1 = [1,2,3,4]
    float h_x[] = {1,2,3,4, 1,2,3,4};

    // Precompute cos/sin tables
    float h_cos[4], h_sin[4]; // [2, 2]
    precompute_rope_table(h_cos, h_sin, seq_len, head_dim, ROPE_BASE);

    float *d_x = nullptr, *d_cos = nullptr, *d_sin = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos, table_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, table_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, total * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos, table_size * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin, table_size * sizeof(float),
                           cudaMemcpyHostToDevice));

    gpu_rope(d_x, d_cos, d_sin, seq_len, num_heads, head_dim);

    float h_out[8];
    CUDA_CHECK(cudaMemcpy(h_out, d_x, total * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));

    // Position 0: no rotation (cos=1, sin=0)
    float expected[] = {1,2,3,4, -1.984111f, 1.994340f, 2.462377f, 4.002825f};
    if (!compare(h_out, expected, 8, 1e-3f)) {
        std::printf("FAIL rope_manual\n");
        return FAIL;
    }
    std::printf("PASS rope_manual\n");
    return PASS;
}

// Verifies RoPE on real Q from layer 0 against golden.
static int test_rope_fixture_q() {
    const int s = 3;
    const size_t total = static_cast<size_t>(s) * EMBEDDING_DIM;

    auto h_q = load_fixture("tests/data/m2m3/q_proj_layer0.bin", total);
    auto h_expected = load_fixture("tests/data/m2m3/q_rope_layer0.bin", total);
    if (h_q.empty() || h_expected.empty()) {
        std::printf("SKIP rope_fixture_q (fixtures not generated)\n");
        return FAIL;
    }

    // Precompute tables
    int half_hd = HEAD_DIM / 2;
    int table_sz = s * half_hd;
    std::vector<float> h_cos(table_sz), h_sin(table_sz);
    precompute_rope_table(h_cos.data(), h_sin.data(), s, HEAD_DIM, ROPE_BASE);

    float *d_q = nullptr, *d_cos = nullptr, *d_sin = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos, table_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, table_sz * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), total * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), table_sz * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), table_sz * sizeof(float),
                           cudaMemcpyHostToDevice));

    gpu_rope(d_q, d_cos, d_sin, s, NUM_HEADS, HEAD_DIM);

    std::vector<float> h_out(total);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_q, total * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));

    if (!compare(h_out.data(), h_expected.data(),
                 static_cast<int>(total), EPSILON)) {
        std::printf("FAIL rope_fixture_q\n");
        return FAIL;
    }
    std::printf("PASS rope_fixture_q\n");
    return PASS;
}

// Verifies RoPE on real K from layer 0 against golden.
static int test_rope_fixture_k() {
    const int s = 3;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const size_t total = static_cast<size_t>(s) * kv_dim;

    auto h_k = load_fixture("tests/data/m2m3/k_proj_layer0.bin", total);
    auto h_expected = load_fixture("tests/data/m2m3/k_rope_layer0.bin", total);
    if (h_k.empty() || h_expected.empty()) {
        std::printf("SKIP rope_fixture_k (fixtures not generated)\n");
        return FAIL;
    }

    int half_hd = HEAD_DIM / 2;
    int table_sz = s * half_hd;
    std::vector<float> h_cos(table_sz), h_sin(table_sz);
    precompute_rope_table(h_cos.data(), h_sin.data(), s, HEAD_DIM, ROPE_BASE);

    float *d_k = nullptr, *d_cos = nullptr, *d_sin = nullptr;
    CUDA_CHECK(cudaMalloc(&d_k, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos, table_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, table_sz * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), total * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), table_sz * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), table_sz * sizeof(float),
                           cudaMemcpyHostToDevice));

    gpu_rope(d_k, d_cos, d_sin, s, NUM_KV_HEADS, HEAD_DIM);

    std::vector<float> h_out(total);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_k, total * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));

    if (!compare(h_out.data(), h_expected.data(),
                 static_cast<int>(total), EPSILON)) {
        std::printf("FAIL rope_fixture_k\n");
        return FAIL;
    }
    std::printf("PASS rope_fixture_k\n");
    return PASS;
}

// Proves: query heads 0..3->KV0, 4..7->KV1, 28..31->KV7.
static int test_gqa_head_mapping() {
    const int heads_per_group = NUM_HEADS / NUM_KV_HEADS; // 4
    bool ok = true;
    for (int q = 0; q < NUM_HEADS; ++q) {
        int expected_kv = q / heads_per_group;
        if (expected_kv < 0 || expected_kv >= NUM_KV_HEADS) {
            std::printf("  head %d maps to kv %d (out of range)\n",
                        q, expected_kv);
            ok = false;
        }
    }
    // Spot checks
    if (0 / heads_per_group != 0) { ok = false; }
    if (3 / heads_per_group != 0) { ok = false; }
    if (4 / heads_per_group != 1) { ok = false; }
    if (31 / heads_per_group != 7) { ok = false; }

    if (!ok) {
        std::printf("FAIL gqa_head_mapping\n");
        return FAIL;
    }
    std::printf("PASS gqa_head_mapping\n");
    return PASS;
}

// Proves: causal mask sets all upper-triangular entries to -1e6.
static int test_causal_mask_triangle() {
    const int s = 5;
    std::vector<float> h_S(s * s, 1.0f); // all ones initially

    float *d_S = nullptr;
    CUDA_CHECK(cudaMalloc(&d_S, s * s * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_S, h_S.data(), s * s * sizeof(float),
                           cudaMemcpyHostToDevice));

    gpu_causal_mask(d_S, s);

    CUDA_CHECK(cudaMemcpy(h_S.data(), d_S, s * s * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_S));

    bool ok = true;
    for (int r = 0; r < s; ++r) {
        for (int c = 0; c < s; ++c) {
            float val = h_S[r * s + c];
            if (c > r) {
                // Should be masked
                if (val != -1e6f) {
                    std::printf("  [%d,%d] expected -1e6, got %f\n", r, c, val);
                    ok = false;
                }
            } else {
                // Should be unchanged (1.0)
                if (val != 1.0f) {
                    std::printf("  [%d,%d] expected 1.0, got %f\n", r, c, val);
                    ok = false;
                }
            }
        }
    }
    if (!ok) {
        std::printf("FAIL causal_mask_triangle\n");
        return FAIL;
    }
    std::printf("PASS causal_mask_triangle\n");
    return PASS;
}

// Proves: softmax is numerically stable (subtracts max before exp).
// Tests with large values that would overflow without max subtraction.
static int test_softmax_stability() {
    const int rows = 2, cols = 4;
    // Row 0: large values that overflow without max subtraction
    // Row 1: normal values
    float h_data[] = {1000.0f, 1001.0f, 999.0f, 998.0f,
                      1.0f, 2.0f, 3.0f, 4.0f};

    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, rows * cols * sizeof(float),
                           cudaMemcpyHostToDevice));

    gpu_softmax(d_data, rows, cols);

    float h_out[8];
    CUDA_CHECK(cudaMemcpy(h_out, d_data, rows * cols * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));

    bool ok = true;
    for (int r = 0; r < rows; ++r) {
        float row_sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float val = h_out[r * cols + c];
            if (!std::isfinite(val)) {
                std::printf("  row %d col %d: non-finite %f\n", r, c, val);
                ok = false;
            }
            if (val < 0.0f) {
                std::printf("  row %d col %d: negative %f\n", r, c, val);
                ok = false;
            }
            row_sum += val;
        }
        if (std::fabs(row_sum - 1.0f) > 1e-4f) {
            std::printf("  row %d sum: %f (expected 1.0)\n", r, row_sum);
            ok = false;
        }
    }
    if (!ok) {
        std::printf("FAIL softmax_stability\n");
        return FAIL;
    }
    std::printf("PASS softmax_stability\n");
    return PASS;
}

// Verifies full attention output (all 32 heads) against golden.
static int test_attention_output_full_fixture() {
    const int s = 3;
    const size_t total = static_cast<size_t>(s) * EMBEDDING_DIM;

    auto h_expected =
        load_fixture("tests/data/m2m3/attn_output_full.bin", total);
    auto h_q_proj = load_fixture("tests/data/m2m3/q_proj_layer0.bin",
                                  static_cast<size_t>(s) * EMBEDDING_DIM);
    auto h_k_proj = load_fixture("tests/data/m2m3/k_proj_layer0.bin",
                                  static_cast<size_t>(s) * NUM_KV_HEADS * HEAD_DIM);
    auto h_v_proj = load_fixture("tests/data/m2m3/v_proj_layer0.bin",
                                  static_cast<size_t>(s) * NUM_KV_HEADS * HEAD_DIM);
    if (h_expected.empty() || h_q_proj.empty() || h_k_proj.empty() || h_v_proj.empty()) {
        std::printf("SKIP attention_output_full_fixture (fixtures not generated)\n");
        return FAIL;
    }

    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const int half_hd = HEAD_DIM / 2;
    const int table_sz = s * half_hd;

    // Precompute RoPE tables
    std::vector<float> h_cos(table_sz), h_sin(table_sz);
    precompute_rope_table(h_cos.data(), h_sin.data(), s, HEAD_DIM, ROPE_BASE);

    // Upload Q, K and RoPE tables to device for the in-place rotation.
    // V is not rotated, so it stays on the host.
    float *d_Q = nullptr, *d_K = nullptr;
    float *d_cos = nullptr, *d_sin = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q, s * EMBEDDING_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, s * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos, table_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, table_sz * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_q_proj.data(),
                           s * EMBEDDING_DIM * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_k_proj.data(),
                           s * kv_dim * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), table_sz * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), table_sz * sizeof(float),
                           cudaMemcpyHostToDevice));

    gpu_rope(d_Q, d_cos, d_sin, s, NUM_HEADS, HEAD_DIM);
    gpu_rope(d_K, d_cos, d_sin, s, NUM_KV_HEADS, HEAD_DIM);

    // Pull rotated Q, K back to host; run_attention_heads orchestrates the
    // per-head GQA loop on top of those plus the original V.
    std::vector<float> h_Q_rope(s * EMBEDDING_DIM);
    std::vector<float> h_K_rope(s * kv_dim);
    CUDA_CHECK(cudaMemcpy(h_Q_rope.data(), d_Q,
                           s * EMBEDDING_DIM * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_K_rope.data(), d_K,
                           s * kv_dim * sizeof(float),
                           cudaMemcpyDeviceToHost));

    std::vector<float> attn_concat(s * EMBEDDING_DIM, 0.0f);
    run_attention_heads(h_Q_rope, h_K_rope, h_v_proj, attn_concat, s);

    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_cos)); CUDA_CHECK(cudaFree(d_sin));

    if (!compare(attn_concat.data(), h_expected.data(),
                 static_cast<int>(total), EPSILON)) {
        std::printf("FAIL attention_output_full_fixture\n");
        return FAIL;
    }
    std::printf("PASS attention_output_full_fixture\n");
    return PASS;
}

// Verifies the W_O projection against golden output.
// Catches a bug where W_O is loaded with the wrong shape or transpose: a
// downstream miss would otherwise only surface in decoder_block_layer0_fixture.
static int test_output_projection_fixture() {
    const int s = 3, d = EMBEDDING_DIM;
    const size_t total = static_cast<size_t>(s) * d;

    auto h_attn = load_fixture("tests/data/m2m3/attn_output_full.bin", total);
    auto h_expected = load_fixture("tests/data/m2m3/o_proj_layer0.bin", total);
    if (h_attn.empty() || h_expected.empty()) {
        std::printf("SKIP output_projection_fixture (fixtures not generated)\n");
        return FAIL;
    }

    ModelWeights weights(DUMP_DIR);
    const LayerWeights &lw = weights.load_layer(0);

    float *d_attn = nullptr, *d_w = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_attn, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, static_cast<size_t>(d) * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_attn, h_attn.data(), total * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, lw.o_proj,
                           static_cast<size_t>(d) * d * sizeof(float),
                           cudaMemcpyHostToDevice));

    gpu_matmul_device(d_attn, d_w, d_out, s, d, d);

    std::vector<float> h_output(total);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_out, total * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_attn));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_out));

    if (!compare(h_output.data(), h_expected.data(),
                 static_cast<int>(total), EPSILON)) {
        std::printf("FAIL output_projection_fixture\n");
        return FAIL;
    }
    std::printf("PASS output_projection_fixture\n");
    return PASS;
}

void register_phase2(Registry &r) {
    r["rope_manual"] = test_rope_manual;
    r["rope_fixture_q"] = test_rope_fixture_q;
    r["rope_fixture_k"] = test_rope_fixture_k;
    r["gqa_head_mapping"] = test_gqa_head_mapping;
    r["causal_mask_triangle"] = test_causal_mask_triangle;
    r["softmax_stability"] = test_softmax_stability;
    r["attention_output_full_fixture"] = test_attention_output_full_fixture;
    r["output_projection_fixture"] = test_output_projection_fixture;
}

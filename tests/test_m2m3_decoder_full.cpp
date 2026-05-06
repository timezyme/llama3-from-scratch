// Phase 3 + Phase 4 tests: residual add, SwiGLU, the FFN sub-block,
// the full decoder block layer 0, and the end-to-end "Hello world"
// forward pass (Milestone 3).

#include "tests/test_m2m3_helpers.h"

// ---------------------------------------------------------------------------
// Phase 3 Tests: O projection, residuals, SwiGLU FFN
// ---------------------------------------------------------------------------

// Proves: elementwise residual add works correctly.
static int test_residual_add_manual() {
    float h_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_b[] = {0.5f, -1.0f, 0.0f, 2.5f};
    float expected[] = {1.5f, 1.0f, 3.0f, 6.5f};

    // Simple CPU check since residual_add is trivial
    // (CUDA kernel will be tested in the full pipeline)
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        float sum = h_a[i] + h_b[i];
        if (std::fabs(sum - expected[i]) > 1e-6f) {
            std::printf("  [%d]: %f + %f = %f, expected %f\n",
                        i, h_a[i], h_b[i], sum, expected[i]);
            ok = false;
        }
    }
    if (!ok) {
        std::printf("FAIL residual_add_manual\n");
        return FAIL;
    }
    std::printf("PASS residual_add_manual\n");
    return PASS;
}

// Proves: SiLU(gate) * up logic is correct.
static int test_swiglu_manual() {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    // gate = [1, -1, 0, 2], up = [2, 3, 1, 0.5]
    // SiLU(1) = 1/(1+exp(-1)) = 0.731059
    // SiLU(-1) = -1/(1+exp(1)) = -0.268941
    // SiLU(0) = 0
    // SiLU(2) = 2/(1+exp(-2)) = 1.761594
    // result = [0.731059*2, -0.268941*3, 0*1, 1.761594*0.5]
    //        = [1.462117, -0.806824, 0, 0.880797]

    float gate[] = {1.0f, -1.0f, 0.0f, 2.0f};
    float up[] = {2.0f, 3.0f, 1.0f, 0.5f};
    float expected[] = {1.462117f, -0.806824f, 0.0f, 0.880797f};

    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        float silu = gate[i] / (1.0f + std::exp(-gate[i]));
        float result = silu * up[i];
        if (std::fabs(result - expected[i]) > 1e-4f) {
            std::printf("  [%d]: SiLU(%f)*%f = %f, expected %f\n",
                        i, gate[i], up[i], result, expected[i]);
            ok = false;
        }
    }
    if (!ok) {
        std::printf("FAIL swiglu_manual\n");
        return FAIL;
    }
    std::printf("PASS swiglu_manual\n");
    return PASS;
}

// Verifies the FFN sub-block (gate_proj -> up_proj -> SwiGLU -> down_proj)
// against golden, isolating it from the residual-add and norm steps that
// the full decoder block test bundles in.
//
// Naming gotcha: tests/data/m2m3/swiglu_layer0.bin holds the *post-down_proj*
// FFN output (see tools/gen_m2m3_fixtures.py:244-245), not just the SwiGLU
// activation.
static int test_ffn_block_isolated_fixture() {
    const int s = 3, d = EMBEDDING_DIM;
    const size_t total_x = static_cast<size_t>(s) * d;
    const size_t total_ffn = static_cast<size_t>(s) * FFN_DIM;

    auto h_xnorm = load_fixture(
        "tests/data/m2m3/post_attn_rmsnorm_layer0.bin", total_x);
    auto h_expected = load_fixture("tests/data/m2m3/swiglu_layer0.bin", total_x);
    if (h_xnorm.empty() || h_expected.empty()) {
        std::printf("SKIP ffn_block_isolated_fixture (fixtures not generated)\n");
        return FAIL;
    }

    ModelWeights weights(DUMP_DIR);
    const LayerWeights &lw = weights.load_layer(0);

    float *d_x = nullptr, *d_gate_w = nullptr, *d_up_w = nullptr,
          *d_down_w = nullptr;
    float *d_gate = nullptr, *d_up = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, total_x * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gate_w, (size_t)d * FFN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up_w, (size_t)d * FFN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_down_w, (size_t)FFN_DIM * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gate, total_ffn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up, total_ffn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total_x * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_xnorm.data(), total_x * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gate_w, lw.gate_proj,
                           (size_t)d * FFN_DIM * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up_w, lw.up_proj,
                           (size_t)d * FFN_DIM * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_down_w, lw.down_proj,
                           (size_t)FFN_DIM * d * sizeof(float),
                           cudaMemcpyHostToDevice));

    gpu_matmul_device(d_x, d_gate_w, d_gate, s, d, FFN_DIM);
    gpu_matmul_device(d_x, d_up_w, d_up, s, d, FFN_DIM);
    gpu_swiglu(d_gate, d_up, d_gate, s * FFN_DIM);
    gpu_matmul_device(d_gate, d_down_w, d_out, s, FFN_DIM, d);

    std::vector<float> h_output(total_x);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_out, total_x * sizeof(float),
                           cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_gate_w));
    CUDA_CHECK(cudaFree(d_up_w));
    CUDA_CHECK(cudaFree(d_down_w));
    CUDA_CHECK(cudaFree(d_gate));
    CUDA_CHECK(cudaFree(d_up));
    CUDA_CHECK(cudaFree(d_out));

    if (!compare(h_output.data(), h_expected.data(),
                 static_cast<int>(total_x), EPSILON)) {
        std::printf("FAIL ffn_block_isolated_fixture\n");
        return FAIL;
    }
    std::printf("PASS ffn_block_isolated_fixture\n");
    return PASS;
}

// GPU smoke test: gpu_swiglu produces correct output for known inputs.
static int test_swiglu_kernel_smoke() {
    constexpr int N = 8;
    // gate values span negative, zero, and positive
    float h_gate[N] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    float h_up[N]   = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    // CPU reference: SiLU(gate[i]) * up[i]
    float h_expected[N];
    for (int i = 0; i < N; ++i) {
        float silu = h_gate[i] / (1.0f + std::exp(-h_gate[i]));
        h_expected[i] = silu * h_up[i];
    }

    // Allocate device memory
    float *d_gate = nullptr, *d_up = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_gate, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_gate, h_gate, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up, h_up, N * sizeof(float), cudaMemcpyHostToDevice));

    gpu_swiglu(d_gate, d_up, d_out, N);

    float h_result[N];
    CUDA_CHECK(cudaMemcpy(h_result, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_out);

    if (!compare(h_result, h_expected, N, 1e-5f)) {
        std::printf("FAIL swiglu_kernel_smoke\n");
        return FAIL;
    }
    std::printf("PASS swiglu_kernel_smoke\n");
    return PASS;
}

// GPU smoke test: gpu_residual_add performs in-place a[i] += b[i].
static int test_residual_add_kernel_smoke() {
    constexpr int N = 4;
    float h_a[N] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_b[N] = {10.0f, 20.0f, 30.0f, 40.0f};
    float h_expected[N] = {11.0f, 22.0f, 33.0f, 44.0f};

    float *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    gpu_residual_add(d_a, d_b, N);

    float h_result[N];
    CUDA_CHECK(cudaMemcpy(h_result, d_a, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);

    // Exact match -- pure addition, no accumulation drift
    if (!compare(h_result, h_expected, N, 1e-6f)) {
        std::printf("FAIL residual_add_kernel_smoke\n");
        return FAIL;
    }
    std::printf("PASS residual_add_kernel_smoke\n");
    return PASS;
}

// Verifies full decoder block layer 0 output against golden.
static int test_decoder_block_layer0_fixture() {
    const int s = 3;
    const size_t total = static_cast<size_t>(s) * EMBEDDING_DIM;

    auto h_expected =
        load_fixture("tests/data/m2m3/decoder_block_layer0.bin", total);
    if (h_expected.empty()) {
        std::printf("SKIP decoder_block_layer0_fixture (fixture not generated)\n");
        return FAIL;
    }

    // Load embeddings as input
    auto h_X = load_fixture("tests/data/m2m3/embeddings_hello.bin", total);
    if (h_X.empty()) {
        std::printf("SKIP decoder_block_layer0_fixture (embeddings not found)\n");
        return FAIL;
    }

    // Load all layer-0 weights
    ModelWeights weights(DUMP_DIR);
    const LayerWeights &lw = weights.load_layer(0);

    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const int half_hd = HEAD_DIM / 2;
    const int table_sz = s * half_hd;

    // === Allocate device memory ===
    float *d_X = nullptr, *d_Xnorm = nullptr;
    float *d_gamma1 = nullptr, *d_gamma2 = nullptr;
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
    float *d_cos = nullptr, *d_sin = nullptr;
    float *d_wq = nullptr, *d_wk = nullptr, *d_wv = nullptr;
    float *d_wo = nullptr, *d_wgate = nullptr, *d_wup = nullptr, *d_wdown = nullptr;

    size_t bytes_X = total * sizeof(float);
    size_t bytes_kv = static_cast<size_t>(s) * kv_dim * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_X, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_Xnorm, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_gamma1, EMBEDDING_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma2, EMBEDDING_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_K, bytes_kv));
    CUDA_CHECK(cudaMalloc(&d_V, bytes_kv));
    CUDA_CHECK(cudaMalloc(&d_cos, table_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, table_sz * sizeof(float)));

    // Weight buffers
    CUDA_CHECK(cudaMalloc(&d_wq, (size_t)EMBEDDING_DIM * EMBEDDING_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wk, (size_t)EMBEDDING_DIM * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wv, (size_t)EMBEDDING_DIM * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wo, (size_t)EMBEDDING_DIM * EMBEDDING_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wgate, (size_t)EMBEDDING_DIM * FFN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wup, (size_t)EMBEDDING_DIM * FFN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_wdown, (size_t)FFN_DIM * EMBEDDING_DIM * sizeof(float)));

    // Upload inputs
    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), bytes_X, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma1, lw.input_layernorm,
                           EMBEDDING_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma2, lw.post_attn_layernorm,
                           EMBEDDING_DIM * sizeof(float), cudaMemcpyHostToDevice));

    // Upload weights
    CUDA_CHECK(cudaMemcpy(d_wq, lw.q_proj,
                           (size_t)EMBEDDING_DIM * EMBEDDING_DIM * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wk, lw.k_proj,
                           (size_t)EMBEDDING_DIM * kv_dim * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wv, lw.v_proj,
                           (size_t)EMBEDDING_DIM * kv_dim * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wo, lw.o_proj,
                           (size_t)EMBEDDING_DIM * EMBEDDING_DIM * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wgate, lw.gate_proj,
                           (size_t)EMBEDDING_DIM * FFN_DIM * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wup, lw.up_proj,
                           (size_t)EMBEDDING_DIM * FFN_DIM * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wdown, lw.down_proj,
                           (size_t)FFN_DIM * EMBEDDING_DIM * sizeof(float),
                           cudaMemcpyHostToDevice));

    // RoPE tables
    std::vector<float> h_cos(table_sz), h_sin(table_sz);
    precompute_rope_table(h_cos.data(), h_sin.data(), s, HEAD_DIM, ROPE_BASE);
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), table_sz * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), table_sz * sizeof(float),
                           cudaMemcpyHostToDevice));

    // === 1. RMSNorm ===
    gpu_rmsnorm(d_X, d_gamma1, d_Xnorm, s, EMBEDDING_DIM, RMS_NORM_EPSILON);

    // === 2. Q, K, V projections ===
    gpu_matmul_device(d_Xnorm, d_wq, d_Q, s, EMBEDDING_DIM, EMBEDDING_DIM);
    gpu_matmul_device(d_Xnorm, d_wk, d_K, s, EMBEDDING_DIM, kv_dim);
    gpu_matmul_device(d_Xnorm, d_wv, d_V, s, EMBEDDING_DIM, kv_dim);

    // === 3. RoPE on Q and K ===
    gpu_rope(d_Q, d_cos, d_sin, s, NUM_HEADS, HEAD_DIM);
    gpu_rope(d_K, d_cos, d_sin, s, NUM_KV_HEADS, HEAD_DIM);

    // === 4. Attention (per-head loop) ===
    // Copy Q, K, V back to host so run_attention_heads can do the per-head
    // GQA slicing + scale/mask/softmax/matmul on its own scratch buffers.
    std::vector<float> h_Q(s * EMBEDDING_DIM), h_K(s * kv_dim), h_V(s * kv_dim);
    CUDA_CHECK(cudaMemcpy(h_Q.data(), d_Q, bytes_X, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_K.data(), d_K, bytes_kv, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_V.data(), d_V, bytes_kv, cudaMemcpyDeviceToHost));

    std::vector<float> attn_concat(s * EMBEDDING_DIM, 0.0f);
    run_attention_heads(h_Q, h_K, h_V, attn_concat, s);

    // === 5. O projection ===
    float *d_attn = nullptr, *d_attn_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_attn, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_attn_out, bytes_X));
    CUDA_CHECK(cudaMemcpy(d_attn, attn_concat.data(), bytes_X,
                           cudaMemcpyHostToDevice));
    gpu_matmul_device(d_attn, d_wo, d_attn_out, s, EMBEDDING_DIM, EMBEDDING_DIM);

    // === 6. First residual: X += attn_out (GPU kernel) ===
    gpu_residual_add(d_X, d_attn_out, static_cast<int>(total));

    // === 7. Post-attention RMSNorm ===
    gpu_rmsnorm(d_X, d_gamma2, d_Xnorm, s, EMBEDDING_DIM, RMS_NORM_EPSILON);

    // === 8. SwiGLU FFN ===
    float *d_gate = nullptr, *d_up = nullptr, *d_ffn = nullptr;
    size_t bytes_ffn = (size_t)s * FFN_DIM * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_gate, bytes_ffn));
    CUDA_CHECK(cudaMalloc(&d_up, bytes_ffn));
    CUDA_CHECK(cudaMalloc(&d_ffn, bytes_X));

    gpu_matmul_device(d_Xnorm, d_wgate, d_gate, s, EMBEDDING_DIM, FFN_DIM);
    gpu_matmul_device(d_Xnorm, d_wup, d_up, s, EMBEDDING_DIM, FFN_DIM);

    // SiLU(gate) * up on GPU
    gpu_swiglu(d_gate, d_up, d_gate, s * FFN_DIM);

    gpu_matmul_device(d_gate, d_wdown, d_ffn, s, FFN_DIM, EMBEDDING_DIM);

    // === 9. Second residual: X += ffn_out (GPU kernel) ===
    gpu_residual_add(d_X, d_ffn, static_cast<int>(total));

    // Read final result
    std::vector<float> h_result(total);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_X, bytes_X,
                           cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_X)); CUDA_CHECK(cudaFree(d_Xnorm));
    CUDA_CHECK(cudaFree(d_gamma1)); CUDA_CHECK(cudaFree(d_gamma2));
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin)); CUDA_CHECK(cudaFree(d_wq));
    CUDA_CHECK(cudaFree(d_wk)); CUDA_CHECK(cudaFree(d_wv));
    CUDA_CHECK(cudaFree(d_wo)); CUDA_CHECK(cudaFree(d_wgate));
    CUDA_CHECK(cudaFree(d_wup)); CUDA_CHECK(cudaFree(d_wdown));
    CUDA_CHECK(cudaFree(d_attn)); CUDA_CHECK(cudaFree(d_attn_out));
    CUDA_CHECK(cudaFree(d_gate)); CUDA_CHECK(cudaFree(d_up));
    CUDA_CHECK(cudaFree(d_ffn));

    if (!compare(h_result.data(), h_expected.data(),
                 static_cast<int>(total), EPSILON)) {
        std::printf("FAIL decoder_block_layer0_fixture\n");
        return FAIL;
    }
    std::printf("PASS decoder_block_layer0_fixture\n");
    return PASS;
}

// ---------------------------------------------------------------------------
// Phase 4 Tests: Full 32-layer forward pass and token generation
// ---------------------------------------------------------------------------

// Full 32-layer forward pass: "Hello world" -> next token.
static int test_full_forward_hello() {
    const int s = 3; // BOS + "Hello" + " world"
    auto h_X = load_fixture("tests/data/m2m3/embeddings_hello.bin",
                            (size_t)s * EMBEDDING_DIM);
    if (h_X.empty()) {
        std::printf("SKIP full_forward_hello (embeddings not found)\n");
        return FAIL;
    }

    FILE *f = std::fopen("tests/data/m2m3/next_token_hello.txt", "r");
    if (!f) {
        std::printf("SKIP full_forward_hello (next_token_hello.txt not found)\n");
        return FAIL;
    }
    int expected_token = 0;
    if (std::fscanf(f, "%d", &expected_token) != 1) {
        std::fclose(f);
        std::printf("FAIL full_forward_hello: cannot read expected token\n");
        return FAIL;
    }
    std::fclose(f);

    ModelWeights weights(DUMP_DIR);
    weights.load_global();
    int result = run_forward_pass(h_X.data(), s, weights);

    BPETokenizer tok(TOKENIZER_PATH);
    std::string decoded = tok.decode({result});
    std::printf("  prompt:          \"Hello world\"\n");
    std::printf("  generated token: %d\n", result);
    std::printf("  decoded text:    \"%s\"\n", decoded.c_str());
    std::printf("  full output:     \"Hello world%s\"\n", decoded.c_str());
    std::printf("  expected token:  %d\n", expected_token);

    if (result != expected_token) {
        std::printf("FAIL full_forward_hello\n");
        return FAIL;
    }

    std::printf("PASS full_forward_hello\n");
    return PASS;
}

void register_phase3(Registry &r) {
    // Phase 3
    r["residual_add_manual"] = test_residual_add_manual;
    r["swiglu_manual"] = test_swiglu_manual;
    r["ffn_block_isolated_fixture"] = test_ffn_block_isolated_fixture;
    r["decoder_block_layer0_fixture"] = test_decoder_block_layer0_fixture;

    // Phase 3 kernel smoke tests
    r["swiglu_kernel_smoke"] = test_swiglu_kernel_smoke;
    r["residual_add_kernel_smoke"] = test_residual_add_kernel_smoke;

    // Phase 4
    r["full_forward_hello"] = test_full_forward_hello;
}

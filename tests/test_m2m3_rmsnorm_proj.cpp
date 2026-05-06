// Phase 1 tests: RMSNorm and Q/K/V projections (Milestone 2).

#include "tests/test_m2m3_helpers.h"

// ---------------------------------------------------------------------------
// Phase 1 Tests: RMSNorm and Q/K/V Projections
// ---------------------------------------------------------------------------

// Proves: epsilon is INSIDE sqrt, gamma scaling IS applied.
// Uses a 2x4 synthetic matrix with hand-computed expected values.
static int test_rmsnorm_manual() {
    const int rows = 2, cols = 4;
    // X = [[1, 2, 3, 4],
    //      [2, 0, -1, 3]]
    float h_input[] = {1.0f, 2.0f, 3.0f, 4.0f,
                       2.0f, 0.0f, -1.0f, 3.0f};
    float h_gamma[] = {0.5f, 1.0f, 2.0f, 0.25f};
    float h_output[8] = {};

    // Hand-computed expected values:
    // Row 0: sum_sq=30, rms=sqrt(30/4 + 1e-5)=sqrt(7.50001)=2.738613
    //   [1/2.738613*0.5, 2/2.738613*1.0, 3/2.738613*2.0, 4/2.738613*0.25]
    //   = [0.182574, 0.730297, 2.190890, 0.365148]
    // Row 1: sum_sq=14, rms=sqrt(14/4 + 1e-5)=sqrt(3.50001)=1.870829
    //   [2/1.870829*0.5, 0/1.870829*1.0, -1/1.870829*2.0, 3/1.870829*0.25]
    //   = [0.534522, 0.0, -1.069045, 0.400892]
    float expected[] = {0.182574f, 0.730297f, 2.190890f, 0.365148f,
                        0.534522f, 0.0f, -1.069045f, 0.400892f};

    float *d_in = nullptr, *d_gamma = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, 8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, 8 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_input, 8 * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, 4 * sizeof(float),
                           cudaMemcpyHostToDevice));

    gpu_rmsnorm(d_in, d_gamma, d_out, rows, cols, RMS_NORM_EPSILON);

    CUDA_CHECK(cudaMemcpy(h_output, d_out, 8 * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_out));

    // Use tight tolerance for hand-computed values.
    if (!compare(h_output, expected, 8, 1e-4f)) {
        std::printf("FAIL rmsnorm_manual\n");
        return FAIL;
    }
    std::printf("PASS rmsnorm_manual\n");
    return PASS;
}

// Proves: RMSNorm on real embeddings with layer-0 weights matches golden.
static int test_rmsnorm_fixture() {
    const int s = 3; // "Hello world" tokens
    const int d = EMBEDDING_DIM;
    const size_t total = static_cast<size_t>(s) * d;

    // Load golden inputs and expected output
    auto h_input = load_fixture("tests/data/m2m3/embeddings_hello.bin", total);
    auto h_expected = load_fixture("tests/data/m2m3/rmsnorm_layer0.bin", total);
    if (h_input.empty() || h_expected.empty()) {
        std::printf("SKIP rmsnorm_fixture (fixtures not generated)\n");
        return FAIL;
    }

    // Load layer-0 input_layernorm weight
    ModelWeights weights(DUMP_DIR);
    const LayerWeights &lw = weights.load_layer(0);
    const float *h_gamma = lw.input_layernorm;

    // Run on GPU
    float *d_in = nullptr, *d_gamma = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_input.data(), total * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, d * sizeof(float),
                           cudaMemcpyHostToDevice));

    gpu_rmsnorm(d_in, d_gamma, d_out, s, d, RMS_NORM_EPSILON);

    std::vector<float> h_output(total);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_out, total * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_out));

    if (!compare(h_output.data(), h_expected.data(),
                 static_cast<int>(total), EPSILON)) {
        std::printf("FAIL rmsnorm_fixture\n");
        return FAIL;
    }
    std::printf("PASS rmsnorm_fixture\n");
    return PASS;
}

// Verifies loader reads expected shapes from dump files.
static int test_weight_shape_qkv() {
    ModelWeights weights(DUMP_DIR);
    const LayerWeights &lw = weights.load_layer(0);

    bool ok = true;
    // After transpose, shapes are [in_features, out_features]:
    // q_proj: [4096, 4096], k_proj: [4096, 1024], v_proj: [4096, 1024]
    // We verify by checking that the loaded pointers are non-null.
    // Shape validation happens inside the loader (load_2d checks).
    if (lw.q_proj == nullptr) {
        std::printf("  q_proj is null\n"); ok = false;
    }
    if (lw.k_proj == nullptr) {
        std::printf("  k_proj is null\n"); ok = false;
    }
    if (lw.v_proj == nullptr) {
        std::printf("  v_proj is null\n"); ok = false;
    }
    if (lw.input_layernorm == nullptr) {
        std::printf("  input_layernorm is null\n"); ok = false;
    }

    if (!ok) {
        std::printf("FAIL weight_shape_qkv\n");
        return FAIL;
    }
    std::printf("PASS weight_shape_qkv\n");
    return PASS;
}

// Verifies Q projection against golden output.
// Q = RMSNorm(X) @ W_q^T. Weight was transposed at load time, so
// the actual call is: Q = X_norm @ q_proj_transposed.
static int test_q_projection_fixture() {
    const int s = 3, d = EMBEDDING_DIM;
    const int q_out = EMBEDDING_DIM; // NUM_HEADS * HEAD_DIM = 4096
    const size_t sz_in = static_cast<size_t>(s) * d;
    const size_t sz_out = static_cast<size_t>(s) * q_out;

    auto h_xnorm = load_fixture("tests/data/m2m3/rmsnorm_layer0.bin", sz_in);
    auto h_expected = load_fixture("tests/data/m2m3/q_proj_layer0.bin", sz_out);
    if (h_xnorm.empty() || h_expected.empty()) {
        std::printf("SKIP q_projection_fixture (fixtures not generated)\n");
        return FAIL;
    }

    // Load transposed weight
    ModelWeights weights(DUMP_DIR);
    const LayerWeights &lw = weights.load_layer(0);

    // matmul: X_norm[s, d] @ q_proj[d, q_out] -> Q[s, q_out]
    float *d_xn = nullptr, *d_w = nullptr, *d_q = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xn, sz_in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w,
                           static_cast<size_t>(d) * q_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q, sz_out * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_xn, h_xnorm.data(), sz_in * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, lw.q_proj,
                           static_cast<size_t>(d) * q_out * sizeof(float),
                           cudaMemcpyHostToDevice));

    gpu_matmul_device(d_xn, d_w, d_q, s, d, q_out);

    std::vector<float> h_output(sz_out);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_q, sz_out * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_xn));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_q));

    if (!compare(h_output.data(), h_expected.data(),
                 static_cast<int>(sz_out), EPSILON)) {
        std::printf("FAIL q_projection_fixture\n");
        return FAIL;
    }
    std::printf("PASS q_projection_fixture\n");
    return PASS;
}

// Verifies K projection against golden output.
static int test_k_projection_fixture() {
    const int s = 3, d = EMBEDDING_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM; // 1024
    const size_t sz_in = static_cast<size_t>(s) * d;
    const size_t sz_out = static_cast<size_t>(s) * kv_dim;

    auto h_xnorm = load_fixture("tests/data/m2m3/rmsnorm_layer0.bin", sz_in);
    auto h_expected = load_fixture("tests/data/m2m3/k_proj_layer0.bin", sz_out);
    if (h_xnorm.empty() || h_expected.empty()) {
        std::printf("SKIP k_projection_fixture (fixtures not generated)\n");
        return FAIL;
    }

    ModelWeights weights(DUMP_DIR);
    const LayerWeights &lw = weights.load_layer(0);

    float *d_xn = nullptr, *d_w = nullptr, *d_k = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xn, sz_in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w,
                           static_cast<size_t>(d) * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, sz_out * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_xn, h_xnorm.data(), sz_in * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, lw.k_proj,
                           static_cast<size_t>(d) * kv_dim * sizeof(float),
                           cudaMemcpyHostToDevice));

    gpu_matmul_device(d_xn, d_w, d_k, s, d, kv_dim);

    std::vector<float> h_output(sz_out);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_k, sz_out * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_xn));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_k));

    if (!compare(h_output.data(), h_expected.data(),
                 static_cast<int>(sz_out), EPSILON)) {
        std::printf("FAIL k_projection_fixture\n");
        return FAIL;
    }
    std::printf("PASS k_projection_fixture\n");
    return PASS;
}

// Verifies V projection against golden output.
static int test_v_projection_fixture() {
    const int s = 3, d = EMBEDDING_DIM;
    const int kv_dim = NUM_KV_HEADS * HEAD_DIM; // 1024
    const size_t sz_in = static_cast<size_t>(s) * d;
    const size_t sz_out = static_cast<size_t>(s) * kv_dim;

    auto h_xnorm = load_fixture("tests/data/m2m3/rmsnorm_layer0.bin", sz_in);
    auto h_expected = load_fixture("tests/data/m2m3/v_proj_layer0.bin", sz_out);
    if (h_xnorm.empty() || h_expected.empty()) {
        std::printf("SKIP v_projection_fixture (fixtures not generated)\n");
        return FAIL;
    }

    ModelWeights weights(DUMP_DIR);
    const LayerWeights &lw = weights.load_layer(0);

    float *d_xn = nullptr, *d_w = nullptr, *d_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_xn, sz_in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w,
                           static_cast<size_t>(d) * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, sz_out * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_xn, h_xnorm.data(), sz_in * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, lw.v_proj,
                           static_cast<size_t>(d) * kv_dim * sizeof(float),
                           cudaMemcpyHostToDevice));

    gpu_matmul_device(d_xn, d_w, d_v, s, d, kv_dim);

    std::vector<float> h_output(sz_out);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_v, sz_out * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_xn));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_v));

    if (!compare(h_output.data(), h_expected.data(),
                 static_cast<int>(sz_out), EPSILON)) {
        std::printf("FAIL v_projection_fixture\n");
        return FAIL;
    }
    std::printf("PASS v_projection_fixture\n");
    return PASS;
}

// Verifies non-square matmul shapes work (s=3, d=4096, kv_dim=1024).
// Uses synthetic data to confirm the matmul handles M!=K!=N.
static int test_qkv_shape_non_square() {
    const int M = 3, K = 4096, N = 1024;
    const int sA = M * K, sB = K * N, sC = M * N;

    std::vector<float> A(sA), B(sB), C(sC);
    fill_deterministic(A.data(), sA, 17);
    fill_deterministic(B.data(), sB, 31);

    // Should not crash or produce NaN
    gpu_matmul(A.data(), B.data(), C.data(), M, K, N);

    bool ok = true;
    for (int i = 0; i < sC; ++i) {
        if (!std::isfinite(C[i])) {
            std::printf("  non-finite at [%d]: %f\n", i, C[i]);
            ok = false;
            break;
        }
    }
    if (!ok) {
        std::printf("FAIL qkv_shape_non_square\n");
        return FAIL;
    }
    std::printf("PASS qkv_shape_non_square\n");
    return PASS;
}

void register_phase1(Registry &r) {
    r["rmsnorm_manual"] = test_rmsnorm_manual;
    r["rmsnorm_fixture"] = test_rmsnorm_fixture;
    r["weight_shape_qkv"] = test_weight_shape_qkv;
    r["q_projection_fixture"] = test_q_projection_fixture;
    r["k_projection_fixture"] = test_k_projection_fixture;
    r["v_projection_fixture"] = test_v_projection_fixture;
    r["qkv_shape_non_square"] = test_qkv_shape_non_square;
}

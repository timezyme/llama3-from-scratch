// Phase 0 tests: matmul parity (host- vs device-pointer, FP32 vs BF16
// weights) and loader/resident-weight smoke tests.

#include "tests/test_m2m3_helpers.h"

// ---------------------------------------------------------------------------
// Phase 0 Tests
// ---------------------------------------------------------------------------

// Verify that the device-pointer matmul matches the host-pointer matmul
// on small 4x4 matrices. This is a unit test for gpu_matmul_device.
static int test_matmul_device_parity_small() {
    const int M = 4, K = 4, N = 4;
    const int sA = M * K, sB = K * N, sC = M * N;

    std::vector<float> A(sA), B(sB), C_host(sC), C_device(sC);
    fill_deterministic(A.data(), sA, 0);
    fill_deterministic(B.data(), sB, 7);

    // Reference: host-pointer path (existing Milestone 1 API)
    gpu_matmul(A.data(), B.data(), C_host.data(), M, K, N);

    // Under test: device-pointer path
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, sA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, sB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, sC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sA * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), sB * sizeof(float),
                           cudaMemcpyHostToDevice));

    gpu_matmul_device(d_A, d_B, d_C, M, K, N);

    CUDA_CHECK(cudaMemcpy(C_device.data(), d_C, sC * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    if (!compare(C_host.data(), C_device.data(), sC, EPSILON)) {
        std::printf("FAIL matmul_device_parity_small\n");
        return FAIL;
    }
    std::printf("PASS matmul_device_parity_small\n");
    return PASS;
}

// Verify parity on a realistic Part-2 shape: (s=3, 4096) x (4096, 1024).
// This catches alignment and tiling edge cases at real model dimensions.
static int test_matmul_device_parity_realistic() {
    const int M = 3, K = 4096, N = 1024;
    const int sA = M * K, sB = K * N, sC = M * N;

    std::vector<float> A(sA), B(sB), C_host(sC), C_device(sC);
    fill_deterministic(A.data(), sA, 42);
    fill_deterministic(B.data(), sB, 99);

    // Reference
    gpu_matmul(A.data(), B.data(), C_host.data(), M, K, N);

    // Under test
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, static_cast<size_t>(sA) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, static_cast<size_t>(sB) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, static_cast<size_t>(sC) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), static_cast<size_t>(sA) * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), static_cast<size_t>(sB) * sizeof(float),
                           cudaMemcpyHostToDevice));

    gpu_matmul_device(d_A, d_B, d_C, M, K, N);

    CUDA_CHECK(cudaMemcpy(C_device.data(), d_C,
                           static_cast<size_t>(sC) * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    if (!compare(C_host.data(), C_device.data(), sC, EPSILON)) {
        std::printf("FAIL matmul_device_parity_realistic\n");
        return FAIL;
    }
    std::printf("PASS matmul_device_parity_realistic\n");
    return PASS;
}

static bool run_bf16_weight_matmul_case(int M, int K, int N,
                                        const char *label) {
    const int sA = M * K, sB = K * N, sC = M * N;

    std::vector<float> A(sA), B(sB), B_bf16_as_float(sB);
    std::vector<uint16_t> B_bf16(sB);
    std::vector<float> C_reference(sC), C_bf16(sC);
    fill_deterministic(A.data(), sA, 123 + M + N);
    fill_deterministic(B.data(), sB, 211 + K);

    for (int i = 0; i < sB; ++i) {
        B_bf16[i] = float_to_bf16_bits(B[i]);
        B_bf16_as_float[i] = bf16_bits_to_float_host(B_bf16[i]);
    }

    float *d_A = nullptr, *d_B_ref = nullptr, *d_C_ref = nullptr,
          *d_C_bf16 = nullptr;
    uint16_t *d_B_bf16 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, static_cast<size_t>(sA) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_ref, static_cast<size_t>(sB) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_bf16,
                          static_cast<size_t>(sB) * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_C_ref, static_cast<size_t>(sC) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_bf16, static_cast<size_t>(sC) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), static_cast<size_t>(sA) * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_ref, B_bf16_as_float.data(),
                          static_cast<size_t>(sB) * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_bf16, B_bf16.data(),
                          static_cast<size_t>(sB) * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));

    gpu_matmul_device(d_A, d_B_ref, d_C_ref, M, K, N);
    gpu_matmul_device_bf16_weight(d_A, d_B_bf16, d_C_bf16, M, K, N);

    CUDA_CHECK(cudaMemcpy(C_reference.data(), d_C_ref,
                          static_cast<size_t>(sC) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(C_bf16.data(), d_C_bf16,
                          static_cast<size_t>(sC) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B_ref));
    CUDA_CHECK(cudaFree(d_B_bf16));
    CUDA_CHECK(cudaFree(d_C_ref));
    CUDA_CHECK(cudaFree(d_C_bf16));

    if (!compare(C_reference.data(), C_bf16.data(), sC, EPSILON)) {
        std::printf("  case %s failed (M=%d K=%d N=%d)\n", label, M, K, N);
        return false;
    }
    std::printf("  case %s passed (M=%d K=%d N=%d)\n", label, M, K, N);
    return true;
}

// BF16-weight path must match the FP32 device matmul when the reference
// weights are rounded to BF16 first.
static int test_bf16_weight_matmul_parity() {
    if (!run_bf16_weight_matmul_case(3, 4096, 1024, "model_kv_width")) {
        std::printf("FAIL bf16_weight_matmul_parity\n");
        return FAIL;
    }
    if (!run_bf16_weight_matmul_case(5, 17, 19, "ragged_edge_tile")) {
        std::printf("FAIL bf16_weight_matmul_parity\n");
        return FAIL;
    }
    std::printf("PASS bf16_weight_matmul_parity\n");
    return PASS;
}

// Proves: the loader can expose validated raw BF16 payloads without widening
// them to FP32. This is the input seam for resident BF16 device weights.
static int test_raw_bf16_loader_parity() {
    LlamaDumpLoader loader(DumpFloatType::BF16);

    const std::string norm_path =
        DUMP_DIR + "/layer_00/model_layers_0_input_layernorm_weight.bin";
    auto raw_norm = loader.load_1d_bf16_raw(norm_path, EMBEDDING_DIM);
    std::unique_ptr<float[]> decoded_norm(
        loader.load_1d(norm_path, EMBEDDING_DIM));
    if (raw_norm.size() != static_cast<size_t>(EMBEDDING_DIM)) {
        std::printf("FAIL raw_bf16_loader_parity: norm size mismatch\n");
        return FAIL;
    }

    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const std::string k_proj_path =
        DUMP_DIR + "/layer_00/model_layers_0_self_attn_k_proj_weight.bin";
    auto raw_k = loader.load_2d_bf16_raw(k_proj_path, kv_dim, EMBEDDING_DIM);
    std::unique_ptr<float[]> decoded_k(
        loader.load_2d(k_proj_path, kv_dim, EMBEDDING_DIM));
    const size_t k_count = static_cast<size_t>(kv_dim) * EMBEDDING_DIM;
    if (raw_k.size() != k_count) {
        std::printf("FAIL raw_bf16_loader_parity: k_proj size mismatch\n");
        return FAIL;
    }

    const size_t norm_indices[] = {0, 1, 127, 1024, EMBEDDING_DIM - 1};
    for (size_t idx : norm_indices) {
        float raw_value = bf16_bits_to_float_host(raw_norm[idx]);
        if (raw_value != decoded_norm[idx]) {
            std::printf("FAIL raw_bf16_loader_parity: norm[%zu] %.8f vs %.8f\n",
                        idx, raw_value, decoded_norm[idx]);
            return FAIL;
        }
    }

    const size_t k_indices[] = {0,
                                1,
                                static_cast<size_t>(EMBEDDING_DIM - 1),
                                static_cast<size_t>(EMBEDDING_DIM),
                                k_count / 2,
                                k_count - 1};
    for (size_t idx : k_indices) {
        float raw_value = bf16_bits_to_float_host(raw_k[idx]);
        if (raw_value != decoded_k[idx]) {
            std::printf("FAIL raw_bf16_loader_parity: k_proj[%zu] %.8f vs %.8f\n",
                        idx, raw_value, decoded_k[idx]);
            return FAIL;
        }
    }

    std::printf("PASS raw_bf16_loader_parity\n");
    return PASS;
}

// Proves: a layer can be loaded into persistent device buffers as transposed
// BF16 projection weights plus FP32 norm weights.
static int test_resident_layer0_weight_smoke() {
    DeviceModelWeights resident(DUMP_DIR);
    const DeviceLayerWeights &layer = resident.load_layer(0);

    if (layer.k_proj == nullptr || layer.input_layernorm == nullptr ||
        layer.device_bytes == 0 || !resident.layer_loaded(0)) {
        std::printf("FAIL resident_layer0_weight_smoke: missing device buffer\n");
        return FAIL;
    }

    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    const size_t k_count = static_cast<size_t>(EMBEDDING_DIM) * kv_dim;
    std::vector<uint16_t> h_k(k_count);
    CUDA_CHECK(cudaMemcpy(h_k.data(), layer.k_proj,
                          k_count * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));

    LlamaDumpLoader loader(DumpFloatType::BF16);
    const std::string k_proj_path =
        DUMP_DIR + "/layer_00/model_layers_0_self_attn_k_proj_weight.bin";
    auto raw_k = loader.load_2d_bf16_raw(k_proj_path, kv_dim, EMBEDDING_DIM);

    const size_t samples[][2] = {{0, 0},
                                 {1, 0},
                                 {0, 1},
                                 {127, 3},
                                 {static_cast<size_t>(EMBEDDING_DIM - 1),
                                  static_cast<size_t>(kv_dim - 1)}};
    for (const auto &sample : samples) {
        size_t in_col = sample[0];
        size_t out_row = sample[1];
        size_t transposed_idx = in_col * kv_dim + out_row;
        size_t raw_idx = out_row * EMBEDDING_DIM + in_col;
        if (h_k[transposed_idx] != raw_k[raw_idx]) {
            std::printf("FAIL resident_layer0_weight_smoke: k[%zu,%zu] "
                        "raw=0x%04x resident=0x%04x\n",
                        in_col, out_row, raw_k[raw_idx],
                        h_k[transposed_idx]);
            return FAIL;
        }
    }

    std::vector<float> h_norm(EMBEDDING_DIM);
    CUDA_CHECK(cudaMemcpy(h_norm.data(), layer.input_layernorm,
                          h_norm.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::unique_ptr<float[]> expected_norm(
        loader.load_1d(DUMP_DIR +
                           "/layer_00/model_layers_0_input_layernorm_weight.bin",
                       EMBEDDING_DIM));
    const size_t norm_indices[] = {0, 17, 1024, EMBEDDING_DIM - 1};
    for (size_t idx : norm_indices) {
        if (h_norm[idx] != expected_norm[idx]) {
            std::printf("FAIL resident_layer0_weight_smoke: norm[%zu] %.8f "
                        "vs %.8f\n",
                        idx, h_norm[idx], expected_norm[idx]);
            return FAIL;
        }
    }

    size_t before_unload = resident.total_device_bytes();
    resident.unload_layer(0);
    if (before_unload == 0 || resident.total_device_bytes() != 0 ||
        resident.layer_loaded(0)) {
        std::printf("FAIL resident_layer0_weight_smoke: unload accounting\n");
        return FAIL;
    }

    std::printf("PASS resident_layer0_weight_smoke\n");
    return PASS;
}

void register_phase0(Registry &r) {
    r["matmul_device_parity_small"] = test_matmul_device_parity_small;
    r["matmul_device_parity_realistic"] = test_matmul_device_parity_realistic;
    r["bf16_weight_matmul_parity"] = test_bf16_weight_matmul_parity;
    r["raw_bf16_loader_parity"] = test_raw_bf16_loader_parity;
    r["resident_layer0_weight_smoke"] = test_resident_layer0_weight_smoke;
}

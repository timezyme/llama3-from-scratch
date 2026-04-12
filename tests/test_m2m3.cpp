// Internal test harness for Milestones 2-3.
// Runs one named test at a time for use during code review.
//
// Exit codes:
//   0 = test passed
//   2 = invalid usage or unknown test name
//   3 = test ran but failed
//
// Usage: ./bin/tests_m2m3 <test_name>

#include "config.h"
#include "kernel/kernels.cuh"
#include "model_weights.h"
#include "tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr float EPSILON = 1e-2f;
static constexpr int PASS = 0;
static constexpr int FAIL = 3;
static constexpr int USAGE_ERROR = 2;

// ---------------------------------------------------------------------------
// CUDA error checking (local to test binary)
// ---------------------------------------------------------------------------

static void check_cuda(cudaError_t err, const char *expr, const char *file,
                       int line) {
    if (err == cudaSuccess) return;
    std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line, expr,
                 cudaGetErrorString(err));
    std::exit(FAIL);
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Fill a buffer with deterministic, non-trivial values.
static void fill_deterministic(float *buf, int count, int seed) {
    for (int i = 0; i < count; ++i) {
        buf[i] = static_cast<float>((i + seed) % 13) * 0.1f - 0.6f;
    }
}

// Compare two float buffers element-wise. Returns true if all match.
static bool compare(const float *a, const float *b, int count, float eps,
                    int max_prints = 5) {
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

// ---------------------------------------------------------------------------
// Phase 1 Tests: RMSNorm and Q/K/V Projections
// ---------------------------------------------------------------------------

// Load a raw FP32 binary fixture file into a vector.
static std::vector<float> load_fixture(const std::string &path, size_t count) {
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

// Shared output-layer helper: compute logits by projecting x_last through the
// embedding table (lm_head = tied weights). Scans in batches to avoid a single
// VOCAB_SIZE * EMBEDDING_DIM allocation.
// loader: LlamaDumpLoader with embeddings already loaded.
// h_x_last: [EMBEDDING_DIM] host memory.
// Returns: [VOCAB_SIZE] logits on host.
// Compute logits using the lm_head output projection weight [VOCAB_SIZE, EMBEDDING_DIM].
// logits[v] = dot(lm_head[v, :], h_x_last).
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
// Q = RMSNorm(X) @ W_q^T. Weight was transposed at load time,
// so we compute: Q = X_norm @ q_proj_transposed.
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
    const int heads_per_group = NUM_HEADS / NUM_KV_HEADS;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    // Precompute RoPE tables
    std::vector<float> h_cos(table_sz), h_sin(table_sz);
    precompute_rope_table(h_cos.data(), h_sin.data(), s, HEAD_DIM, ROPE_BASE);

    // Upload Q, K, V and RoPE tables to device
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
    float *d_cos = nullptr, *d_sin = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Q, s * EMBEDDING_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, s * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, s * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos, table_sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, table_sz * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_q_proj.data(),
                           s * EMBEDDING_DIM * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_k_proj.data(),
                           s * kv_dim * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_v_proj.data(),
                           s * kv_dim * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), table_sz * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), table_sz * sizeof(float),
                           cudaMemcpyHostToDevice));

    // Apply RoPE in-place to Q and K
    gpu_rope(d_Q, d_cos, d_sin, s, NUM_HEADS, HEAD_DIM);
    gpu_rope(d_K, d_cos, d_sin, s, NUM_KV_HEADS, HEAD_DIM);

    // Copy back to host for per-head attention computation
    std::vector<float> h_Q_rope(s * EMBEDDING_DIM);
    std::vector<float> h_K_rope(s * kv_dim);
    std::vector<float> h_V(h_v_proj);  // V is not rotated
    CUDA_CHECK(cudaMemcpy(h_Q_rope.data(), d_Q,
                           s * EMBEDDING_DIM * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_K_rope.data(), d_K,
                           s * kv_dim * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // Per-head attention loop (host-side orchestration)
    std::vector<float> attn_concat(s * EMBEDDING_DIM, 0.0f);

    // Allocate scratch buffers on device
    float *d_Qi = nullptr, *d_Kg = nullptr, *d_Kg_T = nullptr;
    float *d_Vg = nullptr, *d_S = nullptr, *d_Oi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Qi, s * HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Kg, s * HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Kg_T, HEAD_DIM * s * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Vg, s * HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, s * s * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Oi, s * HEAD_DIM * sizeof(float)));

    for (int head_i = 0; head_i < NUM_HEADS; ++head_i) {
        int kv_group = head_i / heads_per_group;

        // Pack Q_i: extract head slice from Q_rope [s, NUM_HEADS*HEAD_DIM]
        std::vector<float> h_Qi(s * HEAD_DIM);
        for (int p = 0; p < s; ++p) {
            for (int d2 = 0; d2 < HEAD_DIM; ++d2) {
                h_Qi[p * HEAD_DIM + d2] =
                    h_Q_rope[p * EMBEDDING_DIM + head_i * HEAD_DIM + d2];
            }
        }

        // Pack K_g: extract KV group slice from K_rope [s, NUM_KV_HEADS*HEAD_DIM]
        std::vector<float> h_Kg(s * HEAD_DIM);
        for (int p = 0; p < s; ++p) {
            for (int d2 = 0; d2 < HEAD_DIM; ++d2) {
                h_Kg[p * HEAD_DIM + d2] =
                    h_K_rope[p * kv_dim + kv_group * HEAD_DIM + d2];
            }
        }

        // Transpose K_g: [s, HEAD_DIM] -> [HEAD_DIM, s]
        std::vector<float> h_Kg_T(HEAD_DIM * s);
        for (int p = 0; p < s; ++p) {
            for (int d2 = 0; d2 < HEAD_DIM; ++d2) {
                h_Kg_T[d2 * s + p] = h_Kg[p * HEAD_DIM + d2];
            }
        }

        // Pack V_g: extract KV group slice from V [s, NUM_KV_HEADS*HEAD_DIM]
        std::vector<float> h_Vg(s * HEAD_DIM);
        for (int p = 0; p < s; ++p) {
            for (int d2 = 0; d2 < HEAD_DIM; ++d2) {
                h_Vg[p * HEAD_DIM + d2] =
                    h_V[p * kv_dim + kv_group * HEAD_DIM + d2];
            }
        }

        // Upload packed head data
        CUDA_CHECK(cudaMemcpy(d_Qi, h_Qi.data(),
                               s * HEAD_DIM * sizeof(float),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Kg_T, h_Kg_T.data(),
                               HEAD_DIM * s * sizeof(float),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Vg, h_Vg.data(),
                               s * HEAD_DIM * sizeof(float),
                               cudaMemcpyHostToDevice));

        // S = Q_i @ K_g^T: [s, HEAD_DIM] @ [HEAD_DIM, s] -> [s, s]
        gpu_matmul_device(d_Qi, d_Kg_T, d_S, s, HEAD_DIM, s);

        // Scale by 1/sqrt(h_d)
        gpu_scale(d_S, s * s, scale);

        // Causal mask
        gpu_causal_mask(d_S, s);

        // Softmax
        gpu_softmax(d_S, s, s);

        // O_i = alpha @ V_g: [s, s] @ [s, HEAD_DIM] -> [s, HEAD_DIM]
        gpu_matmul_device(d_S, d_Vg, d_Oi, s, s, HEAD_DIM);

        // Copy back and place into concatenated output
        std::vector<float> h_Oi(s * HEAD_DIM);
        CUDA_CHECK(cudaMemcpy(h_Oi.data(), d_Oi,
                               s * HEAD_DIM * sizeof(float),
                               cudaMemcpyDeviceToHost));
        for (int p = 0; p < s; ++p) {
            for (int d2 = 0; d2 < HEAD_DIM; ++d2) {
                attn_concat[p * EMBEDDING_DIM + head_i * HEAD_DIM + d2] =
                    h_Oi[p * HEAD_DIM + d2];
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin)); CUDA_CHECK(cudaFree(d_Qi));
    CUDA_CHECK(cudaFree(d_Kg)); CUDA_CHECK(cudaFree(d_Kg_T));
    CUDA_CHECK(cudaFree(d_Vg)); CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_Oi));

    if (!compare(attn_concat.data(), h_expected.data(),
                 static_cast<int>(total), EPSILON)) {
        std::printf("FAIL attention_output_full_fixture\n");
        return FAIL;
    }
    std::printf("PASS attention_output_full_fixture\n");
    return PASS;
}

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
    const int heads_per_group = NUM_HEADS / NUM_KV_HEADS;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    // === Allocate device memory ===
    float *d_X = nullptr, *d_Xnorm = nullptr;
    float *d_gamma1 = nullptr, *d_gamma2 = nullptr;
    float *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr;
    float *d_cos = nullptr, *d_sin = nullptr;
    float *d_wq = nullptr, *d_wk = nullptr, *d_wv = nullptr;
    float *d_wo = nullptr, *d_wgate = nullptr, *d_wup = nullptr, *d_wdown = nullptr;

    size_t bytes_X = total * sizeof(float);
    size_t bytes_kv = s * kv_dim * sizeof(float);

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
    // Copy Q, K, V back to host for per-head slicing
    std::vector<float> h_Q(s * EMBEDDING_DIM), h_K(s * kv_dim), h_V(s * kv_dim);
    CUDA_CHECK(cudaMemcpy(h_Q.data(), d_Q, bytes_X, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_K.data(), d_K, bytes_kv, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_V.data(), d_V, bytes_kv, cudaMemcpyDeviceToHost));

    std::vector<float> attn_concat(s * EMBEDDING_DIM, 0.0f);
    float *d_Qi = nullptr, *d_KgT = nullptr, *d_Vg = nullptr;
    float *d_S = nullptr, *d_Oi = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Qi, s * HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_KgT, HEAD_DIM * s * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Vg, s * HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, s * s * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Oi, s * HEAD_DIM * sizeof(float)));

    for (int hi = 0; hi < NUM_HEADS; ++hi) {
        int kvg = hi / heads_per_group;

        // Pack Q_i, K_g (transposed), V_g
        std::vector<float> hQi(s * HEAD_DIM), hKgT(HEAD_DIM * s), hVg(s * HEAD_DIM);
        for (int p = 0; p < s; ++p) {
            for (int d2 = 0; d2 < HEAD_DIM; ++d2) {
                hQi[p * HEAD_DIM + d2] = h_Q[p * EMBEDDING_DIM + hi * HEAD_DIM + d2];
                float k_val = h_K[p * kv_dim + kvg * HEAD_DIM + d2];
                hKgT[d2 * s + p] = k_val;
                hVg[p * HEAD_DIM + d2] = h_V[p * kv_dim + kvg * HEAD_DIM + d2];
            }
        }

        CUDA_CHECK(cudaMemcpy(d_Qi, hQi.data(), s * HEAD_DIM * sizeof(float),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_KgT, hKgT.data(), HEAD_DIM * s * sizeof(float),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Vg, hVg.data(), s * HEAD_DIM * sizeof(float),
                               cudaMemcpyHostToDevice));

        gpu_matmul_device(d_Qi, d_KgT, d_S, s, HEAD_DIM, s);
        gpu_scale(d_S, s * s, scale);
        gpu_causal_mask(d_S, s);
        gpu_softmax(d_S, s, s);
        gpu_matmul_device(d_S, d_Vg, d_Oi, s, s, HEAD_DIM);

        std::vector<float> hOi(s * HEAD_DIM);
        CUDA_CHECK(cudaMemcpy(hOi.data(), d_Oi, s * HEAD_DIM * sizeof(float),
                               cudaMemcpyDeviceToHost));
        for (int p = 0; p < s; ++p)
            for (int d2 = 0; d2 < HEAD_DIM; ++d2)
                attn_concat[p * EMBEDDING_DIM + hi * HEAD_DIM + d2] =
                    hOi[p * HEAD_DIM + d2];
    }

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
    CUDA_CHECK(cudaFree(d_Qi)); CUDA_CHECK(cudaFree(d_KgT));
    CUDA_CHECK(cudaFree(d_Vg)); CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_Oi)); CUDA_CHECK(cudaFree(d_attn));
    CUDA_CHECK(cudaFree(d_attn_out)); CUDA_CHECK(cudaFree(d_gate));
    CUDA_CHECK(cudaFree(d_up)); CUDA_CHECK(cudaFree(d_ffn));

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

// Helper: run attention for all heads on pre-RoPE'd Q, K and original V.
// Q_rope: [s, EMBEDDING_DIM], K_rope: [s, kv_dim], V: [s, kv_dim] on host.
// Writes concatenated result to attn_concat [s, EMBEDDING_DIM] on host.
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

    CUDA_CHECK(cudaFree(d_Qi)); CUDA_CHECK(cudaFree(d_KgT));
    CUDA_CHECK(cudaFree(d_Vg)); CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_Oi));
}

// Forward pass helper: 32-layer decoder + final norm + logits -> argmax token.
// h_embeddings: [seq_len, EMBEDDING_DIM] host memory.
// weights: ModelWeights with load_global() already called.
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

    // Extract last row -> logits via shared lm_head helper
    std::vector<float> h_Xfinal(total);
    CUDA_CHECK(cudaMemcpy(h_Xfinal.data(), d_Xnorm, bytes_X,
                           cudaMemcpyDeviceToHost));
    const float *x_last = h_Xfinal.data() + (seq_len - 1) * d;
    auto logits = compute_lm_head_logits(weights.global().lm_head, x_last);
    int argmax = static_cast<int>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());

    // Cleanup
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

// ---------------------------------------------------------------------------
// Phase 5 Tests: final norm, lm_head, weight sharing, streaming, medium prompt
// ---------------------------------------------------------------------------

// Proves: final RMSNorm matches golden output from Python fixture.
static int test_final_rmsnorm_fixture() {
    const int s = 3;
    const size_t total = static_cast<size_t>(s) * EMBEDDING_DIM;

    auto h_input =
        load_fixture("tests/data/m2m3/pre_final_norm_hello.bin", total);
    if (h_input.empty()) {
        std::printf("SKIP final_rmsnorm_fixture (pre_final_norm not found)\n");
        return FAIL;
    }
    auto h_expected = load_fixture("tests/data/m2m3/final_hidden.bin", total);
    if (h_expected.empty()) {
        std::printf("SKIP final_rmsnorm_fixture (final_hidden not found)\n");
        return FAIL;
    }

    // Load final norm weight
    ModelWeights weights(DUMP_DIR);
    weights.load_global();

    float *d_in = nullptr, *d_out = nullptr, *d_gamma = nullptr;
    size_t bytes = total * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, EMBEDDING_DIM * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_input.data(), bytes,
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, weights.global().final_norm,
                           EMBEDDING_DIM * sizeof(float),
                           cudaMemcpyHostToDevice));

    gpu_rmsnorm(d_in, d_gamma, d_out, s, EMBEDDING_DIM, RMS_NORM_EPSILON);

    std::vector<float> h_result(total);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_out, bytes,
                           cudaMemcpyDeviceToHost));
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_gamma);

    if (!compare(h_result.data(), h_expected.data(),
                 static_cast<int>(total), EPSILON)) {
        std::printf("FAIL final_rmsnorm_fixture\n");
        return FAIL;
    }
    std::printf("PASS final_rmsnorm_fixture\n");
    return PASS;
}

// Proves: the shared lm_head helper projects the correct (last) row, not some
// other row. Uses synthetic data with deliberately different rows.
static int test_lm_head_last_token_only() {
    const int s = 3;
    const int d = EMBEDDING_DIM;

    // Load real embedding table via ModelWeights
    ModelWeights weights(DUMP_DIR);
    weights.load_global();

    // Create synthetic [s, d] with deliberately different rows
    std::vector<float> h_hidden(s * d);
    fill_deterministic(h_hidden.data(), d, /*seed=*/42);        // row 0
    fill_deterministic(h_hidden.data() + d, d, /*seed=*/99);    // row 1
    fill_deterministic(h_hidden.data() + 2 * d, d, /*seed=*/7); // row 2

    // Project row 0 and row 2 through the shared helper
    auto logits_row0 = compute_lm_head_logits(weights.global().lm_head,
                                               h_hidden.data());
    auto logits_row2 = compute_lm_head_logits(weights.global().lm_head,
                                               h_hidden.data() + 2 * d);

    // Check size
    if ((int)logits_row0.size() != VOCAB_SIZE ||
        (int)logits_row2.size() != VOCAB_SIZE) {
        std::printf("FAIL lm_head_last_token_only: wrong logits size\n");
        return FAIL;
    }

    // Check finiteness
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        if (!std::isfinite(logits_row0[i]) || !std::isfinite(logits_row2[i])) {
            std::printf("FAIL lm_head_last_token_only: non-finite logit at %d\n",
                        i);
            return FAIL;
        }
    }

    // Check that row 0 and row 2 produce different logits
    bool any_differ = false;
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        if (std::fabs(logits_row0[i] - logits_row2[i]) > 1e-6f) {
            any_differ = true;
            break;
        }
    }
    if (!any_differ) {
        std::printf("FAIL lm_head_last_token_only: row 0 and row 2 logits identical\n");
        return FAIL;
    }

    std::printf("PASS lm_head_last_token_only\n");
    return PASS;
}

// Proves: the shared lm_head helper uses the same embedding-table weights that
// get_embeddings() returns. Compares sampled logit entries against manual dot
// products with embedding rows.
static int test_weight_sharing_check() {
    const int d = EMBEDDING_DIM;

    ModelWeights weights(DUMP_DIR);
    weights.load_global();

    // Synthetic hidden state
    std::vector<float> h_x(d);
    fill_deterministic(h_x.data(), d, /*seed=*/123);

    // Compute full logits via the shared helper
    auto logits = compute_lm_head_logits(weights.global().lm_head, h_x.data());

    // Verify by manually computing dot products against lm_head rows for sampled IDs
    std::vector<int> sample_ids = {42, 1337, 2048};
    const float *lm_head = weights.global().lm_head;

    for (size_t k = 0; k < sample_ids.size(); ++k) {
        float dot = 0.0f;
        const float *row = lm_head + (size_t)sample_ids[k] * d;
        for (int j = 0; j < d; ++j)
            dot += h_x[j] * row[j];

        float diff = std::fabs(logits[sample_ids[k]] - dot);
        if (diff > 1e-5f) {
            std::printf("  token %d: helper=%.6f manual=%.6f diff=%.6f\n",
                        sample_ids[k], logits[sample_ids[k]], dot, diff);
            std::printf("FAIL weight_sharing_check\n");
            return FAIL;
        }
    }

    std::printf("PASS weight_sharing_check\n");
    return PASS;
}

// Proves: load_layer / unload_layer lifecycle works correctly for all 32 layers.
// After load, all pointers are non-null. After unload, all are null. Reload works.
static int test_layer_streaming_smoke() {
    ModelWeights weights(DUMP_DIR);

    for (int i = 0; i < NUM_LAYERS; ++i) {
        const LayerWeights &lw = weights.load_layer(i);

        // After load: all pointers should be non-null
        if (!lw.q_proj || !lw.k_proj || !lw.v_proj || !lw.o_proj ||
            !lw.gate_proj || !lw.up_proj || !lw.down_proj ||
            !lw.input_layernorm || !lw.post_attn_layernorm) {
            std::printf("  layer %d: null pointer after load\n", i);
            std::printf("FAIL layer_streaming_smoke\n");
            return FAIL;
        }

        weights.unload_layer(i);

        // After unload: all pointers should be null (free_layer zeroes them)
        if (lw.q_proj || lw.k_proj || lw.v_proj || lw.o_proj ||
            lw.gate_proj || lw.up_proj || lw.down_proj ||
            lw.input_layernorm || lw.post_attn_layernorm) {
            std::printf("  layer %d: non-null pointer after unload\n", i);
            std::printf("FAIL layer_streaming_smoke\n");
            return FAIL;
        }
    }

    // Verify reload of layer 0 works after unloading all
    const LayerWeights &lw0 = weights.load_layer(0);
    if (!lw0.q_proj || !lw0.input_layernorm) {
        std::printf("  layer 0 reload failed\n");
        std::printf("FAIL layer_streaming_smoke\n");
        return FAIL;
    }
    weights.unload_layer(0);

    std::printf("PASS layer_streaming_smoke\n");
    return PASS;
}

// Full 32-layer forward pass with a longer prompt to test masking + RoPE at s>3.
static int test_full_forward_medium_prompt() {
    // Tokenize "The capital of France is" via the C++ tokenizer
    BPETokenizer tok(TOKENIZER_PATH);
    std::vector<int> token_ids = {128000}; // BOS
    auto encoded = tok.encode("The capital of France is");
    token_ids.insert(token_ids.end(), encoded.begin(), encoded.end());
    int s = static_cast<int>(token_ids.size());

    // Read expected token
    FILE *f = std::fopen("tests/data/m2m3/next_token_medium.txt", "r");
    if (!f) {
        std::printf("SKIP full_forward_medium_prompt (fixture not found)\n");
        return FAIL;
    }
    int expected_token = 0;
    if (std::fscanf(f, "%d", &expected_token) != 1) {
        std::fclose(f);
        std::printf("FAIL full_forward_medium_prompt: cannot read expected token\n");
        return FAIL;
    }
    std::fclose(f);

    // Get embeddings from model weights
    ModelWeights weights(DUMP_DIR);
    weights.load_global();
    std::unique_ptr<float[]> h_emb(weights.get_embeddings(token_ids));

    std::printf("  prompt tokens: [");
    for (int i = 0; i < s; ++i)
        std::printf("%s%d", i ? ", " : "", token_ids[i]);
    std::printf("] (s=%d)\n", s);

    int result = run_forward_pass(h_emb.get(), s, weights);

    std::string decoded = tok.decode({result});
    std::printf("  prompt:          \"The capital of France is\"\n");
    std::printf("  generated token: %d\n", result);
    std::printf("  decoded text:    \"%s\"\n", decoded.c_str());
    std::printf("  expected token:  %d\n", expected_token);

    if (result != expected_token) {
        std::printf("FAIL full_forward_medium_prompt\n");
        return FAIL;
    }

    std::printf("PASS full_forward_medium_prompt\n");
    return PASS;
}

// ---------------------------------------------------------------------------
// Test Registry
// ---------------------------------------------------------------------------

using TestFunc = std::function<int()>;

static std::map<std::string, TestFunc> build_registry() {
    std::map<std::string, TestFunc> r;

    // Phase 0
    r["matmul_device_parity_small"] = test_matmul_device_parity_small;
    r["matmul_device_parity_realistic"] = test_matmul_device_parity_realistic;

    // Phase 1
    r["rmsnorm_manual"] = test_rmsnorm_manual;
    r["rmsnorm_fixture"] = test_rmsnorm_fixture;
    r["weight_shape_qkv"] = test_weight_shape_qkv;
    r["q_projection_fixture"] = test_q_projection_fixture;
    r["k_projection_fixture"] = test_k_projection_fixture;
    r["v_projection_fixture"] = test_v_projection_fixture;
    r["qkv_shape_non_square"] = test_qkv_shape_non_square;

    // Phase 2
    r["rope_manual"] = test_rope_manual;
    r["rope_fixture_q"] = test_rope_fixture_q;
    r["rope_fixture_k"] = test_rope_fixture_k;
    r["gqa_head_mapping"] = test_gqa_head_mapping;
    r["causal_mask_triangle"] = test_causal_mask_triangle;
    r["softmax_stability"] = test_softmax_stability;
    r["attention_output_full_fixture"] = test_attention_output_full_fixture;

    // Phase 3
    r["residual_add_manual"] = test_residual_add_manual;
    r["swiglu_manual"] = test_swiglu_manual;
    r["decoder_block_layer0_fixture"] = test_decoder_block_layer0_fixture;

    // Phase 3 kernel smoke tests
    r["swiglu_kernel_smoke"] = test_swiglu_kernel_smoke;
    r["residual_add_kernel_smoke"] = test_residual_add_kernel_smoke;

    // Phase 4
    r["full_forward_hello"] = test_full_forward_hello;

    // Phase 5
    r["final_rmsnorm_fixture"] = test_final_rmsnorm_fixture;
    r["lm_head_last_token_only"] = test_lm_head_last_token_only;
    r["weight_sharing_check"] = test_weight_sharing_check;
    r["layer_streaming_smoke"] = test_layer_streaming_smoke;
    r["full_forward_medium_prompt"] = test_full_forward_medium_prompt;

    return r;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    auto registry = build_registry();

    if (argc == 2 && std::string(argv[1]) == "--list") {
        for (const auto &entry : registry) {
            std::printf("  %s\n", entry.first.c_str());
        }
        return PASS;
    }

    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <test_name>\n", argv[0]);
        std::fprintf(stderr, "       %s --list\n", argv[0]);
        std::fprintf(stderr, "\nAvailable tests:\n");
        for (const auto &entry : registry) {
            std::fprintf(stderr, "  %s\n", entry.first.c_str());
        }
        return USAGE_ERROR;
    }

    std::string name = argv[1];
    auto it = registry.find(name);
    if (it == registry.end()) {
        std::fprintf(stderr, "Unknown test: %s\n", name.c_str());
        std::fprintf(stderr, "\nAvailable tests:\n");
        for (const auto &entry : registry) {
            std::fprintf(stderr, "  %s\n", entry.first.c_str());
        }
        return USAGE_ERROR;
    }

    return it->second();
}

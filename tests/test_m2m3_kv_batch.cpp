// Phase 5 tests: final norm, untied lm_head, layer streaming, KV cache,
// B>1 batched parity, plus the new kv_cache_multi_step_parity test.

#include "tests/test_m2m3_helpers.h"

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

// Proves: the lm_head helper projects the correct last row, not another row.
// llm_part2 §4 requires last-token-only projection before the vocabulary matmul.
static int test_lm_head_last_token_only() {
    const int s = 3;
    const int d = EMBEDDING_DIM;

    // Load global weights via ModelWeights.
    ModelWeights weights(DUMP_DIR);
    weights.load_global();

    // Create synthetic [s, d] with deliberately different rows
    std::vector<float> h_hidden(s * d);
    fill_deterministic(h_hidden.data(), d, /*seed=*/42);        // row 0
    fill_deterministic(h_hidden.data() + d, d, /*seed=*/99);    // row 1
    fill_deterministic(h_hidden.data() + 2 * d, d, /*seed=*/7); // row 2

    // Project row 0 and row 2 through the helper.
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

// Historical test name says "weight sharing", but this checkpoint is untied.
// The test proves compute_lm_head_logits uses the separate lm_head rows
// loaded by ModelWeights, matching config.json tie_word_embeddings=false.
static int test_weight_sharing_check() {
    const int d = EMBEDDING_DIM;

    ModelWeights weights(DUMP_DIR);
    weights.load_global();

    // Synthetic hidden state
    std::vector<float> h_x(d);
    fill_deterministic(h_x.data(), d, /*seed=*/123);

    // Compute full logits via the lm_head helper.
    auto logits = compute_lm_head_logits(weights.global().lm_head, h_x.data());

    // Verify by manually computing dot products against sampled lm_head rows.
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

static int test_embedding_batched_padding() {
    ModelWeights weights(DUMP_DIR);
    weights.load_global();

    // Inference rejects mixed prompt lengths; this keeps the generic padding
    // branch covered for direct embedding callers.
    const std::vector<std::vector<int>> batched_ids = {
        {128000, 882, 128009},
        {128000, 882},
    };

    std::vector<int> lens;
    int smax = 0;
    std::unique_ptr<float[]> h_batched(
        weights.get_embeddings_batched(batched_ids, lens, smax));

    if (lens.size() != batched_ids.size() || lens[0] != 3 || lens[1] != 2 ||
        smax != 3) {
        std::printf("FAIL embedding_batched_padding: bad shape metadata "
                    "(lens=%zu smax=%d)\n",
                    lens.size(), smax);
        return FAIL;
    }

    std::unique_ptr<float[]> h_single(weights.get_embeddings(batched_ids[0]));
    const int d = EMBEDDING_DIM;
    const int row_count = static_cast<int>(batched_ids[0].size());
    if (!compare(h_batched.get(), h_single.get(), row_count * d, 0.0f)) {
        std::printf("FAIL embedding_batched_padding: first batch rows changed\n");
        return FAIL;
    }

    const float *pad_row = h_batched.get() + (static_cast<size_t>(1) * smax + 2) * d;
    for (int i = 0; i < d; ++i) {
        if (pad_row[i] != 0.0f) {
            std::printf("FAIL embedding_batched_padding: pad[%d]=%.8f\n", i,
                        pad_row[i]);
            return FAIL;
        }
    }

    std::printf("PASS embedding_batched_padding\n");
    return PASS;
}

static bool expect_out_of_range(const char *label,
                                const std::function<void()> &fn) {
    try {
        fn();
    } catch (const std::out_of_range &) {
        return true;
    } catch (const std::exception &e) {
        std::printf("FAIL kv_cache_bounds_checks: %s threw wrong error: %s\n",
                    label, e.what());
        return false;
    }

    std::printf("FAIL kv_cache_bounds_checks: %s did not throw\n", label);
    return false;
}

static int test_kv_cache_bounds_checks() {
    KVCache cache(2, 2);

    bool ok = true;
    ok = expect_out_of_range("k_batch layer",
                             [&]() { (void)cache.k_batch(NUM_LAYERS, 0); }) &&
         ok;
    ok = expect_out_of_range("v_batch batch",
                             [&]() { (void)cache.v_batch(0, 2); }) &&
         ok;
    ok = expect_out_of_range("k_at row",
                             [&]() { (void)cache.k_at(0, 2, 0); }) &&
         ok;
    ok = expect_out_of_range("v_at batch",
                             [&]() { (void)cache.v_at(0, 0, -1); }) &&
         ok;

    if (!ok) return FAIL;

    std::printf("PASS kv_cache_bounds_checks\n");
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

// Proves: the KV-cache generation API returns the same first token as the
// single-token next-token path. This is intentionally a full-forward test
// because both paths run the whole model before comparing API behavior.
static int test_full_forward_kv_cache_one_token_parity() {
    const std::string prompt = "The capital of France is";

    ModelWeights single_step_weights(DUMP_DIR);
    int expected = generate_next_token(single_step_weights, prompt);

    ModelWeights cached_weights(DUMP_DIR);
    auto generated = generate_tokens(cached_weights, prompt, 1);
    if (generated.size() != 1) {
        std::printf("FAIL full_forward_kv_cache_one_token_parity: expected 1 "
                    "generated token, got %zu\n",
                    generated.size());
        return FAIL;
    }

    const int actual = generated[0];
    std::printf("  prompt:             \"%s\"\n", prompt.c_str());
    std::printf("  generate_next_token: %d\n", expected);
    std::printf("  generate_tokens[0]:  %d\n", actual);

    if (actual != expected) {
        std::printf("FAIL full_forward_kv_cache_one_token_parity\n");
        return FAIL;
    }

    std::printf("PASS full_forward_kv_cache_one_token_parity\n");
    return PASS;
}

// Proves: the KV-cached multi-step path produces the same per-step tokens as
// running T separate full forwards over the growing chat-templated sequence.
// Catches a position-id update or write-offset bug that takes effect after
// step 1 (T=2 alone could still pass if the bug compounds monotonically).
static int test_kv_cache_multi_step_parity() {
    const std::string prompt = "Hello world";
    const int T = 4;

    // KV-cached path: returns the T newly generated tokens (not the prefix).
    ModelWeights cached_weights(DUMP_DIR);
    auto cached = generate_tokens(cached_weights, prompt, T);
    if (static_cast<int>(cached.size()) != T) {
        std::printf("FAIL kv_cache_multi_step_parity: cached path returned %zu "
                    "tokens (expected %d)\n",
                    cached.size(), T);
        return FAIL;
    }

    // Reference path: T separate full forwards over the chat-templated prefix.
    // Must apply the same chat template that generate_tokens applies internally
    // (src/inference_chat.cu:38), otherwise the two paths run on different
    // sequences and the comparison is invalid.
    BPETokenizer tok(TOKENIZER_PATH);
    ModelWeights ref_weights(DUMP_DIR);
    ref_weights.load_global();
    auto tokens = apply_chat_template(tok, prompt);

    std::vector<int> reference;
    reference.reserve(T);
    for (int step = 0; step < T; ++step) {
        std::unique_ptr<float[]> h_emb(ref_weights.get_embeddings(tokens));
        int next = run_forward_pass(h_emb.get(),
                                    static_cast<int>(tokens.size()),
                                    ref_weights);
        reference.push_back(next);
        tokens.push_back(next);
    }

    if (cached != reference) {
        std::printf("  cached:    [");
        for (size_t i = 0; i < cached.size(); ++i)
            std::printf("%s%d", i ? ", " : "", cached[i]);
        std::printf("]\n");
        std::printf("  reference: [");
        for (size_t i = 0; i < reference.size(); ++i)
            std::printf("%s%d", i ? ", " : "", reference[i]);
        std::printf("]\n");
        std::printf("FAIL kv_cache_multi_step_parity\n");
        return FAIL;
    }

    std::printf("PASS kv_cache_multi_step_parity\n");
    return PASS;
}

// Proves: the resident BF16 layer path preserves the first-token result from
// the existing streaming path before promoting it to the CLI default.
static int test_full_forward_resident_one_token_parity() {
    const std::string prompt = "The capital of France is";

    int expected = -1;
    {
        ModelWeights streaming_weights(DUMP_DIR);
        auto expected_tokens = generate_tokens(streaming_weights, prompt, 1);
        if (expected_tokens.size() != 1) {
            std::printf("FAIL full_forward_resident_one_token_parity: streaming "
                        "path returned %zu tokens\n",
                        expected_tokens.size());
            return FAIL;
        }
        expected = expected_tokens[0];
    }

    ModelWeights resident_global_weights(DUMP_DIR);
    DeviceModelWeights resident_layers(DUMP_DIR);
    auto actual_tokens = generate_tokens_resident(resident_global_weights,
                                                  resident_layers, prompt, 1);
    if (actual_tokens.size() != 1) {
        std::printf("FAIL full_forward_resident_one_token_parity: resident "
                    "path returned %zu tokens\n",
                    actual_tokens.size());
        return FAIL;
    }

    const int actual = actual_tokens[0];
    std::printf("  prompt:            \"%s\"\n", prompt.c_str());
    std::printf("  streaming token:    %d\n", expected);
    std::printf("  resident BF16 token:%d\n", actual);

    if (actual != expected) {
        std::printf("FAIL full_forward_resident_one_token_parity\n");
        return FAIL;
    }

    std::printf("PASS full_forward_resident_one_token_parity\n");
    return PASS;
}

static bool assert_prompt_lengths_match(const std::string &pA,
                                        const std::string &pB,
                                        const char *test_name) {
    BPETokenizer tok(TOKENIZER_PATH);
    auto rA = tok.encode(pA);
    auto rB = tok.encode(pB);
    if (rA.size() != rB.size()) {
        std::printf("FAIL %s: prompt token lengths diverged (A=%zu, B=%zu)\n",
                    test_name, rA.size(), rB.size());
        return false;
    }
    return true;
}

static bool same_tokens(const std::vector<int> &a, const std::vector<int> &b,
                        const char *label) {
    if (a == b) {
        return true;
    }
    std::printf("  %s token mismatch: expected [", label);
    for (size_t i = 0; i < a.size(); ++i) {
        std::printf("%s%d", i ? ", " : "", a[i]);
    }
    std::printf("] got [");
    for (size_t i = 0; i < b.size(); ++i) {
        std::printf("%s%d", i ? ", " : "", b[i]);
    }
    std::printf("]\n");
    return false;
}

static GenerateDebugResult run_debug_generation(
    DeviceModelWeights &resident, const std::vector<std::string> &prompts,
    int max_new_tokens) {
    ModelWeights weights(DUMP_DIR);
    return generate_tokens_resident_debug(weights, resident, prompts,
                                          max_new_tokens);
}

static int test_batched_b2_distinct_parity() {
    const std::string pA = "What is two plus two";
    const std::string pB = "Why does the sun rise";
    if (!assert_prompt_lengths_match(pA, pB, "batched_b2_distinct_parity")) {
        return FAIL;
    }

    DeviceModelWeights resident(DUMP_DIR);
    auto baseline_a = run_debug_generation(resident, {pA}, 1);
    auto baseline_b = run_debug_generation(resident, {pB}, 1);
    auto batched = run_debug_generation(resident, {pA, pB}, 1);

    if (baseline_a.tokens.size() != 1 || baseline_b.tokens.size() != 1 ||
        batched.tokens.size() != 2 || baseline_a.last_hidden.size() !=
        static_cast<size_t>(EMBEDDING_DIM) || baseline_b.last_hidden.size() !=
        static_cast<size_t>(EMBEDDING_DIM) || batched.last_hidden.size() !=
        static_cast<size_t>(2 * EMBEDDING_DIM)) {
        std::printf("FAIL batched_b2_distinct_parity: bad result shape\n");
        return FAIL;
    }

    bool ok = same_tokens(baseline_a.tokens[0], batched.tokens[0], "batch 0") &&
              same_tokens(baseline_b.tokens[0], batched.tokens[1], "batch 1");

    const float diff_a = max_abs_diff(baseline_a.last_hidden.data(),
                                      batched.last_hidden.data(),
                                      EMBEDDING_DIM);
    const float diff_b = max_abs_diff(
        baseline_b.last_hidden.data(),
        batched.last_hidden.data() + EMBEDDING_DIM, EMBEDDING_DIM);
    std::printf("  hidden max abs diff: batch0=%.8f batch1=%.8f\n", diff_a,
                diff_b);
    if (diff_a >= 1e-3f || diff_b >= 1e-3f) {
        ok = false;
    }

    if (!ok) {
        std::printf("FAIL batched_b2_distinct_parity\n");
        return FAIL;
    }

    std::printf("PASS batched_b2_distinct_parity\n");
    return PASS;
}

void register_phase5(Registry &r) {
    r["final_rmsnorm_fixture"] = test_final_rmsnorm_fixture;
    r["lm_head_last_token_only"] = test_lm_head_last_token_only;
    r["weight_sharing_check"] = test_weight_sharing_check;
    r["layer_streaming_smoke"] = test_layer_streaming_smoke;
    r["embedding_batched_padding"] = test_embedding_batched_padding;
    r["kv_cache_bounds_checks"] = test_kv_cache_bounds_checks;
    r["full_forward_medium_prompt"] = test_full_forward_medium_prompt;
    r["full_forward_kv_cache_one_token_parity"] =
        test_full_forward_kv_cache_one_token_parity;
    r["kv_cache_multi_step_parity"] = test_kv_cache_multi_step_parity;
    r["full_forward_resident_one_token_parity"] =
        test_full_forward_resident_one_token_parity;
    r["batched_b2_distinct_parity"] = test_batched_b2_distinct_parity;
}

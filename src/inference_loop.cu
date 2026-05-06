// Orchestrator paths for the inference module:
//   - generate_next_token_impl              one prompt, one greedy token
//   - generate_tokens_impl                  one prompt, KV-cached decode loop
//   - generate_tokens_resident_batched_impl B>1 prompts, KV-cached decode loop
//
// Each path tokenizes through the chat template, builds the device
// RoPE tables and KV cache, calls `forward_step` for prefill and any
// decode iterations, and projects the last hidden state through
// lm_head on the host. Stops on EOT or `max_new_tokens`.

#include "config.h"
#include "device_weights.h"
#include "inference.h"
#include "inference_internal.h"
#include "instrument.h"
#include "kernel/kernels.cuh"
#include "kv_cache.h"
#include "model_weights.h"
#include "tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// Wrap a CUDA runtime call and turn failures into C++ exceptions. This keeps
// allocation/copy call sites readable while still surfacing the CUDA error
// string immediately. Keep the trailing backslashes: they make the do/while
// block part of the macro expansion.
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(std::string("CUDA error: ") +          \
                                     cudaGetErrorString(err));              \
        }                                                                   \
    } while (0)

// Build the RoPE cos/sin tables on the host and upload them to VRAM
// once at the start of generation. This is exactly the optimization
// llm_part2 §3.2 recommends ("Pre-compute cos(p*theta_i) and
// sin(p*theta_i) for all positions ... before the forward pass and
// store them on the GPU"). After this call the device kernel does
// pure table reads instead of per-thread cos/sin transcendentals.
//
// We size the tables to S_MAX rather than the prompt length so a
// single allocation covers both prefill and any decode positions
// without re-uploading. Caller owns the returned device pointers
// (free with cudaFree).
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

// Pre-load all 32 layers' BF16 weights into VRAM if a resident-weights
// holder was provided. This is paid once per process; later forward
// steps reuse the resident copy. Telemetry prints VRAM use so we can
// compare resident weights with the KV cache.
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

// Verify every prompt in a batch tokenized to the same length. Returns
// the common length on success. Mixed-length batching is intentionally
// out of scope; refusing here gives a clear error before
// the forward pass shapes go off the rails.
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
                "(mixed-length batching is not supported)");
        }
    }
    return s;
}

// B>1 generation with resident weights and a KV cache. Every step advances
// `batch` prompts in lockstep through the same forward pass. Finished
// slots keep feeding EOT tokens so the batch shape and KV layout stay fixed.
//
// Returns: per-prompt token IDs plus the last hidden state for any
// debug callers that want to verify numerics.
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
    KVCache cache(S_MAX, batch);   // per-layer device tensors for K/V
    probe_vram("after_kvcache_alloc");
    load_resident_layers(&resident_weights);

    float *d_cos = nullptr, *d_sin = nullptr;
    alloc_rope_tables(&d_cos, &d_sin);

    // ----- Prefill: one forward pass over the whole prompt for all batch slots -----
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

    // Argmax-decode the first generated token per slot, mark done if EOT.
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

    // ----- Decode: one new token per step for every active batch slot -----
    for (int step = 1; step < max_new_tokens; ++step) {
        if (std::all_of(done.begin(), done.end(), [](bool v) { return v; })) {
            break;
        }

        // Build a B*1 batch of next-token IDs to embed.
        std::vector<std::vector<int>> one_ids(batch);
        for (int b = 0; b < batch; ++b) {
            // Finished slots keep advancing as EOT rows so the batch
            // dimensions (and KV cache layout) stay constant. We just
            // ignore the resulting tokens for those slots when decoding
            // logits below. This wastes a small amount of compute but
            // avoids resharding the cache mid-generation.
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

// Single-token (greedy) generation. This is the path used by the M2-3
// grading test: one full forward pass over the prompt, return the
// argmax. No decode loop; allocates a KV cache anyway (single prefill
// fills it to len = prompt_len) for code-path uniformity.
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

// Single-prompt KV-cached multi-token generation.
//
// Flow: tokenize -> chat-template -> prefill (one pass over the whole
// prompt, populates the KV cache) -> decode loop (one new token per
// pass, each pass projects Q for that one token and reads the entire
// cached K/V prefix). Stops on EOT or after max_new_tokens.
//
// Without KV caching, every decode step would have to re-encode the
// entire growing sequence — quadratic compute. With caching the prefix
// work is paid once during prefill and each decode step is linear in
// kv_seq.
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

// End-to-end Llama 3 8B Instruct inference (llm_part2 §3.1).
// Each block runs RMSNorm -> Q/K/V -> RoPE -> GQA -> output projection
// -> residual -> RMSNorm -> SwiGLU FFN -> residual, repeated for 32 layers.
// RMSNorm is root-mean-square layer normalization; RoPE is rotary
// position embedding; GQA is Grouped Query Attention; FFN is feed-forward.
//
// lm_head differs from the assignment pitfall text: llm_part2 §4 says
// it is tied to embeddings, but assets/llama3/config.json sets
// tie_word_embeddings=false. This checkpoint loads lm_head separately.
// The output path still follows §4 by projecting only x_last to V=128256.
//
// Three forward-pass shapes are implemented:
//   - generate_next_token: prefill only, returns one argmax token. Used by
//     the M2-3 grading test and the cheapest CLI path.
//   - generate_tokens:     prefill + decode loop with a KV cache. Each
//     decode step projects Q for the single new token, appends one K/V
//     row to the cache, and attends over the full cached prefix.
//   - generate_tokens (B>1, batched): same prefill/decode with B prompts
//     advancing in lockstep. Requires equal-length tokenizations because
//     mixed-length batching is out of scope.
//
// This file owns the public API surface declared in
// `include/inference.h`. Each `generate_*` entry is a thin delegator
// to a `*_impl` function in `inference_loop.cu`. The chat template,
// per-step decoder block, and orchestrators live in
// `inference_chat.cu`, `inference_layer.cu`, and `inference_loop.cu`
// respectively.

#include "config.h"
#include "inference.h"
#include "inference_internal.h"
#include "tokenizer.h"

// Public entry for "one prompt, one greedy token". Streams weights
// from disk per layer (no resident GPU copy). This is the M2-3 test
// path and the cheapest CLI path.
int generate_next_token(ModelWeights &weights, const std::string &prompt) {
    return generate_next_token_impl(weights, nullptr, prompt);
}

// Public entry for "one prompt, up to N tokens" with KV cache, but with
// per-layer streaming weights. Used by tests that want decode behavior
// without paying the resident-weight upload.
std::vector<int> generate_tokens(ModelWeights &weights,
                                 const std::string &prompt,
                                 int max_new_tokens) {
    return generate_tokens_impl(weights, nullptr, prompt, max_new_tokens);
}

// Resident-weights variants: weights live in VRAM as BF16 across all
// calls (warmup pays the upload once). These are the paths the CLI
// uses for multi-token and batched inference.
int generate_next_token_resident(ModelWeights &weights,
                                 DeviceModelWeights &resident_weights,
                                 const std::string &prompt) {
    return generate_next_token_impl(weights, &resident_weights, prompt);
}

std::vector<int> generate_tokens_resident(ModelWeights &weights,
                                          DeviceModelWeights &resident_weights,
                                          const std::string &prompt,
                                          int max_new_tokens) {
    return generate_tokens_impl(weights, &resident_weights, prompt,
                                max_new_tokens);
}

// Batched (B>1) public entry.
std::vector<std::vector<int>> generate_tokens_resident(
    ModelWeights &weights, DeviceModelWeights &resident_weights,
    const std::vector<std::string> &prompts, int max_new_tokens) {
    return generate_tokens_resident_batched_impl(weights, resident_weights,
                                                 prompts, max_new_tokens)
        .tokens;
}

// Same as the batched path, but also returns the last hidden state per
// slot — useful for numerical-parity tests against reference.py.
GenerateDebugResult generate_tokens_resident_debug(
    ModelWeights &weights, DeviceModelWeights &resident_weights,
    const std::vector<std::string> &prompts, int max_new_tokens) {
    return generate_tokens_resident_batched_impl(weights, resident_weights,
                                                 prompts, max_new_tokens);
}

// Detokenize a single token ID using a process-lifetime tokenizer.
// `static` keeps the BPE tables resident across calls (decoding one
// token per CLI print-step is otherwise dominated by tokenizer setup).
std::string decode_token(int token_id) {
    static BPETokenizer tok(TOKENIZER_PATH);
    return tok.decode({token_id});
}

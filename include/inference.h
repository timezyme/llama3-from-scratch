// Public inference API for Llama 3 8B Instruct.
//
// Three forward-pass shapes are exposed, picked based on what the
// caller is doing:
//
//   - generate_next_token: one prompt, one greedy token. The cheapest
//     path; used by the M2-3 grading tests.
//   - generate_tokens (single prompt): KV-cached multi-token decode.
//     Tokenizes the prompt, runs prefill, then loops decode steps until
//     EOT or max_new_tokens.
//   - generate_tokens (vector of prompts): B>1 batched generation.
//     Requires equal-length tokenizations.
//
// Each shape has a `_resident` variant that uses BF16 weights kept on
// the GPU across calls (paid once at startup), so a long REPL session
// or a multi-prompt batch doesn't re-upload weights every call.

#pragma once

#include "device_weights.h"
#include "model_weights.h"

#include <string>
#include <vector>

// Run one prefill pass: prompt -> next token ID.
// Tokenizes the prompt, runs all 32 decoder layers, applies the output
// layer, and returns the argmax token ID.
int generate_next_token(ModelWeights &weights, const std::string &prompt);

// Multi-token autoregressive generation using a KV cache.
// Tokenizes the prompt, runs the prefill pass (populating the cache),
// then runs up to `max_new_tokens` decode steps. Stops on EOT or limit.
// Returns the list of generated token IDs (in addition to the prompt).
std::vector<int> generate_tokens(ModelWeights &weights,
                                 const std::string &prompt,
                                 int max_new_tokens);

// Same generation APIs, but decoder layers are read from resident BF16 device
// buffers instead of being streamed from disk for each step.
int generate_next_token_resident(ModelWeights &weights,
                                 DeviceModelWeights &resident_weights,
                                 const std::string &prompt);

std::vector<int> generate_tokens_resident(ModelWeights &weights,
                                          DeviceModelWeights &resident_weights,
                                          const std::string &prompt,
                                          int max_new_tokens);

std::vector<std::vector<int>> generate_tokens_resident(
    ModelWeights &weights, DeviceModelWeights &resident_weights,
    const std::vector<std::string> &prompts, int max_new_tokens);

// Returned by the _debug variant for tests that need to compare the
// final hidden state against reference.py before lm_head reduces it
// down to a single token ID.
struct GenerateDebugResult {
    std::vector<std::vector<int>> tokens;       // generated IDs per batch slot
    std::vector<float> last_hidden;             // [batch, EMBEDDING_DIM]
};

GenerateDebugResult generate_tokens_resident_debug(
    ModelWeights &weights, DeviceModelWeights &resident_weights,
    const std::vector<std::string> &prompts, int max_new_tokens);

// Decode a token ID back to text using the BPE tokenizer.
std::string decode_token(int token_id);

// Inference pipeline for Llama 3 8B.
// Provides the shared forward-pass entry points used by both
// the CLI executable and the internal test binary.

#pragma once

#include "model_weights.h"

#include <string>
#include <vector>

// Run one forward pass: prompt -> next token ID.
// Tokenizes the prompt, runs all 32 decoder layers, applies the output
// layer, and returns the argmax token ID. No KV cache; full sequence
// is recomputed each call.
int generate_next_token(ModelWeights &weights, const std::string &prompt);

// Multi-token autoregressive generation using a KV cache.
// Tokenizes the prompt, runs the prefill pass (populating the cache),
// then runs up to `max_new_tokens` decode steps. Stops on EOT or limit.
// Returns the list of generated token IDs (in addition to the prompt).
std::vector<int> generate_tokens(ModelWeights &weights,
                                 const std::string &prompt,
                                 int max_new_tokens);

// Decode a token ID back to text using the BPE tokenizer.
std::string decode_token(int token_id);

// Inference pipeline for Llama 3 8B.
// Provides the shared forward-pass entry points used by both
// the CLI executable and the internal test binary.

#pragma once

#include "model_weights.h"

#include <string>
#include <vector>

// Run one forward pass: prompt -> next token ID.
// Tokenizes the prompt, runs all 32 decoder layers, applies the output
// layer, and returns the argmax token ID.
int generate_next_token(ModelWeights &weights, const std::string &prompt);

// Decode a token ID back to text using the BPE tokenizer.
std::string decode_token(int token_id);

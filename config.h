// Llama 3 8B Instruct model constants and file paths.
// All architecture hyperparameters match the HuggingFace config.json.

#pragma once

#include <string>

// File paths (relative to the working directory at runtime).
inline const std::string TOKENIZER_PATH = "assets/llama3/token.model";
inline const std::string DUMP_DIR = "assets/llama3/dump";

// Embedding and normalization.
inline constexpr int EMBEDDING_DIM = 4096;
inline constexpr float RMS_NORM_EPSILON = 1e-5f;

// Attention: 32 query heads, 8 KV heads (4:1 GQA ratio).
inline constexpr int NUM_HEADS = 32;
inline constexpr int NUM_KV_HEADS = 8;
inline constexpr int HEAD_DIM = 128;

// Feed-forward network (SwiGLU).
inline constexpr int FFN_DIM = 14336;

// Vocabulary and sequence encoding.
inline constexpr int VOCAB_SIZE = 128256;
inline constexpr float ROPE_BASE = 500000.0f; // Llama 3 uses 500k, not 10k

// Decoder stack depth.
inline constexpr int NUM_LAYERS = 32;

static_assert(HEAD_DIM == EMBEDDING_DIM / NUM_HEADS, "HEAD_DIM mismatch");
static_assert(EMBEDDING_DIM == NUM_HEADS * HEAD_DIM, "dimension consistency");

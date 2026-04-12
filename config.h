#pragma once

#include <string>

inline const std::string TOKENIZER_PATH = "assets/llama3/token.model";
inline const std::string DUMP_DIR = "assets/llama3/dump";

// Embedding and model dimensions
inline constexpr int EMBEDDING_DIM = 4096;
inline constexpr float RMS_NORM_EPSILON = 1e-5f;

// Llama 3 8B architecture constants
inline constexpr int NUM_HEADS = 32;
inline constexpr int NUM_KV_HEADS = 8;
inline constexpr int HEAD_DIM = 128;
inline constexpr int FFN_DIM = 14336;
inline constexpr int VOCAB_SIZE = 128256;
inline constexpr float ROPE_BASE = 500000.0f;
inline constexpr int NUM_LAYERS = 32;

static_assert(HEAD_DIM == EMBEDDING_DIM / NUM_HEADS, "HEAD_DIM mismatch");
static_assert(EMBEDDING_DIM == NUM_HEADS * HEAD_DIM, "dimension consistency");

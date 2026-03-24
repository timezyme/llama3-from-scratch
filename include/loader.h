// Binary weight loader for Llama 3 model dumps.
// Reads files produced by tools/dumper.py (safetensors -> 280-byte header + payload).
// Supports FP32, FP16, and BF16 payloads, converting all values to FP32 on load.

#pragma once
#include "prelude.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Declared precision of the dump payload (matches dumper.py output).
enum class DumpFloatType { FP16, BF16, FP32 };

class LlamaDumpLoader {
  public:
    DumpFloatType float_type;

    // Construct a loader configured for a given float type.
    // Actual file paths and dimensions are provided per-call.
    explicit LlamaDumpLoader(DumpFloatType float_type);

    ~LlamaDumpLoader();

    // --- Embedding table helpers ---
    // The embedding dump is a 2D tensor [vocab_size, embedding_dim].
    // load_embeddings() caches the raw blob so get_embeddings() can
    // decode individual rows without re-reading the file.

    // Return the vocab size from the embedding dump header.
    // Loads and caches the file on first call.
    size_t vocab_size(const std::string &dump_path, int embedding_dim);

    // Load and cache the raw embedding dump blob. Returns false on mismatch.
    bool load_embeddings(const std::string &dump_path, int embedding_dim);

    // Look up embedding rows for the given token IDs.
    // Returns a newly allocated FP32 buffer [token_ids.size() x embedding_dim].
    // Caller owns the result (free with delete[]).
    float_t *get_embeddings(const std::vector<int> &token_ids);

    // --- Generic tensor loaders ---
    // Load a 1D tensor (e.g. bias, RMS norm weights) with shape validation.
    float_t *load_1d(const std::string &dump_file, size_t dim0);

    // Load a 2D tensor (e.g. weight matrix) with shape validation.
    float_t *load_2d(const std::string &dump_file, size_t dim0, size_t dim1);

  private:
    // Cached embedding blob and metadata to avoid re-reading the file.
    std::vector<uint8_t> embeddings_blob_;
    size_t embeddings_payload_offset_ = 0;
    size_t embeddings_vocab_size_ = 0;
    int embeddings_dim_ = 0;
    uint32_t embeddings_dtype_code_ = 0;
    std::string embeddings_source_file_;
};

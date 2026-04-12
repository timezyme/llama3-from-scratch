// Model weight management for Llama 3 8B inference.
// Loads weights from binary dump files, transposes 2D projection weights
// at load time, and supports layer-by-layer streaming to conserve GPU memory.

#pragma once

#include "config.h"
#include "loader.h"

#include <string>
#include <vector>

// Per-layer weight set for one decoder block.
// All 2D projection weights are transposed at load time so that
// gpu_matmul_device(X, weight, result, s, in_dim, out_dim) works directly.
// Stored on the host; caller uploads to GPU as needed.
struct LayerWeights {
    // Attention projections (transposed at load: [in_features, out_features])
    float *q_proj = nullptr;      // transposed: [4096, 4096]
    float *k_proj = nullptr;      // transposed: [4096, 1024]
    float *v_proj = nullptr;      // transposed: [4096, 1024]
    float *o_proj = nullptr;      // transposed: [4096, 4096]

    // FFN projections (transposed at load: [in_features, out_features])
    float *gate_proj = nullptr;   // transposed: [4096, 14336]
    float *up_proj = nullptr;     // transposed: [4096, 14336]
    float *down_proj = nullptr;   // transposed: [14336, 4096]

    // Norm weights (1D, no transpose needed)
    float *input_layernorm = nullptr;     // [EMBEDDING_DIM]
    float *post_attn_layernorm = nullptr; // [EMBEDDING_DIM]

    // Non-copyable: pointers are owning (freed by ModelWeights::free_layer).
    LayerWeights() = default;
    LayerWeights(const LayerWeights &) = delete;
    LayerWeights &operator=(const LayerWeights &) = delete;
    LayerWeights(LayerWeights &&) = default;
    LayerWeights &operator=(LayerWeights &&) = default;
};

// Global model weights shared across all layers.
struct GlobalWeights {
    // Final RMSNorm weight before the output layer.
    float *final_norm = nullptr;  // [EMBEDDING_DIM]

    // Output projection (lm_head): [VOCAB_SIZE, EMBEDDING_DIM].
    // Llama 3 Instruct does NOT tie lm_head to the embedding table
    // (config.json: tie_word_embeddings = false).
    // Stored as loaded (row-major [V, d]); logits = lm_head @ x_last.
    float *lm_head = nullptr;    // [VOCAB_SIZE * EMBEDDING_DIM]
};

// Loads and manages all Llama 3 8B model weights.
// Supports layer-by-layer loading to stay within GPU memory budget.
class ModelWeights {
  public:
    explicit ModelWeights(const std::string &dump_dir);
    ~ModelWeights();

    // Non-copyable
    ModelWeights(const ModelWeights &) = delete;
    ModelWeights &operator=(const ModelWeights &) = delete;

    // Load global weights: embeddings and final norm.
    void load_global();

    // Load all weights for a specific layer (0 to NUM_LAYERS-1).
    // Caches the result; subsequent calls return immediately.
    const LayerWeights &load_layer(int layer_idx);

    // Free a specific layer's host-side weight buffers.
    void unload_layer(int layer_idx);

    // Access global weights.
    const GlobalWeights &global() const { return global_; }

    // Embedding lookup: returns FP32 buffer [token_ids.size(), EMBEDDING_DIM].
    // Caller owns the returned pointer (delete[]).
    float *get_embeddings(const std::vector<int> &token_ids);

    // Access the underlying loader for direct embedding table access.
    LlamaDumpLoader &loader() { return loader_; }

  private:
    std::string dump_dir_;
    LlamaDumpLoader loader_;
    GlobalWeights global_;
    LayerWeights layers_[NUM_LAYERS];
    bool layer_loaded_[NUM_LAYERS] = {};

    // Build the dump file path for a layer tensor.
    // e.g., layer_path(0, "self_attn.q_proj.weight")
    //       -> "dump_dir/layers.0.self_attn.q_proj.weight.bin"
    std::string layer_path(int layer_idx, const std::string &tensor_name) const;

    // Transpose a row-major [rows, cols] matrix to [cols, rows].
    // Caller owns the returned pointer (delete[]).
    static float *transpose(const float *src, size_t rows, size_t cols);

    // Free all buffers in a single layer.
    void free_layer(int layer_idx);
};

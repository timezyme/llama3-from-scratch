// Model weight loading for Llama 3 8B.
// Reads binary dump files, validates shapes, and transposes 2D weights
// so that the runtime matmul path always consumes contiguous row-major
// matrices without additional transpose.

#include "model_weights.h"

#include <memory>
#include <sstream>
#include <stdexcept>

ModelWeights::ModelWeights(const std::string &dump_dir)
    : dump_dir_(dump_dir), loader_(DumpFloatType::BF16) {}

ModelWeights::~ModelWeights() {
    for (int i = 0; i < NUM_LAYERS; ++i) {
        if (layer_loaded_[i]) {
            free_layer(i);
        }
    }
    delete[] global_.final_norm;
    global_.final_norm = nullptr;
    delete[] global_.lm_head;
    global_.lm_head = nullptr;
}

void ModelWeights::load_global() {
    // Load and cache the embedding table for row lookups.
    if (!loader_.load_embeddings(dump_dir_ + "/embeddings.bin", EMBEDDING_DIM)) {
        throw std::runtime_error("failed to load embeddings from " + dump_dir_);
    }

    // Load the final RMSNorm weight.
    global_.final_norm =
        loader_.load_1d(dump_dir_ + "/global/model_norm_weight.bin",
                        EMBEDDING_DIM);

    // Load the lm_head output projection (NOT tied to embeddings in Llama 3 Instruct).
    global_.lm_head =
        loader_.load_2d(dump_dir_ + "/global/lm_head_weight.bin",
                        VOCAB_SIZE, EMBEDDING_DIM);
}

float *ModelWeights::get_embeddings(const std::vector<int> &token_ids) {
    return loader_.get_embeddings(token_ids);
}

const LayerWeights &ModelWeights::load_layer(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= NUM_LAYERS) {
        throw std::runtime_error("layer index out of range");
    }
    if (layer_loaded_[layer_idx]) {
        return layers_[layer_idx];
    }

    LayerWeights &lw = layers_[layer_idx];

    const int kv_dim = NUM_KV_HEADS * HEAD_DIM; // 1024

    // --- 1D norm weights (no transpose) ---
    lw.input_layernorm =
        loader_.load_1d(layer_path(layer_idx, "input_layernorm_weight"),
                        EMBEDDING_DIM);
    lw.post_attn_layernorm =
        loader_.load_1d(layer_path(layer_idx, "post_attention_layernorm_weight"),
                        EMBEDDING_DIM);

    // --- 2D projection weights (load as [out, in], transpose to [in, out]) ---

    // Q projection: dump [4096, 4096] -> transposed [4096, 4096]
    {
        std::unique_ptr<float[]> raw(loader_.load_2d(
            layer_path(layer_idx, "self_attn_q_proj_weight"),
            EMBEDDING_DIM, EMBEDDING_DIM));
        lw.q_proj = transpose(raw.get(), EMBEDDING_DIM, EMBEDDING_DIM);
    }

    // K projection: dump [1024, 4096] -> transposed [4096, 1024]
    {
        std::unique_ptr<float[]> raw(loader_.load_2d(
            layer_path(layer_idx, "self_attn_k_proj_weight"),
            kv_dim, EMBEDDING_DIM));
        lw.k_proj = transpose(raw.get(), kv_dim, EMBEDDING_DIM);
    }

    // V projection: dump [1024, 4096] -> transposed [4096, 1024]
    {
        std::unique_ptr<float[]> raw(loader_.load_2d(
            layer_path(layer_idx, "self_attn_v_proj_weight"),
            kv_dim, EMBEDDING_DIM));
        lw.v_proj = transpose(raw.get(), kv_dim, EMBEDDING_DIM);
    }

    // O projection: dump [4096, 4096] -> transposed [4096, 4096]
    {
        std::unique_ptr<float[]> raw(loader_.load_2d(
            layer_path(layer_idx, "self_attn_o_proj_weight"),
            EMBEDDING_DIM, EMBEDDING_DIM));
        lw.o_proj = transpose(raw.get(), EMBEDDING_DIM, EMBEDDING_DIM);
    }

    // Gate projection: dump [14336, 4096] -> transposed [4096, 14336]
    {
        std::unique_ptr<float[]> raw(loader_.load_2d(
            layer_path(layer_idx, "mlp_gate_proj_weight"),
            FFN_DIM, EMBEDDING_DIM));
        lw.gate_proj = transpose(raw.get(), FFN_DIM, EMBEDDING_DIM);
    }

    // Up projection: dump [14336, 4096] -> transposed [4096, 14336]
    {
        std::unique_ptr<float[]> raw(loader_.load_2d(
            layer_path(layer_idx, "mlp_up_proj_weight"),
            FFN_DIM, EMBEDDING_DIM));
        lw.up_proj = transpose(raw.get(), FFN_DIM, EMBEDDING_DIM);
    }

    // Down projection: dump [4096, 14336] -> transposed [14336, 4096]
    {
        std::unique_ptr<float[]> raw(loader_.load_2d(
            layer_path(layer_idx, "mlp_down_proj_weight"),
            EMBEDDING_DIM, FFN_DIM));
        lw.down_proj = transpose(raw.get(), EMBEDDING_DIM, FFN_DIM);
    }

    layer_loaded_[layer_idx] = true;
    return layers_[layer_idx];
}

void ModelWeights::unload_layer(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= NUM_LAYERS) {
        return;
    }
    if (!layer_loaded_[layer_idx]) {
        return;
    }
    free_layer(layer_idx);
    layer_loaded_[layer_idx] = false;
}

std::string ModelWeights::layer_path(int layer_idx,
                                     const std::string &tensor_name) const {
    // Dump layout: dump_dir/layer_XX/model_layers_X_<tensor>.bin
    // Directory uses zero-padded 2-digit index (layer_00, layer_01, ...)
    // Filename uses non-padded index (model_layers_0_, model_layers_1_, ...)
    std::ostringstream dir;
    dir << dump_dir_ << "/layer_";
    if (layer_idx < 10) dir << "0";
    dir << layer_idx;

    std::ostringstream file;
    file << dir.str() << "/model_layers_" << layer_idx << "_" << tensor_name
         << ".bin";
    return file.str();
}

float *ModelWeights::transpose(const float *src, size_t rows, size_t cols) {
    float *dst = new float[rows * cols];
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    return dst;
}

void ModelWeights::free_layer(int layer_idx) {
    LayerWeights &lw = layers_[layer_idx];
    delete[] lw.q_proj;             lw.q_proj = nullptr;
    delete[] lw.k_proj;             lw.k_proj = nullptr;
    delete[] lw.v_proj;             lw.v_proj = nullptr;
    delete[] lw.o_proj;             lw.o_proj = nullptr;
    delete[] lw.gate_proj;          lw.gate_proj = nullptr;
    delete[] lw.up_proj;            lw.up_proj = nullptr;
    delete[] lw.down_proj;          lw.down_proj = nullptr;
    delete[] lw.input_layernorm;    lw.input_layernorm = nullptr;
    delete[] lw.post_attn_layernorm; lw.post_attn_layernorm = nullptr;
}

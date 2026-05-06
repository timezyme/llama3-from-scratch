// Per-layer host-side weight management for Llama 3 8B.
// Checkpoint projection weights are [out_features, in_features], so we
// transpose once at load time to [in, out]. That makes runtime matmul
// calls match llm_part2 §4's X * W^T math without an extra kernel.
// Layers load/unload individually for the streaming inference path.

#include "model_weights.h"

#include <algorithm>
#include <cstring>
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

// Load the model-wide tensors that don't belong to any single decoder
// layer: the embedding table, the final RMSNorm gamma, and the lm_head
// output projection.
//
// Llama 3 8B Instruct sets `tie_word_embeddings = false` in its
// config.json, so lm_head is a separate 128256x4096 tensor (not the
// embedding table). The Part 2 assignment text says "shared with the
// embedding table"; that is true for vanilla Llama 3 but NOT for the
// instruct variant we use, which is why we always load lm_head as its
// own tensor here.
void ModelWeights::load_global() {
    // Cache the raw embedding payload; rows are decoded on demand.
    if (!loader_.load_embeddings(dump_dir_ + "/embeddings.bin", EMBEDDING_DIM)) {
        throw std::runtime_error("failed to load embeddings from " + dump_dir_);
    }

    // model.norm.weight (gamma for the final RMSNorm before lm_head).
    global_.final_norm =
        loader_.load_1d(dump_dir_ + "/global/model_norm_weight.bin",
                        EMBEDDING_DIM);

    // lm_head.weight as [V, d]. Stored row-major so logits = lm_head @ x_last.
    global_.lm_head =
        loader_.load_2d(dump_dir_ + "/global/lm_head_weight.bin",
                        VOCAB_SIZE, EMBEDDING_DIM);
}

// Single-prompt embedding lookup — thin pass-through to the loader.
float *ModelWeights::get_embeddings(const std::vector<int> &token_ids) {
    return loader_.get_embeddings(token_ids);
}

// Batched embedding lookup for B>1 inference. Output is a flat
// [batch, s_max, d] tensor where shorter prompts are zero-padded at
// the END (after their valid tokens). out_lens reports the original
// per-prompt length so callers can ignore padding when reading
// last-token logits. out_smax is the longest prompt length seen.
//
// Note: this function supports unequal-length batches at the embedding
// layer, but the rest of the inference path currently requires equal
// lengths (validate_equal_lengths in inference.cu enforces that).
float *ModelWeights::get_embeddings_batched(
    const std::vector<std::vector<int>> &batched_ids,
    std::vector<int> &out_lens, int &out_smax) {
    if (batched_ids.empty()) {
        throw std::runtime_error("get_embeddings_batched: empty batch");
    }

    out_lens.clear();
    out_lens.reserve(batched_ids.size());
    out_smax = 0;
    for (const auto &ids : batched_ids) {
        const int len = static_cast<int>(ids.size());
        if (len <= 0) {
            throw std::runtime_error(
                "get_embeddings_batched: empty token sequence");
        }
        out_lens.push_back(len);
        out_smax = std::max(out_smax, len);
    }

    const size_t batch = batched_ids.size();
    const size_t row_width = static_cast<size_t>(EMBEDDING_DIM);
    const size_t total = batch * out_smax * row_width;
    std::unique_ptr<float[]> stacked(new float[total]());

    for (size_t b = 0; b < batch; ++b) {
        std::unique_ptr<float[]> single(loader_.get_embeddings(batched_ids[b]));
        const size_t rows = batched_ids[b].size();
        float *dst = stacked.get() + (b * out_smax) * row_width;
        std::memcpy(dst, single.get(), rows * row_width * sizeof(float));
    }

    return stacked.release();
}

// Load all 9 tensors for a decoder layer: 2 RMSNorm gammas, 4 attention
// projections, and 3 FFN/feed-forward projections. Idempotent: if the
// layer was already loaded, returns the cached struct. Each 2D weight is
// transposed in CPU memory so the caller's matmul never has to.
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

// Drop all CPU buffers for one layer's tensors. The streaming path
// loads layer N, uploads it to the GPU, frees the CPU copy, then moves
// to layer N+1.
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

// Build the path to a per-layer tensor's dump file.
// Layout produced by tools/dumper.py:
//   dump_dir/layer_XX/model_layers_X_<tensor>.bin
// where the directory uses a zero-padded 2-digit index (layer_00..31)
// and the filename uses a non-padded index (model_layers_0..31_).
// The asymmetry mirrors HuggingFace's safetensors-shard naming so the
// dumper script doesn't have to remap indices.
std::string ModelWeights::layer_path(int layer_idx,
                                     const std::string &tensor_name) const {
    std::ostringstream dir;
    dir << dump_dir_ << "/layer_";
    if (layer_idx < 10) dir << "0";
    dir << layer_idx;

    std::ostringstream file;
    file << dir.str() << "/model_layers_" << layer_idx << "_" << tensor_name
         << ".bin";
    return file.str();
}

// Transpose a row-major [rows, cols] FP32 matrix into a new
// [cols, rows] buffer. Heap-allocated; caller owns the result.
// Used at load time to flip every projection weight from the
// HuggingFace [out, in] layout to the [in, out] layout our matmul
// path consumes.
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

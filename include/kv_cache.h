// Key/Value cache for optional incremental autoregressive decoding.
// llm_part1 §3.1.1 makes KV caching optional, but the layout here
// follows llm_part2 §3.3's discussion of avoiding repeated prefix work.
//
// Holds, per decoder layer, two flat device buffers shaped
//   [batch, max_len, NUM_KV_HEADS * HEAD_DIM]
// for K and V. The K/V projection matmuls write each new token's
// per-head K and V vectors straight into the next free row of these
// buffers. Attention then reads the full prefix [0, kv_seq) from the
// cache instead of recomputing K and V for prior tokens.
//
// Why allocate up to max_len up front: avoids ever resizing or
// migrating the buffer mid-generation, which would either break the
// pointers held by the kernels or force a large device-to-device
// copy. The cost is bounded VRAM: at the project's S_MAX=1024,
// 32 layers x 1024 rows x (NUM_KV_HEADS * HEAD_DIM = 1024) x 4 bytes
// x 2 (K and V) = 256 MB at batch=1.
//
// Lifetime: cudaMalloc on construction, cudaFree on destruction.
// reset() rewinds the logical length without freeing memory so a REPL
// session can reuse the same cache for back-to-back prompts.

#pragma once

#include "config.h"

#include <stdexcept>

class KVCache {
public:
    // Allocate device-side K/V buffers for all layers.
    // Throws std::runtime_error on cudaMalloc failure.
    explicit KVCache(int max_seq_len, int batch = 1);

    ~KVCache();

    KVCache(const KVCache &) = delete;
    KVCache &operator=(const KVCache &) = delete;

    // Reset logical length to 0 without freeing buffers.
    void reset() { len_ = 0; }

    // Number of tokens currently cached.
    int len() const { return len_; }

    // Maximum sequence length capacity.
    int max_len() const { return max_len_; }

    // Batch capacity.
    int batch() const { return batch_; }

    // Advance length after appending `n` rows. Throws if len + n exceeds max_len.
    void advance(int n);

    // Per-layer K buffer pointer for batch 0 (device). Layout: [max_len, kv_dim].
    float *k(int layer) const { return k_batch(layer, 0); }

    // Per-layer V buffer pointer for batch 0 (device). Layout: [max_len, kv_dim].
    float *v(int layer) const { return v_batch(layer, 0); }

    // Per-layer K batch slice (device). Layout: [max_len, kv_dim].
    float *k_batch(int layer, int b) const {
        check_layer(layer);
        check_batch(b);
        return d_K_[layer] + static_cast<size_t>(b) * max_len_ * kv_dim();
    }

    // Per-layer V batch slice (device). Layout: [max_len, kv_dim].
    float *v_batch(int layer, int b) const {
        check_layer(layer);
        check_batch(b);
        return d_V_[layer] + static_cast<size_t>(b) * max_len_ * kv_dim();
    }

    // Pointer offset into K[layer] at batch `b`, row `row`.
    float *k_at(int layer, int row, int b = 0) const {
        check_layer(layer);
        check_row(row);
        check_batch(b);
        return d_K_[layer] +
               (static_cast<size_t>(b) * max_len_ + row) * kv_dim();
    }

    // Pointer offset into V[layer] at batch `b`, row `row`.
    float *v_at(int layer, int row, int b = 0) const {
        check_layer(layer);
        check_row(row);
        check_batch(b);
        return d_V_[layer] +
               (static_cast<size_t>(b) * max_len_ + row) * kv_dim();
    }

    static constexpr int kv_dim() { return NUM_KV_HEADS * HEAD_DIM; }

private:
    int max_len_ = 0;
    int batch_ = 1;
    int len_ = 0;
    float *d_K_[NUM_LAYERS] = {};
    float *d_V_[NUM_LAYERS] = {};

    void check_layer(int layer) const {
        if (layer < 0 || layer >= NUM_LAYERS) {
            throw std::out_of_range("KVCache: layer index out of range");
        }
    }

    void check_row(int row) const {
        if (row < 0 || row >= max_len_) {
            throw std::out_of_range("KVCache: row index out of range");
        }
    }

    void check_batch(int b) const {
        if (b < 0 || b >= batch_) {
            throw std::out_of_range("KVCache: batch index out of range");
        }
    }

    void free_all();
};

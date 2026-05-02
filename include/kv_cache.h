// KV cache for incremental autoregressive decoding.
//
// Stores per-layer K and V on the device so each generation step only
// needs to project Q for the new token and append one K/V row instead
// of recomputing the full sequence.
//
// Layout: per layer, K and V are flat row-major buffers of shape
// [s_max, NUM_KV_HEADS * HEAD_DIM]. New tokens are written at offset
// `len * NUM_KV_HEADS * HEAD_DIM` by the caller (typically the K/V
// projection matmul writes directly into the cache slot).
//
// Lifetime: allocations happen once per cache; reset() rewinds len
// without freeing memory.

#pragma once

#include "config.h"

class KVCache {
public:
    // Allocate device-side K/V buffers for all layers.
    // Throws std::runtime_error on cudaMalloc failure.
    explicit KVCache(int max_seq_len);

    ~KVCache();

    KVCache(const KVCache &) = delete;
    KVCache &operator=(const KVCache &) = delete;

    // Reset logical length to 0 without freeing buffers.
    void reset() { len_ = 0; }

    // Number of tokens currently cached.
    int len() const { return len_; }

    // Maximum sequence length capacity.
    int max_len() const { return max_len_; }

    // Advance length after appending `n` rows. Asserts len + n <= max_len.
    void advance(int n);

    // Per-layer K buffer pointer (device). Layout: [max_len, kv_dim].
    float *k(int layer) const { return d_K_[layer]; }

    // Per-layer V buffer pointer (device). Layout: [max_len, kv_dim].
    float *v(int layer) const { return d_V_[layer]; }

    // Pointer offset into K[layer] at row `row`.
    float *k_at(int layer, int row) const {
        return d_K_[layer] + static_cast<size_t>(row) * kv_dim();
    }

    // Pointer offset into V[layer] at row `row`.
    float *v_at(int layer, int row) const {
        return d_V_[layer] + static_cast<size_t>(row) * kv_dim();
    }

    static constexpr int kv_dim() { return NUM_KV_HEADS * HEAD_DIM; }

private:
    int max_len_ = 0;
    int len_ = 0;
    float *d_K_[NUM_LAYERS] = {};
    float *d_V_[NUM_LAYERS] = {};

    void free_all();
};

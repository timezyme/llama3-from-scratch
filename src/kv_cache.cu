// KVCache implementation for the optional caching extension mentioned in
// llm_part1 §3.1.1 and llm_part2 §3.3.
//
// Holds one [batch, max_len, kv_dim] device buffer per layer for K and
// for V. During prefill, K/V projection matmuls write rows
// [len_before .. len_before+q_seq) into each layer's buffers; during
// decode (q_seq=1) we append a single row per step. Attention then
// reads the full [0, kv_seq) prefix of K and V without recomputing
// the prompt tokens on every decode step.
//
// Without caching, llm_part2 §3.3 says generation reprocesses growing
// sequences of length s0+1 through s0+T. With caching, each decode step
// projects one new token and attends over the cached prefix.

#include "kv_cache.h"

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

namespace {

void cuda_check(cudaError_t err, const char *expr, const char *file, int line) {
    if (err == cudaSuccess) return;
    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << " for " << expr << ": "
        << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
}

} // namespace

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

// Allocate device-side K and V buffers for every layer. Each is sized
// for the full [batch, max_seq_len, kv_dim] capacity up front so we
// never have to reallocate (or copy) mid-generation. With this
// project's S_MAX=1024 and batch=1, this is 256 MiB total.
KVCache::KVCache(int max_seq_len, int batch)
    : max_len_(max_seq_len), batch_(batch), len_(0) {
    if (max_seq_len <= 0) {
        throw std::runtime_error("KVCache: max_seq_len must be positive");
    }
    if (batch <= 0) {
        throw std::runtime_error("KVCache: batch must be positive");
    }
    const size_t bytes = static_cast<size_t>(batch) * max_seq_len * kv_dim() *
                         sizeof(float);
    try {
        for (int i = 0; i < NUM_LAYERS; ++i) {
            CUDA_CHECK(cudaMalloc(&d_K_[i], bytes));
            CUDA_CHECK(cudaMalloc(&d_V_[i], bytes));
        }
    } catch (...) {
        free_all();
        throw;
    }
}

KVCache::~KVCache() { free_all(); }

// Bump the logical token count after `n` rows of K/V have been
// written into the cache. The buffers themselves are written by the
// K/V projection matmuls in inference.cu (they target k_at()/v_at()
// directly); this function is bookkeeping only.
void KVCache::advance(int n) {
    if (n <= 0) return;
    if (len_ + n > max_len_) {
        throw std::runtime_error("KVCache: advance past max_len");
    }
    len_ += n;
}

// Free every layer's K and V buffer. Called by the destructor and by
// the constructor's catch handler to roll back partial allocations.
void KVCache::free_all() {
    for (int i = 0; i < NUM_LAYERS; ++i) {
        if (d_K_[i]) {
            cudaFree(d_K_[i]);
            d_K_[i] = nullptr;
        }
        if (d_V_[i]) {
            cudaFree(d_V_[i]);
            d_V_[i] = nullptr;
        }
    }
}

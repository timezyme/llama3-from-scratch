// KVCache implementation: device-side allocations for per-layer K/V tensors.

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

KVCache::KVCache(int max_seq_len) : max_len_(max_seq_len), len_(0) {
    if (max_seq_len <= 0) {
        throw std::runtime_error("KVCache: max_seq_len must be positive");
    }
    const size_t bytes = static_cast<size_t>(max_seq_len) * kv_dim() *
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

void KVCache::advance(int n) {
    if (n <= 0) return;
    if (len_ + n > max_len_) {
        throw std::runtime_error("KVCache: advance past max_len");
    }
    len_ += n;
}

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

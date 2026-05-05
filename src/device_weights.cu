// Resident (always-on-GPU) BF16 weights for the L4 inference path.
//
// The streaming path moves each layer disk -> CPU RAM -> PCIe -> VRAM
// (video RAM) on every forward step. This path instead uploads once and
// reuses resident layer weights across decode steps.
//
// We transpose during load, matching model_weights.cpp: checkpoint
// [out, in] becomes [in, out] in VRAM. The only runtime difference from
// the FP32 streaming path is the BF16-weight matmul variant.

#include "device_weights.h"

#include <cuda_runtime.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

void check_cuda(cudaError_t err, const char *expr, const char *file, int line) {
    if (err == cudaSuccess) {
        return;
    }
    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << " for " << expr << ": "
        << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

// Allocate `count * sizeof(T)` bytes of VRAM and return the pointer.
// Throws on failure (the CUDA_CHECK macro converts cudaError into a
// runtime_error so callers get a single exception path).
template <typename T>
T *device_alloc(size_t count) {
    T *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)));
    return ptr;
}

// Transpose a row-major BF16 matrix on the host before uploading to
// VRAM. We bake the transpose in here so the resident-VRAM path uses
// the same [in, out] convention as the streaming path — the kernel
// code never has to special-case which weight layout it sees.
std::vector<uint16_t> transpose_bf16(const std::vector<uint16_t> &src,
                                     size_t rows, size_t cols) {
    std::vector<uint16_t> dst(rows * cols);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    return dst;
}

void free_ptr(uint16_t *&ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

void free_ptr(float *&ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

} // namespace

DeviceModelWeights::DeviceModelWeights(const std::string &dump_dir)
    : dump_dir_(dump_dir), loader_(DumpFloatType::BF16) {}

DeviceModelWeights::~DeviceModelWeights() {
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        free_layer(layer);
    }
}

// Load and upload one decoder layer's tensors into VRAM. Idempotent:
// returns the cached struct if already loaded. Throws (and frees
// whatever has been allocated for this layer) on any failure to
// avoid leaving partially-allocated layers behind.
const DeviceLayerWeights &DeviceModelWeights::load_layer(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= NUM_LAYERS) {
        throw std::runtime_error("DeviceModelWeights: layer index out of range");
    }
    if (layer_loaded_[layer_idx]) {
        return layers_[layer_idx];
    }

    const int kv_dim = NUM_KV_HEADS * HEAD_DIM;
    DeviceLayerWeights &lw = layers_[layer_idx];
    size_t bytes = 0;

    try {
        lw.input_layernorm = load_fp32_1d_device(
            layer_path(layer_idx, "input_layernorm_weight"), EMBEDDING_DIM,
            bytes);
        lw.device_bytes += bytes;
        lw.post_attn_layernorm = load_fp32_1d_device(
            layer_path(layer_idx, "post_attention_layernorm_weight"),
            EMBEDDING_DIM, bytes);
        lw.device_bytes += bytes;

        lw.q_proj = load_bf16_transposed(
            layer_path(layer_idx, "self_attn_q_proj_weight"), EMBEDDING_DIM,
            EMBEDDING_DIM, bytes);
        lw.device_bytes += bytes;
        lw.k_proj = load_bf16_transposed(
            layer_path(layer_idx, "self_attn_k_proj_weight"), kv_dim,
            EMBEDDING_DIM, bytes);
        lw.device_bytes += bytes;
        lw.v_proj = load_bf16_transposed(
            layer_path(layer_idx, "self_attn_v_proj_weight"), kv_dim,
            EMBEDDING_DIM, bytes);
        lw.device_bytes += bytes;
        lw.o_proj = load_bf16_transposed(
            layer_path(layer_idx, "self_attn_o_proj_weight"), EMBEDDING_DIM,
            EMBEDDING_DIM, bytes);
        lw.device_bytes += bytes;
        lw.gate_proj = load_bf16_transposed(
            layer_path(layer_idx, "mlp_gate_proj_weight"), FFN_DIM,
            EMBEDDING_DIM, bytes);
        lw.device_bytes += bytes;
        lw.up_proj = load_bf16_transposed(
            layer_path(layer_idx, "mlp_up_proj_weight"), FFN_DIM,
            EMBEDDING_DIM, bytes);
        lw.device_bytes += bytes;
        lw.down_proj = load_bf16_transposed(
            layer_path(layer_idx, "mlp_down_proj_weight"), EMBEDDING_DIM,
            FFN_DIM, bytes);
        lw.device_bytes += bytes;
    } catch (...) {
        free_layer(layer_idx);
        throw;
    }

    layer_loaded_[layer_idx] = true;
    total_device_bytes_ += lw.device_bytes;
    return lw;
}

// Pre-load every decoder layer. Called once at process startup so the
// generation loop never has to upload weights mid-stream.
void DeviceModelWeights::load_all_layers() {
    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        load_layer(layer);
    }
}

void DeviceModelWeights::unload_layer(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= NUM_LAYERS) {
        return;
    }
    free_layer(layer_idx);
}

bool DeviceModelWeights::layer_loaded(int layer_idx) const {
    if (layer_idx < 0 || layer_idx >= NUM_LAYERS) {
        return false;
    }
    return layer_loaded_[layer_idx];
}

size_t DeviceModelWeights::layer_device_bytes(int layer_idx) const {
    if (layer_idx < 0 || layer_idx >= NUM_LAYERS) {
        return 0;
    }
    return layers_[layer_idx].device_bytes;
}

std::string DeviceModelWeights::layer_path(
    int layer_idx, const std::string &tensor_name) const {
    std::ostringstream dir;
    dir << dump_dir_ << "/layer_";
    if (layer_idx < 10) {
        dir << "0";
    }
    dir << layer_idx;

    std::ostringstream file;
    file << dir.str() << "/model_layers_" << layer_idx << "_" << tensor_name
         << ".bin";
    return file.str();
}

void DeviceModelWeights::free_layer(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= NUM_LAYERS) {
        return;
    }

    DeviceLayerWeights &lw = layers_[layer_idx];
    if (layer_loaded_[layer_idx]) {
        total_device_bytes_ -= lw.device_bytes;
    }

    free_ptr(lw.q_proj);
    free_ptr(lw.k_proj);
    free_ptr(lw.v_proj);
    free_ptr(lw.o_proj);
    free_ptr(lw.gate_proj);
    free_ptr(lw.up_proj);
    free_ptr(lw.down_proj);
    free_ptr(lw.input_layernorm);
    free_ptr(lw.post_attn_layernorm);

    lw.device_bytes = 0;
    layer_loaded_[layer_idx] = false;
}

// Read a BF16 dump as raw uint16 bits, transpose on the host, and
// upload to VRAM. The whole pipeline stays in BF16 — no FP32 detour —
// so the host RAM peak is roughly half what the FP32 path would use.
uint16_t *DeviceModelWeights::load_bf16_transposed(
    const std::string &path, size_t rows, size_t cols, size_t &bytes) {
    auto raw = loader_.load_2d_bf16_raw(path, rows, cols);
    auto transposed = transpose_bf16(raw, rows, cols);
    uint16_t *device = device_alloc<uint16_t>(transposed.size());
    bytes = transposed.size() * sizeof(uint16_t);
    CUDA_CHECK(cudaMemcpy(device, transposed.data(), bytes,
                          cudaMemcpyHostToDevice));
    return device;
}

// Load a 1D tensor as FP32 (no transpose, no widening from BF16).
// Used for the per-layer RMSNorm gammas — they are small and stay in
// FP32 throughout for simpler RMSNorm kernel arithmetic.
float *DeviceModelWeights::load_fp32_1d_device(const std::string &path,
                                               size_t dim, size_t &bytes) {
    std::unique_ptr<float[]> host(loader_.load_1d(path, dim));
    float *device = device_alloc<float>(dim);
    bytes = dim * sizeof(float);
    CUDA_CHECK(cudaMemcpy(device, host.get(), bytes, cudaMemcpyHostToDevice));
    return device;
}

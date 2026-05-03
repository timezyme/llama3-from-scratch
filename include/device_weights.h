// Device-resident BF16 model weights for the L4 optimization path.
//
// This does not replace ModelWeights yet. It is the resident-weight owner used
// by the next inference path once parity is proven.

#pragma once

#include "config.h"
#include "loader.h"

#include <cstddef>
#include <cstdint>
#include <string>

struct DeviceLayerWeights {
    uint16_t *q_proj = nullptr;    // [4096, 4096] BF16, transposed
    uint16_t *k_proj = nullptr;    // [4096, 1024] BF16, transposed
    uint16_t *v_proj = nullptr;    // [4096, 1024] BF16, transposed
    uint16_t *o_proj = nullptr;    // [4096, 4096] BF16, transposed
    uint16_t *gate_proj = nullptr; // [4096, 14336] BF16, transposed
    uint16_t *up_proj = nullptr;   // [4096, 14336] BF16, transposed
    uint16_t *down_proj = nullptr; // [14336, 4096] BF16, transposed

    float *input_layernorm = nullptr;     // [4096] FP32
    float *post_attn_layernorm = nullptr; // [4096] FP32

    size_t device_bytes = 0;
};

class DeviceModelWeights {
  public:
    explicit DeviceModelWeights(const std::string &dump_dir);
    ~DeviceModelWeights();

    DeviceModelWeights(const DeviceModelWeights &) = delete;
    DeviceModelWeights &operator=(const DeviceModelWeights &) = delete;

    const DeviceLayerWeights &load_layer(int layer_idx);
    void load_all_layers();
    void unload_layer(int layer_idx);

    bool layer_loaded(int layer_idx) const;
    size_t layer_device_bytes(int layer_idx) const;
    size_t total_device_bytes() const { return total_device_bytes_; }

  private:
    std::string dump_dir_;
    LlamaDumpLoader loader_;
    DeviceLayerWeights layers_[NUM_LAYERS];
    bool layer_loaded_[NUM_LAYERS] = {};
    size_t total_device_bytes_ = 0;

    std::string layer_path(int layer_idx, const std::string &tensor_name) const;
    void free_layer(int layer_idx);

    uint16_t *load_bf16_transposed(const std::string &path, size_t rows,
                                   size_t cols, size_t &bytes);
    float *load_fp32_1d_device(const std::string &path, size_t dim,
                               size_t &bytes);
};

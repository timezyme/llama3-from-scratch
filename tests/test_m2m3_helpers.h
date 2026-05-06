// Shared declarations for the M2-3 test binary.
//
// The Phase 0 split groups tests by milestone phase across multiple TUs.
// This header carries the symbols that need external linkage so each
// per-group TU can call them without duplicating definitions.
//
// Local helpers used by exactly one group stay `static` in their TU.

#pragma once

#include "config.h"
#include "device_weights.h"
#include "inference.h"
#include "kernel/kernels.cuh"
#include "kv_cache.h"
#include "model_weights.h"
#include "tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

inline constexpr float EPSILON = 1e-2f;
inline constexpr int PASS = 0;
inline constexpr int FAIL = 3;
inline constexpr int USAGE_ERROR = 2;

// ---------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------

void check_cuda(cudaError_t err, const char *expr, const char *file, int line);

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Helpers shared across TUs
// ---------------------------------------------------------------------------

void fill_deterministic(float *buf, int count, int seed);
uint16_t float_to_bf16_bits(float value);
float bf16_bits_to_float_host(uint16_t bits);
bool compare(const float *a, const float *b, int count, float eps,
             int max_prints = 5);
float max_abs_diff(const float *a, const float *b, int count);
std::vector<float> load_fixture(const std::string &path, size_t count);
void run_attention_heads(const std::vector<float> &h_Q_rope,
                         const std::vector<float> &h_K_rope,
                         const std::vector<float> &h_V,
                         std::vector<float> &attn_concat, int s);
int run_forward_pass(const float *h_embeddings, int seq_len,
                     ModelWeights &weights);

// ---------------------------------------------------------------------------
// Registry plumbing
// ---------------------------------------------------------------------------

using TestFunc = std::function<int()>;
using Registry = std::map<std::string, TestFunc>;

void register_phase0(Registry &r);
void register_phase1(Registry &r);
void register_phase2(Registry &r);
void register_phase3(Registry &r);
void register_phase5(Registry &r);

// Defined in src/inference_chat.cu; consumed by kv_cache_multi_step_parity
// in test_m2m3_kv_batch.cpp. The internal declaration lives in
// src/inference_internal.h, which is not exported -- this extern keeps the
// test TU off the inference internal header.
extern std::vector<int> apply_chat_template(const BPETokenizer &tok,
                                            const std::string &prompt);

// Defined in src/inference_layer.cu (production lm_head projection).
// Re-declared here so test TUs can call it without including the private
// inference_internal.h. Defining it again in helpers.cpp would be an ODR
// violation -- inference_layer.o is linked into bin/tests_m2m3.
extern std::vector<float> compute_lm_head_logits(const float *lm_head,
                                                 const float *h_x_last);

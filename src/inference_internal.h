// Project-internal cross-file declarations for the inference module.
//
// Do not include from outside `src/inference*.cu`. In particular, do not
// include from `include/inference.h` (would create a cycle).
//
// This header carries the small set of helpers and constants that
// `inference.cu`, `inference_chat.cu`, `inference_layer.cu`, and
// `inference_loop.cu` share after the source split. The public API in
// `include/inference.h` is unchanged.

#pragma once

#include "inference.h"

#include <string>
#include <vector>

class BPETokenizer;
class ModelWeights;
class DeviceModelWeights;
class KVCache;

// Sequence-length cap for the KV cache and RoPE table. The assignment
// (llm_part1 §3.1.1) bounds the prompt at 1000 tokens; we round up to
// 1024 for headroom plus generated tokens.
inline constexpr int S_MAX = 1024;

// <|eot_id|>: end-of-turn sentinel emitted by the Llama 3 Instruct
// tokenizer. Decoding stops as soon as a slot produces this ID.
inline constexpr int EOT_ID = 128009;

// Wrap a raw prompt in the Llama 3 Instruct chat template, returning
// the full token-ID stream that primes the model to produce an
// assistant reply. Defined in `inference_chat.cu`.
std::vector<int> apply_chat_template(const BPETokenizer &tok,
                                     const std::string &prompt);

// Project the last hidden vector through lm_head on the host.
// Defined in `inference_layer.cu`.
std::vector<float> compute_lm_head_logits(const float *lm_head,
                                          const float *h_x_last);

// One forward step through all 32 decoder blocks. `batch` defaults to 1
// so the existing single-prompt callers in the orchestrator TU keep
// their original signatures. Defined in `inference_layer.cu`.
std::vector<float> forward_step(const float *h_input, int q_seq,
                                ModelWeights &weights, KVCache &cache,
                                const float *d_cos_full,
                                const float *d_sin_full,
                                DeviceModelWeights *resident_weights,
                                int batch = 1);

// Allocate and fill the device-side cos/sin RoPE tables sized to S_MAX.
// Caller owns the returned device pointers (free with cudaFree).
// Defined in `inference_loop.cu`.
void alloc_rope_tables(float **d_cos_out, float **d_sin_out);

// Pre-load all 32 layers' BF16 weights into VRAM if `resident_weights`
// is non-null. Defined in `inference_loop.cu`.
void load_resident_layers(DeviceModelWeights *resident_weights);

// Verify every prompt in a batch tokenized to the same length. Returns
// the common length on success; throws on mismatch or empty batch.
// Defined in `inference_loop.cu`.
int validate_equal_lengths(const std::vector<std::vector<int>> &batched_ids,
                           const char *context);

// Orchestrator implementations. The thin facade in `inference.cu`
// dispatches to these. Defined in `inference_loop.cu`.
int generate_next_token_impl(ModelWeights &weights,
                             DeviceModelWeights *resident_weights,
                             const std::string &prompt);

std::vector<int> generate_tokens_impl(ModelWeights &weights,
                                      DeviceModelWeights *resident_weights,
                                      const std::string &prompt,
                                      int max_new_tokens);

GenerateDebugResult generate_tokens_resident_batched_impl(
    ModelWeights &weights, DeviceModelWeights &resident_weights,
    const std::vector<std::string> &prompts, int max_new_tokens);

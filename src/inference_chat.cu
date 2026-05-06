// Llama 3 Instruct chat-template wrapping (llm_part2 §3.1).
//
// Pure host code: no CUDA APIs are called here, but the file is built
// with nvcc to mirror the rest of the inference module.

#include "inference_internal.h"
#include "tokenizer.h"

#include <string>
#include <vector>

namespace {

// Llama 3 Instruct chat-template special token IDs (taken from the
// official tokenizer added_tokens). The instruct fine-tune expects this
// wrapper, so raw prompts are normalized through it before inference.
constexpr int BEGIN_OF_TEXT   = 128000;
constexpr int START_HEADER    = 128006;
constexpr int END_HEADER      = 128007;
constexpr int NEWLINE_NEWLINE = 271;    // BPE (Byte Pair Encoding) token for "\n\n"
constexpr int USER_TOKEN      = 882;    // BPE token for "user"
constexpr int ASSISTANT_TOKEN = 78191;  // BPE token for "assistant"

} // namespace

// Wrap raw prompt text in the Llama 3 Instruct chat template:
//   <|begin_of_text|>
//   <|start_header_id|>user<|end_header_id|>\n\n {prompt} <|eot_id|>
//   <|start_header_id|>assistant<|end_header_id|>\n\n
// The trailing assistant header (with no body) primes the model to
// generate the assistant's reply.
//
// Note: llm_part1 §3.1.1 says "You are not required to insert special
// tokens in this milestone; special-token handling will be specified
// later." We add the chat template here because the end-to-end path
// runs the instruction-tuned variant, whose prompts are expected to
// follow this wrapper. M1 grading still uses bare tokenization.
std::vector<int> apply_chat_template(const BPETokenizer &tok,
                                     const std::string &prompt) {
    auto encoded = tok.encode(prompt);
    std::vector<int> ids;
    ids.reserve(encoded.size() + 10);
    ids.push_back(BEGIN_OF_TEXT);
    ids.push_back(START_HEADER);
    ids.push_back(USER_TOKEN);
    ids.push_back(END_HEADER);
    ids.push_back(NEWLINE_NEWLINE);
    ids.insert(ids.end(), encoded.begin(), encoded.end());
    ids.push_back(EOT_ID);
    ids.push_back(START_HEADER);
    ids.push_back(ASSISTANT_TOKEN);
    ids.push_back(END_HEADER);
    ids.push_back(NEWLINE_NEWLINE);
    return ids;
}

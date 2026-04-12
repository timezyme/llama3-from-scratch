// Inference pipeline for Llama 3 8B.
// Stub for Phase 0. Will be implemented in Phases 1-4.

#include "inference.h"
#include "tokenizer.h"

#include <stdexcept>

int generate_next_token(ModelWeights & /*weights*/,
                        const std::string & /*prompt*/) {
    throw std::runtime_error("generate_next_token not yet implemented");
}

std::string decode_token(int token_id) {
    static BPETokenizer tok(TOKENIZER_PATH);
    return tok.decode({token_id});
}

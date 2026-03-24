#include "test_api.h"
#include "config.h"
#include "loader.h"
#include "tokenizer.h"
#include "kernel/kernels.cuh"
#include <memory>

namespace {
BPETokenizer &shared_tokenizer() {
    static BPETokenizer tok(TOKENIZER_PATH);
    return tok;
}

LlamaDumpLoader &shared_embeddings_loader() {
    static LlamaDumpLoader loader(DumpFloatType::BF16);
    static const bool loaded = []() {
        if (!loader.load_embeddings("assets/llama3/dump/embeddings.bin",
                                    EMBEDDING_DIM)) {
            throw runtime_error("failed to load embeddings dump file");
        }
        return true;
    }();
    (void)loaded;
    return loader;
}
} // namespace

vector<int> TestAPI::tokenize(string input) {
    BPETokenizer &tok = shared_tokenizer();
    vector<int> token_ids = tok.encode(input);
    token_ids.insert(token_ids.begin(), tok.bos_id());
    return token_ids;
}

string TestAPI::detokenize(vector<int> token_ids) {
    BPETokenizer &tok = shared_tokenizer();
    return tok.decode(token_ids);
}

vector<float> TestAPI::get_embeddings(vector<int> token_ids) {
    LlamaDumpLoader &loader = shared_embeddings_loader();
    std::unique_ptr<float_t[]> raw(loader.get_embeddings(token_ids));
    const size_t total = token_ids.size() * static_cast<size_t>(EMBEDDING_DIM);
    vector<float> out(raw.get(), raw.get() + total);
    return out;
}

vector<float> TestAPI::matmul(const vector<float> &A, const vector<float> &B,
                              int M, int K, int N) {
    if (M < 0 || K < 0 || N < 0) {
        throw runtime_error("matmul dimensions must be non-negative");
    }
    const size_t mk = static_cast<size_t>(M) * static_cast<size_t>(K);
    const size_t kn = static_cast<size_t>(K) * static_cast<size_t>(N);
    const size_t mn = static_cast<size_t>(M) * static_cast<size_t>(N);

    if (A.size() != mk) {
        throw runtime_error("matmul A size mismatch");
    }
    if (B.size() != kn) {
        throw runtime_error("matmul B size mismatch");
    }

    vector<float> C(mn, 0.0f);
    gpu_matmul(A.data(), B.data(), C.data(), M, K, N);
    return C;
}

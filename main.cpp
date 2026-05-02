// Llama 3 8B inference CLI.
// Usage: ./bin/llm "prompt"                  (single token, greedy)
//        ./bin/llm --max-tokens N "prompt"   (N-token autoregressive, KV cached)

#include "config.h"
#include "tokenizer.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#ifdef CUDA_ENABLED
#include "inference.h"

namespace {

void print_usage(const char *argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--max-tokens N] \"prompt\"\n"
                 "  --max-tokens N   generate up to N tokens (default 1)\n",
                 argv0);
}

} // namespace
#endif

int main(int argc, char *argv[]) {
#ifndef CUDA_ENABLED
    (void)argc;
    (void)argv;
    std::cerr << "Error: inference requires CUDA (nvcc not found at build time)\n";
    return 1;
#else
    int max_tokens = 1;
    std::string prompt = "Hello world";

    int i = 1;
    while (i < argc) {
        if (std::strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = std::atoi(argv[i + 1]);
            if (max_tokens < 1) {
                std::fprintf(stderr, "Error: --max-tokens must be >= 1\n");
                return 1;
            }
            i += 2;
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            prompt = argv[i];
            ++i;
        }
    }

    std::printf("Prompt: \"%s\"\n", prompt.c_str());
    std::printf("Max tokens: %d\n", max_tokens);

    ModelWeights weights(DUMP_DIR);

    if (max_tokens == 1) {
        int token_id = generate_next_token(weights, prompt);
        std::string decoded = decode_token(token_id);
        std::printf("Generated token: %d\n", token_id);
        std::printf("Decoded text:    \"%s\"\n", decoded.c_str());
        std::printf("Full output:     \"%s%s\"\n", prompt.c_str(),
                    decoded.c_str());
    } else {
        auto ids = generate_tokens(weights, prompt, max_tokens);
        std::string decoded;
        for (int id : ids) decoded += decode_token(id);
        std::printf("Generated %zu tokens:\n", ids.size());
        std::printf("Decoded text:    \"%s\"\n", decoded.c_str());
        std::printf("Full output:     \"%s%s\"\n", prompt.c_str(),
                    decoded.c_str());
    }

    return 0;
#endif
}

// Llama 3 8B inference CLI.
// Usage: ./bin/llm "prompt"                  (single token, greedy)
//        ./bin/llm --max-tokens N "prompt"   (N-token autoregressive, KV cached)
//        ./bin/llm --prompt "p1" --prompt "p2" --max-tokens N

#include "config.h"
#include "tokenizer.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef CUDA_ENABLED
#include "inference.h"

namespace {

void print_usage(const char *argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--max-tokens N] \"prompt\"\n"
                 "       %s [--max-tokens N] --prompt P [--prompt P ...]\n"
                 "  --max-tokens N   generate up to N tokens (default 1)\n"
                 "  --prompt P       add one prompt to a batch (repeatable)\n",
                 argv0,
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
    std::vector<std::string> prompts;
    std::vector<std::string> positional_prompts;

    int i = 1;
    while (i < argc) {
        if (std::strcmp(argv[i], "--max-tokens") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Error: --max-tokens requires a value\n");
                return 1;
            }
            max_tokens = std::atoi(argv[i + 1]);
            if (max_tokens < 1) {
                std::fprintf(stderr, "Error: --max-tokens must be >= 1\n");
                return 1;
            }
            i += 2;
        } else if (std::strcmp(argv[i], "--prompt") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Error: --prompt requires a value\n");
                return 1;
            }
            prompts.push_back(argv[i + 1]);
            i += 2;
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            positional_prompts.push_back(argv[i]);
            ++i;
        }
    }

    if (!prompts.empty() && !positional_prompts.empty()) {
        std::fprintf(stderr,
                     "Error: use either positional prompt or --prompt, not both\n");
        return 1;
    }
    if (prompts.empty()) {
        if (positional_prompts.empty()) {
            prompts.push_back("Hello world");
        } else if (positional_prompts.size() == 1) {
            prompts.push_back(positional_prompts[0]);
        } else {
            std::fprintf(stderr, "Error: expected exactly one positional prompt\n");
            return 1;
        }
    }

    if (prompts.size() > 1) {
        // Fast UX precheck; inference tokenizes again when applying the chat template.
        BPETokenizer tok(TOKENIZER_PATH);
        const int s0 = static_cast<int>(tok.encode(prompts[0]).size());
        for (size_t b = 1; b < prompts.size(); ++b) {
            const int sb = static_cast<int>(tok.encode(prompts[b]).size());
            if (sb != s0) {
                std::fprintf(stderr,
                             "Error: --prompt args tokenize to different lengths "
                             "(b=0 -> %d tokens, b=%zu -> %d tokens). "
                             "Mixed-length batching is out of scope for this "
                             "build.\n",
                             s0, b, sb);
                return 1;
            }
        }
    }

    if (prompts.size() == 1) {
        std::printf("Prompt: \"%s\"\n", prompts[0].c_str());
    } else {
        std::printf("Prompts: %zu\n", prompts.size());
        for (size_t b = 0; b < prompts.size(); ++b) {
            std::printf("  [%zu] \"%s\"\n", b, prompts[b].c_str());
        }
    }
    std::printf("Max tokens: %d\n", max_tokens);

    ModelWeights weights(DUMP_DIR);
    if (prompts.size() == 1 && max_tokens == 1) {
        int token_id = generate_next_token(weights, prompts[0]);
        std::string decoded = decode_token(token_id);
        std::printf("Generated token: %d\n", token_id);
        std::printf("Decoded text:    \"%s\"\n", decoded.c_str());
        std::printf("Full output:     \"%s%s\"\n", prompts[0].c_str(),
                    decoded.c_str());
    } else if (prompts.size() == 1) {
        DeviceModelWeights resident_weights(DUMP_DIR);
        auto ids = generate_tokens_resident(weights, resident_weights,
                                            prompts[0], max_tokens);
        std::string decoded;
        for (int id : ids) decoded += decode_token(id);
        std::printf("Generated %zu tokens:\n", ids.size());
        std::printf("Decoded text:    \"%s\"\n", decoded.c_str());
        std::printf("Full output:     \"%s%s\"\n", prompts[0].c_str(),
                    decoded.c_str());
    } else {
        DeviceModelWeights resident_weights(DUMP_DIR);
        auto batched_ids = generate_tokens_resident(weights, resident_weights,
                                                    prompts, max_tokens);
        for (size_t b = 0; b < prompts.size(); ++b) {
            std::string decoded;
            for (int id : batched_ids[b]) decoded += decode_token(id);
            std::printf("Output [%zu]:\n", b);
            std::printf("  Generated %zu tokens\n", batched_ids[b].size());
            std::printf("  Decoded text: \"%s\"\n", decoded.c_str());
            std::printf("  Full output:  \"%s%s\"\n", prompts[b].c_str(),
                        decoded.c_str());
        }
    }

    return 0;
#endif
}

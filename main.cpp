// Llama 3 8B Instruct inference CLI.
//
// Argument-parsing and dispatch only — all the heavy lifting lives in
// inference.cu. The CLI's job is to:
//
//   1. Parse argv into (max_tokens, prompts, interactive flag).
//   2. Pick one of four execution paths based on what was asked:
//        a) interactive: REPL with resident weights (warmup once).
//        b) one prompt + max_tokens=1: cheapest path; single greedy token.
//        c) one prompt + max_tokens>1: KV-cached decode loop.
//        d) many prompts: batched decode in lockstep.
//   3. Detokenize the generated IDs and print the completion.
//
// Usage:
//   ./bin/llm "prompt"                              (single greedy token)
//   ./bin/llm --max-tokens N "prompt"               (N tokens, KV cache)
//   ./bin/llm --prompt "p1" --prompt "p2" --max-tokens N    (B>1 batch)
//   ./bin/llm --interactive --max-tokens N         (REPL on stdin, N >= 2)

#include "config.h"
#include "inference.h"
#include "tokenizer.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

// Print CLI usage text to stderr. Called for `--help` and on argument errors.
void print_usage(const char *argv0) {
    std::fprintf(stderr,
                 "Usage: %s [--max-tokens N] \"prompt\"\n"
                 "       %s [--max-tokens N] --prompt P [--prompt P ...]\n"
                 "       %s --interactive --max-tokens N\n"
                 "  --max-tokens N   generate up to N tokens (default 1)\n"
                 "  --prompt P       add one prompt to a batch (repeatable)\n"
                 "  --interactive    REPL mode (requires --max-tokens >= 2); "
                 "load resident weights once, "
                 "read prompts from stdin (Ctrl-D or 'exit' to quit)\n",
                 argv0,
                 argv0,
                 argv0);
}

} // namespace

int main(int argc, char *argv[]) {
    // Defaults: one prompt, one generated token, no batch, no REPL.
    int max_tokens = 1;
    bool interactive = false;
    std::vector<std::string> prompts;             // collected via --prompt
    std::vector<std::string> positional_prompts;  // collected as bare argv

    // Walk argv. Recognized flags consume their value (i += 2); anything
    // unrecognized is treated as the prompt text. Order doesn't matter.
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
        } else if (std::strcmp(argv[i], "--interactive") == 0) {
            interactive = true;
            ++i;
        } else if (std::strcmp(argv[i], "-h") == 0 ||
                   std::strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            positional_prompts.push_back(argv[i]);
            ++i;
        }
    }

    // REPL mode: load BF16 weights into GPU memory once, then answer prompts
    // in a loop. The warmup pays the upload cost; later prompts reuse the
    // resident weights.
    if (interactive) {
        if (!prompts.empty() || !positional_prompts.empty()) {
            std::fprintf(stderr,
                         "Error: --interactive does not take prompts on the CLI; "
                         "pipe them via stdin\n");
            return 1;
        }
        if (max_tokens < 2) {
            std::fprintf(stderr,
                         "Error: --interactive requires --max-tokens >= 2 "
                         "(single-token mode does not benefit from resident "
                         "weights; use ./bin/llm \"prompt\" instead)\n");
            return 1;
        }
        ModelWeights weights(DUMP_DIR);                  // host-side (CPU) copy
        DeviceModelWeights resident_weights(DUMP_DIR);   // device-side (GPU) copy

        std::printf("[interactive] warming up resident BF16 weights "
                    "(~165s on cold start)...\n");
        std::fflush(stdout);
        auto warmup = generate_tokens_resident(weights, resident_weights,
                                                "warmup", 1);
        (void)warmup;
        std::printf("\n[interactive] ready. max-tokens per prompt: %d. "
                    "Ctrl-D or 'exit' to quit.\n",
                    max_tokens);
        std::fflush(stdout);

        // Prompt loop: read a line, generate tokens, print the completion.
        std::string line;
        while (true) {
            std::printf("> ");
            std::fflush(stdout);
            if (!std::getline(std::cin, line)) break;
            if (line.empty()) continue;
            if (line == "exit" || line == "quit") break;

            auto ids = generate_tokens_resident(weights, resident_weights,
                                                 line, max_tokens);
            std::string decoded;
            for (int id : ids) decoded += decode_token(id);
            std::printf("\n%s%s\n\n", line.c_str(), decoded.c_str());
            std::fflush(stdout);
        }

        std::printf("\n[interactive] exit\n");
        return 0;
    }

    // Resolve prompts. The two intake forms (positional vs --prompt) are
    // mutually exclusive: mixing them is almost always a typo. With nothing
    // supplied we print usage and exit 1.
    if (!prompts.empty() && !positional_prompts.empty()) {
        std::fprintf(stderr,
                     "Error: use either positional prompt or --prompt, not both\n");
        return 1;
    }
    if (prompts.empty()) {
        if (positional_prompts.empty()) {
            print_usage(argv[0]);
            return 1;
        } else if (positional_prompts.size() == 1) {
            prompts.push_back(positional_prompts[0]);
        } else {
            std::fprintf(stderr, "Error: expected exactly one positional prompt\n");
            return 1;
        }
    }

    // Batched inference (B>1) processes every prompt in a single GPU forward
    // pass, which only works if all prompts tokenize to the same length.
    // Reject up-front with a clear error rather than failing later inside the
    // chat-template path. We tokenize here for the length check only;
    // inference will tokenize again when it applies the chat template.
    if (prompts.size() > 1) {
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

    // Pick an inference path. Three cases, each with a different cost profile:
    //   1. one prompt, one token   -> stream weights from CPU once, generate, exit
    //   2. one prompt, many tokens -> keep weights resident on the GPU so each
    //                                 decode step is fast (KV cache reused)
    //   3. many prompts (batched)  -> same as (2), but every step advances all
    //                                 prompts together
    ModelWeights weights(DUMP_DIR);
    if (prompts.size() == 1 && max_tokens == 1) {
        // Path 1: cheapest case. No GPU residency, no KV cache.
        int token_id = generate_next_token(weights, prompts[0]);
        std::string decoded = decode_token(token_id);
        std::printf("Generated token: %d\n", token_id);
        std::printf("Decoded text:    \"%s\"\n", decoded.c_str());
        std::printf("Full output:     \"%s%s\"\n", prompts[0].c_str(),
                    decoded.c_str());
    } else if (prompts.size() == 1) {
        // Path 2: multi-token decode. Pay the GPU upload cost once so each
        // subsequent token only costs a single forward pass.
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
        // Path 3: batched decode. One forward pass advances all B prompts in
        // lockstep, then we print each prompt's completion separately.
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
}

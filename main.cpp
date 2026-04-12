// Llama 3 8B single-token inference CLI.
// Usage: ./bin/llm "prompt text"
//        ./bin/llm              (defaults to "Hello world")

#include "config.h"
#include "tokenizer.h"

#include <iostream>
#include <string>

#ifdef CUDA_ENABLED
#include "inference.h"
#endif

int main(int argc, char *argv[]) {
#ifndef CUDA_ENABLED
    (void)argc;
    (void)argv;
    std::cerr << "Error: inference requires CUDA (nvcc not found at build time)\n";
    return 1;
#else
    std::string prompt = "Hello world";
    if (argc > 1)
        prompt = argv[1];

    std::printf("Prompt: \"%s\"\n", prompt.c_str());

    ModelWeights weights(DUMP_DIR);
    int token_id = generate_next_token(weights, prompt);
    std::string decoded = decode_token(token_id);

    std::printf("Generated token: %d\n", token_id);
    std::printf("Decoded text:    \"%s\"\n", decoded.c_str());
    std::printf("Full output:     \"%s%s\"\n", prompt.c_str(), decoded.c_str());

    return 0;
#endif
}

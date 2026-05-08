# How This Project Meets The Assignment

This is a plain-English tour of the code. It is written for someone who has never used C++, CUDA, or GPU programming before.

The short version: this project takes a text prompt, turns it into numbers, runs those numbers through Llama 3's math, and picks the next token the model thinks should come next. The code does that without PyTorch or Hugging Face at runtime. Python is used only before runtime to download and dump the weights into files this C++ program can read.

## A Tiny Glossary

**Token**
A token is a small piece of text. It might be a word, part of a word, punctuation, or a special marker like "begin text".

**Embedding**
An embedding is a list of numbers that represents a token. The model cannot work with text directly, so every token becomes 4096 numbers.

**Weight**
A weight is a number the trained model learned. The code loads billions of these numbers from files.

**Matrix multiply**
Most model work is repeated multiplication between blocks of numbers. If the tokenizer is how text enters the model, matrix multiply is the engine that moves almost everything else.

**GPU**
A GPU is a chip built to do many small math jobs at the same time. CUDA is NVIDIA's way to write code for that chip.

**CUDA kernel**
A kernel is a function that runs on the GPU. In this project, kernels handle matrix multiply, normalization, attention helpers, activation, and residual add.

## The Whole Flow

When you run:

```bash
./bin/llm "The capital of France is"
```

the code follows this path:

1. `main.cpp` reads the prompt and calls the inference code.
2. `src/tokenizer_bpe.cpp` turns text into token IDs.
3. `src/model_weights.cpp` loads the model weights and looks up embeddings for those token IDs.
4. `src/inference.cu` runs 32 decoder layers.
5. CUDA kernels in `kernel/` do the heavy math on the GPU.
6. The final hidden vector is scored against `lm_head`.
7. The highest-scoring token ID is decoded back into text.

That is the project in one breath. The rest of this document maps each assignment requirement to the code that handles it.

## Part 1 Requirements

Part 1 asks for the input pipeline: load model data, tokenize text, look up embeddings, and implement matrix multiply.

| Requirement | What it means | Code |
|-------------|---------------|------|
| Download model assets | Put Llama 3 files under `assets/llama3/` | `tools/llama3_downloader.py` |
| Dump weights | Convert Hugging Face weight files into simple binary files | `tools/dumper.py` |
| Load weights in C++ | Read those binary files at runtime | `include/loader.h`, `src/loader.cpp` |
| Tokenize prompts | Turn text into token IDs | `include/tokenizer.h`, `src/tokenizer_bpe.cpp` |
| Embedding lookup | Turn token IDs into 4096-number vectors | `src/model_weights.cpp`, `LlamaDumpLoader::get_embeddings` |
| Matrix multiply | Multiply number blocks on the GPU | `kernel/matmul.cu` |
| CPU fallback | Let Milestone 1 tests build without CUDA | `kernel/matmul_cpu.cpp` |

### Tokenization

`src/tokenizer_bpe.cpp` implements byte pair encoding. A useful way to think about it: the tokenizer starts with raw bytes, then repeatedly joins pairs that the vocabulary says belong together. That is how `"Hello world"` becomes a short list of integer IDs instead of thousands of separate characters.

The tokenizer also understands Llama 3's special tokens. These are not normal words. They mark things like the start of text or the start of a chat-message header.

### Weight Loading

The original model weights are stored in Hugging Face `safetensors` files. Those are good for Python tooling, but not convenient for this C++ runtime.

So the project uses a two-step setup:

1. `tools/dumper.py` writes each needed tensor into a small binary format.
2. `src/loader.cpp` reads that format and converts FP16 or BF16 values into FP32 values the kernels can use.

The loader checks shapes as it reads. That matters because a wrong shape can still be "valid memory" while producing completely wrong model output.

### Embeddings

The embedding table is like a dictionary from token ID to vector. `ModelWeights::get_embeddings` asks the loader for the rows that match the prompt's token IDs.

If the prompt has `s` tokens, the embedding result has shape:

```text
s x 4096
```

That means one 4096-number row for each token.

### Matrix Multiply

`kernel/matmul.cu` is the main GPU math kernel. It multiplies:

```text
A[M, K] * B[K, N] = C[M, N]
```

The implementation breaks the matrices into smaller tiles. Each block of GPU threads loads part of the input into shared memory, reuses it, and writes a tile of the output. It also uses double buffering, which means it can prepare the next tile while working on the current one.

That sounds fancy, but the idea is simple: don't keep walking back to slow memory for the same numbers.

## Part 2 Requirements

Part 2 asks for the full model: normalization, attention, feed-forward layers, the 32-layer loop, and next-token output.

| Requirement | What it means | Code |
|-------------|---------------|------|
| RMSNorm | Rescale each token row before major model steps | `kernel/rmsnorm.cu` |
| Q/K/V projections | Create query, key, and value tensors for attention | `src/inference.cu`, `kernel/matmul.cu` |
| RoPE | Add position information to Q and K | `kernel/rope.cu` |
| Grouped-query attention | Let each query head attend over shared K/V heads | `src/inference.cu`, `kernel/attention.cu` |
| Causal mask | Stop a token from looking at future tokens | `kernel/attention.cu` |
| Stable softmax | Turn attention scores into probabilities without overflow | `kernel/attention.cu` |
| Output projection | Mix attention heads back into the model width | `src/inference.cu` |
| Residual add | Add each block's result back into the running state | `kernel/residual.cu` |
| SwiGLU feed-forward network | Run the model's second large math block | `kernel/swiglu.cu`, `src/inference.cu` |
| 32 decoder layers | Repeat the decoder block once per model layer | `src/inference.cu` |
| Final RMSNorm and lm_head | Convert the last token vector into vocabulary scores | `src/inference.cu`, `src/model_weights.cpp` |
| Argmax and decode | Pick the highest score and turn it back into text | `src/inference.cu`, `src/tokenizer_bpe.cpp` |

## The Decoder Layer In Plain English

Each decoder layer does two jobs.

First, it lets tokens read the earlier tokens in the prompt. This is attention. For example, in `"The capital of France is"`, the last token can use information from `"France"` when guessing the next token.

Second, it runs a feed-forward network. This is another learned math block that changes the representation after attention.

The code does this 32 times because Llama 3 8B has 32 decoder layers.

Inside one layer, `src/inference.cu` runs this order:

1. Normalize the current token vectors with `gpu_rmsnorm`.
2. Make Q, K, and V with `gpu_matmul_device`.
3. Apply RoPE to Q and K with `gpu_rope`.
4. Run grouped-query attention.
5. Project the attention result back to 4096 columns.
6. Add that result back into the running token vectors.
7. Normalize again.
8. Run the feed-forward network: gate projection, up projection, SwiGLU, down projection.
9. Add that result back too.

The repeated "add it back" steps are residual connections. They help the model keep earlier information while each layer adds something new.

## How Attention Works Here

Attention is the part that decides which earlier tokens matter.

The model creates three views of the same data:

- **Q** asks, "What am I looking for?"
- **K** says, "What information do I have?"
- **V** says, "What content should I pass along?"

Llama 3 uses grouped-query attention. There are 32 query heads but only 8 key/value heads. Four query heads share one key/value head. This saves memory and work.

The code currently reshapes each head on the host side, then launches GPU matmuls and helper kernels for each head. It is correct and tested. It is not the fastest possible design because some data moves between CPU and GPU during attention. That is tracked as TODO item 8.

## How The Output Token Is Chosen

After all 32 layers, the project only needs the last token's vector. It applies final RMSNorm, then compares that vector with every row of `lm_head`.

`lm_head` produces one score per vocabulary entry. The current required path uses greedy decoding:

```text
pick the token with the largest score
```

That is called argmax. It is simple and deterministic.

## Optional Bonus Work In This Branch

The required Milestone 1 through Milestone 3 path is separate from the bonus work. The current branch also includes early bonus work:

| Bonus item | Current state |
|------------|---------------|
| TODO #1: KV cache + resident BF16 weights (+5%) | **Shipped.** Resident BF16 path keeps all 32 layers on the GPU; KV cache reuses prefix work across decode steps. Perf gate closed (`docs/JOURNAL.md` 2026-05-02 — `weights.load_all_resident_bf16` once, no per-step streaming timers). |
| TODO #2: B>1 batched generation (+5%) | **Shipped.** `--prompt p1 --prompt p2 --max-tokens N` runs the full pipeline batched. Validated by `batched_b2_distinct_parity` (B=1 vs B=2 within 1e-3 hidden-state diff). |
| Multi-token loop | `main.cpp` accepts `--max-tokens N` (autoregressive) and `--interactive` (REPL with persistent resident weights). |
| Telemetry | `include/instrument.h` prints timing and VRAM checkpoints. |
| BF16-weight matmul primitive | `gpu_matmul_device_bf16_weight` reads BF16 weights, expands per-tile, accumulates in FP32. Used by the resident-weights path. WMMA / tensor-core matmul (TODO #3) still open. |

## How We Know It Is Correct

The project uses three layers of checks.

**Milestone 1 tests**
`bin/tests` has 7 grading tests. They check tokenization, embeddings, and matrix multiply.

**Milestone 2-3 tests**
`bin/tests_m2m3` has CUDA tests for individual operators and full forward passes. These tests cover RMSNorm, Q/K/V, RoPE, attention, SwiGLU, residuals, one decoder block, and the full 32-layer run.

**Reference comparison**
`tools/verify_reference.py` compares the project logic against `reference.py`, the PyTorch version used as the grading reference.

The local Mac can run the CPU-side Milestone 1 checks. CUDA checks need a GPU machine because `nvcc` is not installed locally.

## Where To Start Reading

If you are new to the code, read in this order:

1. `config.h` - model sizes and file paths.
2. `main.cpp` - how the command line enters the program.
3. `src/tokenizer_bpe.cpp` - text to token IDs.
4. `include/loader.h` and `src/loader.cpp` - binary weight loading.
5. `src/model_weights.cpp` - embeddings and per-layer weights.
6. `src/inference.cu` - the full model flow.
7. `kernel/matmul.cu` - the main GPU math kernel.
8. The smaller kernels: `rmsnorm.cu`, `rope.cu`, `attention.cu`, `swiglu.cu`, and `residual.cu`.

You do not need to understand every CUDA line on the first pass. Follow the data: text becomes token IDs, token IDs become vectors, vectors pass through 32 layers, and the last vector becomes the next token.

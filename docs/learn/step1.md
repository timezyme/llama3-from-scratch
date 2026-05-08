---

## Step 1: CLI entry and 4-path dispatch

**File:** `main.cpp:50-241`
**Where in the pipeline:** the first thing that runs.

### High-level picture

`main.cpp` does one job: figure out what the user asked for, then hand off to the right inference function. It doesn't do any math and doesn't touch the GPU. It just dispatches.

You type something like `./bin/llm "Hello"` on the command line, and `main()` parses those arguments and picks one of four paths:

| Path | When | What happens | Key line |
| --- | --- | --- | --- |
| 1. Single token | 1 prompt, `max_tokens=1` (default) | Streams FP32 weights from CPU one layer at a time, generates one token, exits | `main.cpp:203` |
| 2. Multi-token | 1 prompt, `max_tokens > 1` | Uploads all weights to GPU as BF16 once, then decodes N tokens with a KV cache | `main.cpp:211` |
| 3. Batched | Multiple `--prompt` args | Same as path 2, but processes all prompts in lockstep | `main.cpp:223` |
| 4. Interactive REPL | `--interactive` flag | Uploads weights once, then loops reading prompts from stdin | `main.cpp:95` |

The split between path 1 and path 2 comes from a hardware constraint: Llama 3 8B has ~16 billion parameters. In FP32 (4 bytes each) that's ~32 GB, more than the L4's 24 GB VRAM. So path 1 streams one layer at a time from CPU memory. In BF16 (2 bytes each) it's ~16 GB, which fits, so paths 2-4 upload everything once and reuse it.

Two C++ objects handle weight management:
- `ModelWeights` (line 109, 202) -- holds weights on the CPU
- `DeviceModelWeights` (line 110, 214) -- holds the GPU-resident BF16 copy

### New concepts

FP32 vs BF16 -- FP32 is a standard 32-bit float. BF16 ("brain floating point 16") is a 16-bit format that keeps FP32's exponent range but cuts the mantissa (precision) in half. Half the bytes means the model fits in GPU memory, at the cost of some numerical precision.

Mantissa (also called significand) -- the precision digits of a float. A float is `(-1)^sign x 2^exponent x 1.mantissa`: the exponent sets the range, the mantissa sets how many significant digits. FP32 has 23 mantissa bits (\~7 decimal digits); BF16 keeps FP32's 8-bit exponent but cuts the mantissa to 7 bits (\~2 digits). That's why BF16-to-FP32 conversion is just a 16-bit left-shift (`milifloat.h:18`) -- BF16 is the top half of an FP32 word.

Resident weights -- "resident" means the weights stay in GPU memory across multiple forward passes instead of being re-uploaded each time.

### TA-scrutiny items at this location

None directly. `main.cpp` is pure dispatch. But a TA might use it as a jumping-off point to ask *why* the paths differ, which leads to the FP32/BF16 memory constraint above.

---

TA-style question for you:

Path 1 (single token) creates only `ModelWeights`, the CPU-side weights. Paths 2-4 create both `ModelWeights` and `DeviceModelWeights`. Why does the multi-token path need both? Why can't it just use the GPU copy?

**Answer**

The embedding table lives on the CPU inside `ModelWeights`. When you give the model a prompt, the first thing it does is convert each token ID into a 4,096-float vector by looking up a row in that table. That's a table lookup, not a matrix multiply, so it happens on the CPU.

`DeviceModelWeights` holds the 32 layers' worth of projection weights, FFN weights, and norm gammas on the GPU in BF16 -- the stuff that feeds the CUDA kernels.

The multi-token path needs both because they serve different stages: `ModelWeights` does the embedding lookup on the CPU, then the resulting `s x 4096` matrix gets sent to the GPU, where `DeviceModelWeights` takes over for the heavy compute through all 32 layers.

Path 1 only needs `ModelWeights` because it streams everything, including layer weights, from CPU memory one layer at a time. It never builds a persistent GPU copy.
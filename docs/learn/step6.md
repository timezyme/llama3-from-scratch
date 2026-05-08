---

## Step 6: Embedding Lookup

**File:** `src/model_weights.cpp:59-61` (entry), `src/loader.cpp:348-381` (implementation)
**Where in the pipeline:** The tokenizer produced a list of integer token IDs. This step converts each ID into a 4,096-float vector. The output is the first real numerical data that enters the GPU.

### High-level picture

The embedding table is a giant lookup table: 128,256 rows (one per possible token) by 4,096 columns (the embedding dimension). Each row is a learned vector that represents what that token "means" to the model. The lookup is dead simple — given token ID `9906`, go to row 9906 and copy out the 4,096 floats.

For a prompt with `s` tokens, the output is a matrix of shape `[s, 4096]`. This is the `X` that feeds into the first decoder block.

### How it works in code

`ModelWeights::get_embeddings()` at line 59 is a one-liner that delegates to the loader. The real work is in `LlamaDumpLoader::get_embeddings()` at line 348:

1. The entire embedding file was read into memory during `load_global()` (step 4); rows are decoded on demand here
2. For each token ID, compute the byte offset: `token_id * row_bytes` (line 373)
3. Decode that row's 4,096 values from BF16 to FP32 (line 374-377)
4. Write them into the output matrix at `row * 4096`

No matrix math, no GPU — just indexed reads from a cached byte array, decoded element-by-element on the CPU.

### This is NOT a matmul

A common misconception: embedding lookup looks like a matrix multiply (one-hot vector times the embedding matrix), and mathematically it is. But nobody implements it that way — a one-hot times a matrix is just selecting one row, so you skip the multiply and do a direct memory read. That's exactly what line 373 does.

### TA-scrutiny items

- **Embedding table lives on CPU, not GPU.** The Part 1 assignment bonus questions ask "where is the best place for the embedding table, CPU or GPU?" This implementation keeps it on CPU. The tradeoff: CPU lookup avoids using precious GPU VRAM for a table that's only read once per prompt (not per layer), but it means the resulting `[s, 4096]` FP32 matrix must be transferred to the GPU over PCIe before computation starts. For small `s` (typical prompts), the transfer is negligible.

---

**TA-style question:**

The embedding table is 128,256 x 4,096. A typical prompt after chat-template wrapping might be ~15 tokens. What fraction of the table's rows are actually used in a single forward pass, and what does this imply about whether caching the whole table on GPU is a good use of VRAM?

**answer**

15 rows out of 128,256 — about 0.01%. You're using a tiny sliver of the table per forward pass. Storing the whole thing on GPU would cost ~1 GB in BF16 (or ~2 GB in FP32) of VRAM that's almost entirely cold data. Meanwhile, that VRAM is needed for the 32 layers of weights (~14.5 GB in BF16), activations, and the KV cache.

Keeping it on CPU and transferring just the 15 x 4,096 x 4 = ~240 KB of FP32 result over PCIe is a much better tradeoff. PCIe can move 240 KB in microseconds — invisible next to the GPU compute that follows.

---

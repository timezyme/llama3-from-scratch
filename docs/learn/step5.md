## Step 5: ModelWeights & Transpose-at-Load

**File:** `src/model_weights.cpp:41-190`, `src/model_weights.cpp:232-239`, `include/model_weights.h`

**Where this fits:** after the binary loader, before the GPU math.

### The simple idea

Before the GPU can run the model, the code has to get the model's numbers ready.

Those numbers are called **weights**.

Think of `ModelWeights` as the model's toolbox manager:

1. It knows where the weight files are.
2. It loads the right weights for each layer.
3. It flips some weight matrices into the shape our matmul expects.
4. It frees layer weights when the streaming path is done with them.

### What gets loaded

The model has **32 decoder layers**.

Each layer has its own tools:

- norm weights
- Q/K/V/O attention weights
- gate/up/down FFN weights

The code loads one layer's tools in `ModelWeights::load_layer`.

There are also **global weights** used outside the 32-layer loop:

- the embedding table
- the final norm weight
- `lm_head`, which turns the final hidden vector into vocabulary scores

Important repo-specific detail: this checkpoint has `tie_word_embeddings=false`, so `lm_head` is **not** the embedding table.

### The big idea: transpose once

**Transpose** means: flip rows and columns.

Example:

```
[2 rows, 3 cols] -> [3 rows, 2 cols]
```

The HuggingFace checkpoint stores projection weights as:

```
[out, in]
```

But this project's matmul code wants to multiply like this:

```
X @ weight
```

So the weight needs to be stored as:

```
[in, out]
```

Instead of flipping the weight every time we use it, the code flips it **once when loading**.

That is what `transpose()` does in `src/model_weights.cpp:232`.

After that, the forward pass can just use the weight directly.

### Tiny shape example

For K projection, the dump file gives:

```
[1024, 4096]
```

That means:

```
out = 1024
in  = 4096
```

But the incoming token vectors have width 4096, so matmul wants:

```
[4096, 1024]
```

So `ModelWeights` loads K, transposes it, and stores it ready to use.

Q is different:

```
Q:   4096 wide = 32 query heads * 128
K/V: 1024 wide = 8 KV heads * 128
```

Q's shape still looks like `[4096, 4096]` after transpose because it is square, but the values are still flipped across the diagonal.

K and V are smaller because Llama 3 uses **GQA**: 32 Q heads share 8 K/V heads. That means 4 Q heads share each K/V head.

### What is not transposed

Only 2D projection matrices are transposed.

The 1D norm weights do not need it. They are just lists of 4096 scale numbers.

The `lm_head` is also kept in its loaded layout because the CPU logits code uses it directly as `[vocab, hidden]`.

### TA answer

If a TA asks why the weight shapes look flipped, say:

> HuggingFace stores projection weights as `[out, in]`, but our matmul path wants `[in, out]`. We transpose each 2D projection weight once at load time, so inference does not pay that cost repeatedly.

---

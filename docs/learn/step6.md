## Step 6: Embedding Lookup

**File:** `src/model_weights.cpp:59-61`, `src/loader.cpp:348-381`

The tokenizer gave us a list of token IDs.

Now we need to turn each token ID into numbers the model can use.

That is what **embedding lookup** does.

Think of the embedding table as a giant dictionary:

```text
token ID -> 4096-number vector
```

So if the token ID is `9906`, the code goes to row `9906` in the embedding table and copies out that row.

### The simple idea

Each token becomes one vector.

```text
token IDs:   [128000, 9906, 1917]
embeddings:  3 rows x 4096 numbers
```

If the prompt has `s` tokens, the output shape is:

```text
[s, 4096]
```

This output is called `X`.

`X` is the first real matrix that enters the decoder layers.

### How the code does it

`ModelWeights::get_embeddings()` is just the entry point.

The real work happens in `LlamaDumpLoader::get_embeddings()`:

1. Take one token ID.
2. Use it as a row number.
3. Copy that row from the embedding table.
4. Decode the row into FP32 numbers.
5. Put it into the output matrix.

No attention yet.

No decoder layer yet.

No matrix multiply yet.

Just row lookup.

### This is not a matmul

A **one-hot vector** is a vector that is all zeros except for a single `1` at one position — the position tells you "which item." For the embedding case, the vector has length 128,256 (the vocab size), with the `1` at the position of the token ID.

Tiny example: pretend the vocab is just 5 words `[cat, dog, fish, bird, ant]`. Token ID 2 = "fish". The one-hot for "fish" is `[0, 0, 1, 0, 0]`.

Mathematically, embedding lookup can be described as a one-hot vector times the embedding table.

But the code does not do that.

A one-hot vector would mostly be zeros, so multiplying would waste work.

Instead, the code directly picks the row it needs.

That is the whole trick.

### TA answer

If a TA asks what embedding lookup does, say:

> It turns token IDs into vectors. Each token ID selects one row from the embedding table, so a sequence of `s` token IDs becomes an `[s, 4096]` matrix. That matrix is the input to the first decoder layer.


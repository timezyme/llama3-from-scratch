## Step 17: KV Cache (Bonus Feature)

The KV cache is needed because old tokens do not need new K and V vectors every
time we generate another token. The cache stores those old K and V rows once, so
decode can process only the new token and still attend to the whole prefix.

### Main idea

During prefill, the model processes the whole prompt. Each layer writes K and V
rows for every prompt token.

During decode, each step has only one new token. The model computes K and V for
that new token, appends them to the cache, and attention reads the cached prefix.

This is exact, not a shortcut. In a causal decoder, later tokens cannot change
the K and V rows already computed for earlier tokens.

The cache does not make attention ignore the past. The new Q still compares
against all cached K rows, then mixes the matching V rows. The win is that old
tokens are not run through all the projection and FFN work again.

The cache allocates K and V buffers for every layer up front. The constructor at
`src/kv_cache.cu:39` creates fixed device buffers, so generation does not resize
or move them mid-loop.

### Where it fits

The K and V projection matmuls write straight into the cache. In `forward_step`,
the K write is visible at `src/inference_layer.cu:334`, and the V write follows
right after it. Attention later reads the layer's cached K and V slices at
`src/inference_layer.cu:378`.

RoPE is applied only to the newly written K rows. Older cached K rows were
already rotated when they were first written, so rotating them again would give
the wrong positions.

After the step finishes, `cache.advance(q_seq)` at `src/inference_layer.cu:442`
records that the new rows are now part of the prefix.

### Review question

Why is KV caching correct, and what work does it save?

**answer**

It is correct because earlier tokens' K and V rows do not change after they are
computed. Causal attention lets later tokens read earlier tokens, but later
tokens do not rewrite earlier token states.

It saves the work of recomputing K and V, and the rest of the decoder path, for
old prefix tokens on every decode step. The new token still attends over the
cached prefix, so the past is still visible.

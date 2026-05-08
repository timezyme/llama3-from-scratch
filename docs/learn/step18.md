## Step 18: B>1 Batched Generation (Bonus Feature)

Batching is useful because one-token decode can be too small to keep the GPU
busy. Instead of running separate forward passes for several prompts, this path
stacks the prompts into one taller batch and advances them together.

### Main idea

At B=1 decode, the model usually processes one new token:

```text
rows = 1 * q_seq
```

That gives the matmul kernels very few rows of work. With batching, `forward_step`
sets `rows = batch * q_seq` at `src/inference_layer.cu:187`, so the same kernels
process more rows in one call.

The batched path starts in `generate_tokens_resident_batched_impl` at
`src/inference_loop.cu:116`. It does one prefill for all prompts, then one decode
loop where each batch slot produces one token per step.

This does not change the model math for each prompt. Each prompt still has its
own generated token IDs and its own K/V cache region. The goal is to share the
same forward-pass schedule and keep the GPU busier.

### Where it fits

This is an optional bonus path. The base project can assume one prompt at a
time.

The simple constraint is that all prompts must tokenize to the same length.
The code checks that at `src/inference_loop.cu:141`. Equal lengths keep the
batched tensors rectangular and let every slot share the same `q_seq`.

Mixed lengths would need padding plus attention masks so short prompts do not
attend to fake pad tokens. This implementation does not do that.

Finished slots also stay in the batch until the whole batch is done. If one slot
emits `EOT_ID` early, the decode loop feeds `EOT_ID` for that slot at
`src/inference_loop.cu:200`. That keeps the batch shape and cache layout fixed.

### Review question

Why does this batching path require equal tokenized prompt lengths, and why do
finished slots keep feeding `EOT_ID`?

**answer**

Equal lengths let all prompts share the same rectangular batch shape and the
same `q_seq`. Without that, the code would need padding and pad-aware attention
masks.

Finished slots keep feeding `EOT_ID` so the batch size does not change while
generation is running. That avoids reshaping tensors or moving cache regions
mid-loop.

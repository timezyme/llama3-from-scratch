## Step 14: 32-Layer Loop

The 32-layer loop is needed because the model was trained as 32 decoder blocks
in a fixed order. Inference has to run all 32 blocks in that same order, using
the learned weights for each layer.

### Main idea

One decoder block is:

```text
RMSNorm -> Q/K/V -> RoPE -> attention -> O projection -> residual
        -> RMSNorm -> FFN -> residual
```

Llama 3 8B has 32 decoder blocks. The constant is `NUM_LAYERS` at
`config.h:29`.

The loop starts at `src/inference_layer.cu:261`. Each pass takes the current
`X`, runs one full decoder block, and writes the updated result back into `X`.
So layer 0 produces the input to layer 1, layer 1 produces the input to layer 2,
and so on.

The code sequence stays the same each time. The weights change.

Each layer has its own 9 learned tensors:

```text
2 RMSNorm weights
4 attention projection weights
3 FFN projection weights
```

Using layer 0's weights for every pass would still run, but it would be the
wrong model.

### Where it fits

Before the loop, `X` holds the token embeddings for this pass.

Inside the loop, the code chooses the current layer's weights. The resident path
gets GPU-resident BF16 weights at `src/inference_layer.cu:270`. The streaming
path loads that layer's CPU weights at `src/inference_layer.cu:273`, copies them
to the GPU, then unloads the CPU copy after the layer finishes.

The scratch buffers do not get reallocated for every layer. Buffers like `d_X`,
`d_Xnorm`, `d_Q`, `d_attn`, `d_gate`, and `d_ffn` are allocated before the loop
and reused.

After all 32 layers finish, the KV cache length advances, and the final RMSNorm
runs next.

### Review question

In the 32-layer loop, what changes each iteration, and what stays the same?

**answer**

The weights change. Each layer has its own RMSNorm, attention, and FFN weights,
and the loop must use the correct layer's tensors.

The code path and scratch buffers stay the same. `X` is the running hidden state:
the output of one layer becomes the input to the next layer.

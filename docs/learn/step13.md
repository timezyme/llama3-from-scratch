## Step 13: SwiGLU FFN and Residual Add

The FFN is needed because attention only mixes information between tokens. After
that, each token still needs its own compute step to rebuild its 4,096-number
state, then the residual add writes that update back into `X`.

### Main idea

The feed-forward network, or FFN, works on each token row by itself. Attention
is the "communicate" step that lets tokens see each other. The FFN is the
"compute" step where each token rebuilds its own representation from what it
saw.

The FFN has an expand, gate, and shrink shape:

```text
X_norm  -> gate      [rows, 14336]
X_norm  -> up        [rows, 14336]
gate/up -> SwiGLU    [rows, 14336]
SwiGLU  -> ffn_out   [rows, 4096]
```

The wide middle, 14,336 numbers per row, exists for capacity. Each token gets
temporarily projected into a much larger space so the model has room to compute
richer feature combinations than would fit in 4,096, then the down projection
picks the most useful ones and writes them back at the original width.

The second RMSNorm happens first at `src/inference_layer.cu:409`. This uses
`post_attention_layernorm.weight`, not the `input_layernorm.weight` from the
attention half of the layer.

Then the gate and up matmuls start at `src/inference_layer.cu:412`. They make
two wide 14,336-number rows from the same `X_norm` input.

SwiGLU combines those two rows element by element:

```text
SiLU(gate) * up
```

SiLU means `g / (1 + exp(-g))`.

The kernel loads `gate[i]` at `kernel/swiglu.cu:83`, computes SiLU on it, then
multiplies by `up[i]`. This is a gate because the transformed `gate` value
scales `up` at that element, so the model can learn which features pass through
and which ones get reduced.

### Where it fits

This is the second half of one decoder layer.

After SwiGLU, the down projection shrinks the row back to 4,096 numbers. Then
the second residual add updates `X` at `src/inference_layer.cu:431`:

```text
X = X + ffn_out
```

After this add, the decoder layer is done. The next layer receives the updated
`X`.

### Review question

What are the FFN steps, and which RMSNorm weight must be used before them?

**answer**

The FFN first applies the second RMSNorm, then runs gate and up projections,
combines them with `SiLU(gate) * up`, projects back down to 4,096 numbers, and
adds the result back into `X`.

The RMSNorm weight must be `post_attention_layernorm.weight`. Each decoder layer
has two different RMSNorm weights: `input_layernorm.weight` before attention and
`post_attention_layernorm.weight` before the FFN. Reusing the first one here is
a silent bug because the shapes still match, but the learned scaling is wrong.

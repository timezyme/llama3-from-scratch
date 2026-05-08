## Step 12: O Projection and Residual Add

The O projection is needed because the 32 attention heads produce a packed
result that still has to be learned-mixed back into the model's normal `X`
space. The residual add keeps the original `X` and adds attention as an update.

### Main idea

Attention produces one packed tensor called `O`:

```text
O: [rows, 4096]
```

It is really 32 head outputs stitched side by side:

```text
32 heads x 128 numbers = 4096
```

The output projection mixes that packed attention result into `attn_out`:

```text
attn_out = O x W_O_stored
```

`W_O_stored` means the output-projection weight after the Step 5 transpose.

In `forward_step`, the resident path does this matmul at
`src/inference_layer.cu:390`. The streaming path does the same projection at
`src/inference_layer.cu:394`.

The important part is what happens next:

```text
X = X + attn_out
```

That is the first residual add. Without it, the layer would replace the running
hidden state instead of updating it.

### Where it fits

This step closes the attention half of the decoder block.

The add happens in place on `d_X` at `src/inference_layer.cu:397`. That matters
because `d_X` is the layer input, not the normalized copy `d_Xnorm` used to make
Q, K, and V.

The CUDA kernel is simple. One thread handles one number, and the whole operation
is just `a[i] += b[i]` at `kernel/residual.cu:65`.

After this, the updated `X` goes into the second RMSNorm and the feed-forward
network.

### Review question

In the first residual add, what gets added to what, and why is it important that
the target is `X`, not `X_norm`?

**answer**

The code adds `attn_out` into the original layer input:

```text
X = X + attn_out
```

It should target `X` because the residual path is the skip connection. It keeps
the original signal alive while attention adds an update. `X_norm` was only a
temporary normalized copy used to compute Q, K, and V.

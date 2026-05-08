## Step 5: ModelWeights and Transpose-at-Load

Before the model can run, the code has to load the learned weights.

`ModelWeights` is the CPU-side object that knows how to load those files.

### Main idea

The main idea is:

```text
transpose once while loading, not during every forward pass
```

The checkpoint stores projection weights as:

```text
[out, in]
```

But the project matmul path wants them stored as:

```text
[in, out]
```

So `ModelWeights::load_layer` loads each decoder-layer projection and flips it
once. That happens at `src/model_weights.cpp:111`.

The actual row/column flip is the small `transpose` helper at
`src/model_weights.cpp:232`.

After that, the decoder can use the stored weights directly in matmul calls.

### Small example

For the K projection, the checkpoint file is shaped like this:

```text
[1024, 4096]
```

After transpose, the code stores it like this:

```text
[4096, 1024]
```

Now a row with 4,096 numbers can multiply into a K vector with 1,024 numbers.

```text
[rows, 4096] x [4096, 1024] = [rows, 1024]
```

### Where it fits

Global weights are loaded separately at `src/model_weights.cpp:41`. That includes
embeddings, final RMSNorm weights, and the separate `lm_head`.

Inside `forward_step`, each decoder layer chooses its weight source at
`src/inference_layer.cu:269`. Streaming uses `ModelWeights`; resident GPU
inference uses `DeviceModelWeights`.

Both paths keep the same idea: weights are already in the layout matmul expects.

### Review question

The checkpoint stores K projection as `[1024, 4096]`, but matmul needs
`[4096, 1024]`.

Why transpose at load time instead of inside every matmul?

**answer**

Because the shape only needs to be fixed once.

After loading, every forward pass can use:

```text
[rows, 4096] x [4096, 1024]
```

If we waited until matmul time, we would repeat the same layout fix again and
again. Transposing once makes the runtime path simpler and avoids shape bugs.

### Review answer

If someone asks what this step does, say:

> Step 5 loads model weights into the layout our matmul code expects. The
> checkpoint stores projection weights as `[out, in]`, but runtime matmul uses
> `[in, out]`. So the loader transposes each 2D layer weight once when it loads
> the layer. After that, the decoder can reuse the stored weight directly.

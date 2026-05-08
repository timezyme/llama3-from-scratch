## Step 4: Dump Format & Loader

The model weights live in HuggingFace `.safetensors` files. Plain C++ cannot read that format.

So we build a tiny bridge:

1. A Python script writes each tensor as one binary file. That happens in `write_tensor_file()` at `tools/dumper.py:159-186`.
2. The C++ loader reads those binary files at startup. That happens in `load_dense_tensor_checked()` at `src/loader.cpp:217-234`.

This bridge runs **once, offline**. It is not part of inference.

### The main idea

The important idea is:

```text
a tiny binary format that both sides agree on
```

Each binary file looks like this:

```text
[ 280-byte header ][ raw number bytes ]
```

The header carries four things:

- the tensor name (ASCII, padded to 256 bytes)
- the number type (FP32, FP16, or BF16)
- how many dimensions
- the size of each dimension

Python writes that header layout in `tools/dumper.py:32-33`. C++ defines the matching constants in `src/loader.cpp:29-43` and reads the bytes back in `parse_header()` at `src/loader.cpp:142-163`.

Because both sides agree on those 280 bytes, we do not need to link any ML library at runtime.

### Small example

The K projection weight for layer 0 looks like this on disk:

```text
header:   name  = "model.layers.0.self_attn.k_proj.weight"
          dtype = BF16
          shape = [1024, 4096]
payload:  1024 * 4096 * 2 bytes of raw BF16
```

The C++ loader parses the header, checks the shape matches what the model expects, and decodes the payload to FP32.

BF16 is short for bfloat16, a 16-bit float that shares FP32's 8-bit exponent. That makes `bf16_to_float()` in `include/milifloat.h:18-23` just a 16-bit left shift — the BF16 bits become the top half of a 32-bit float.

### Where this fits

This step is offline prep. It writes the binary files once.

Step 5 is where those files actually get loaded into `ModelWeights` and reshaped for the matmul kernels.

### TA answer

If a TA asks how the weights get from HuggingFace into the C++ engine, say:

> A Python script writes each tensor as a binary file: a 280-byte header (name, dtype, shape) followed by raw payload bytes. The C++ loader parses the header, validates the shape, and decodes the payload to FP32. The format is dead simple on purpose — it is the only contract the two sides share, so we do not need to link any ML library at runtime.

---

## Step 4: Dump Format & Loader

**Files:** `tools/dumper.py`, `src/loader.cpp`, `include/milifloat.h`
**Where in the pipeline:** This is the bridge between the Python model files (HuggingFace `.safetensors`) and the C++ inference engine. It runs once offline — not during inference.

### High-level picture

The model weights ship from HuggingFace in `.safetensors` format, which is a Python ecosystem format. Your C++ code can't read that directly. So there's a two-step process:

1. **Python dumper** (`tools/dumper.py`) reads the safetensors, converts each tensor to BF16, and writes a simple binary file per tensor.
2. **C++ loader** (`src/loader.cpp`) reads those binary files at inference time.

Each binary file has a dead-simple format:

```
[  280-byte header  ][  raw payload bytes  ]

Header layout (little-endian):
  Bytes 0-255:   tensor name (ASCII, null-padded)
  Bytes 256-259: dtype code (0=FP32, 1=FP16, 2=BF16)
  Bytes 260-263: ndims (1 or 2)
  Bytes 264-271: shape[0] (uint64)
  Bytes 272-279: shape[1] (uint64, 0 if 1D)
```

The dumper organizes output into directories: `embeddings.bin` for the embedding table, `layer_00/` through `layer_31/` for per-layer weights, and `global/` for things like the final norm and lm_head.

### Two loading paths in C++

The loader has two modes depending on which inference path called it:

| Mode           | Function                                  | What it does                             | Used by                         |
| -------------- | ----------------------------------------- | ---------------------------------------- | ------------------------------- |
| **FP32 widen** | `load_dense_tensor_checked` (line 217)    | Decodes every element to FP32            | Path 1 (single-token streaming) |
| **Raw BF16**   | `load_bf16_raw_tensor_checked` (line 240) | memcpy the payload as-is — no conversion | Paths 2-4 (GPU-resident)        |

The FP32 path calls `decode_value()` (line 192) on every element, which dispatches based on dtype code to either a straight memcpy (FP32), `half_to_float()` (FP16), or `bf16_to_float()` (BF16).

### New concept: BF16 vs FP16

Both are 16-bit floats, but they carve up the bits differently:

| Format   | Exponent bits | Mantissa bits | Range        | Precision |
| -------- | ------------- | ------------- | ------------ | --------- |
| FP32     | 8             | 23            | huge         | high      |
| **BF16** | **8**         | **7**         | same as FP32 | low       |
| FP16     | 5             | 10            | small        | medium    |

BF16 keeps FP32's full exponent (same range of representable numbers) but cuts precision to 7 mantissa bits. This is why `bf16_to_float()` in `milifloat.h:18` is a single left-shift by 16 — the BF16 bits are literally the top 16 bits of an FP32 value. FP16 has a different exponent width, so `half_to_float()` (line 28) has to rebias the exponent from 5-bit (bias 15) to 8-bit (bias 127), handle subnormals, infinities, and NaNs — much more code.

### Embedding table: lazy per-row decode

The embedding table is 128,256 rows x 4,096 columns. Instead of decoding all ~525 million values to FP32 up front, `load_embeddings()` (line 300) caches the raw blob and `get_embeddings()` (line 348) decodes only the rows you need — one per token in the prompt. For a 10-token prompt, that's 10 rows decoded instead of 128,256.

### TA-scrutiny items

- **BF16/FP16/FP32 decode**: Know the difference and why BF16 conversion is a one-liner while FP16 is not.
- **Header validation**: The loader strictly validates shape, dtype, and file size (lines 114-136). This is a deliberate design choice — silent header drift produces "model runs but outputs garbage."

---

**TA-style question:**

The lazy embedding approach decodes only the rows you need. But there's a second, subtler reason for not decoding the full table up-front. The full table is 128,256 x 4,096. In FP32, how many bytes is that? And what does that number mean on a machine with limited RAM when you also have 32 layers of weights to load?

**answer**

128,256 x 4,096 x 4 bytes = \~2 GB in FP32. The raw BF16 blob is half that (\~1 GB). If you eagerly decoded the whole table to FP32, you'd hold *both* the 1 GB raw blob (while reading it) and the 2 GB FP32 output simultaneously — 3 GB just for embeddings. Meanwhile you also need to load 32 layers of weights. On a memory-constrained machine, that peak allocation matters.

By keeping the raw blob and decoding per-row, you only ever allocate `s x 4096 x 4` bytes of FP32 output (a few hundred KB for a typical prompt), and the 1 GB raw blob serves as both storage and source. You never pay the 2 GB FP32 expansion of the full table.

---
## Step 9: Q/K/V Projections

This step turns each normalized token row into Q, K, and V, the three arrays
attention uses to compare tokens and move information.

### Main idea

After RMSNorm, the decoder runs three matmuls on `X_norm`:

```text
Q = X_norm x W_Q  -> [rows, 4096]
K = X_norm x W_K  -> [rows, 1024]
V = X_norm x W_V  -> [rows, 1024]
```

Step 5 already transposed the checkpoint weights, so here `W_Q`, `W_K`, and
`W_V` mean the stored `[in, out]` matrices.

A query, or Q, is what a token is looking for. A key, or K, is what another
token can match against. A value, or V, is the information attention will mix
into the output.

The key detail is the shape difference. The constants start at `config.h:17`:
32 query heads, 8 key/value heads, and 128 numbers per head. So Q is
`32 x 128 = 4096` columns, while K and V are each `8 x 128 = 1024` columns.

That is grouped-query attention, or GQA. Four query heads share one K/V head.
The model still gets 32 query heads, but it stores less K/V data.

### Where it fits

Inside `forward_step`, RMSNorm writes `d_Xnorm` first. Then the Q projection
writes the temporary `d_Q` buffer at `src/inference_layer.cu:325`.

K and V use the same normalized input, but this repo writes them straight into
the KV cache. The K matmul starts at `src/inference_layer.cu:332`, and the V
matmul starts at `src/inference_layer.cu:335`. The cache part is an extension
in this repo; the core requirement is still the same three projections.

This step sits between RMSNorm and RoPE. RoPE will rotate Q and K next. V is
not rotated because values carry the information that attention later copies
and mixes.

### Review question

Why does Q have 4,096 columns, while K and V each have only 1,024 columns?

**answer**

Q has 32 heads, and each head has 128 numbers:

```text
32 x 128 = 4096
```

K and V have 8 heads, with the same 128 numbers per head:

```text
8 x 128 = 1024
```

That is GQA. The query side keeps 32 heads, but every four query heads share
one K head and one V head. So Q is wider than K/V because there are more query
heads, not because the head size is different.

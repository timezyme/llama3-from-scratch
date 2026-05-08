## Step 11: GQA Attention and Softmax

Attention is needed because each token has to choose which tokens so far to read
from. Softmax turns raw Q/K scores into weights for mixing V, and GQA does that
with fewer K/V heads so the K/V cache is smaller.

### Main idea

After RoPE, each query head asks: which tokens so far should I read from?

For one query head, the code does this:

```text
scores = Q x K^T
scores = scores / sqrt(128)
mask future tokens
weights = softmax(scores)
output = weights x V
```

The controller loop is `run_attention_heads` at `src/inference_layer.cu:87`.
It runs this process for all 32 query heads.

This is grouped-query attention, or GQA. There are 32 Q heads but only 8 K/V
heads, so four Q heads share one K/V head. That keeps all 32 query views while
cutting K/V memory by 4x. The mapping is `kvg = hi / 4` at
`src/inference_layer.cu:99`.

### Where it fits

This step sits after RoPE and before the output projection.

The causal mask comes before softmax. It writes a large negative number above
the diagonal, where `col > row`, so token `row` cannot read future token `col`.
That check is at `kernel/attention.cu:135`.

Softmax then turns each score row into weights that add up to 1. The important
part is numerical stability. The kernel finds the row max first, then computes
`exp(score - row_max)`. That subtraction happens at `kernel/attention.cu:186`.

Without that subtraction, a large score can make `exp(score)` overflow to
`+inf`. Once that happens, the row can become `NaN`, and the model output is
wrong from there.

### Review question

Why does attention apply the causal mask before softmax, and why does softmax
subtract the row max before `exp`?

**answer**

The causal mask removes future tokens before the row becomes probabilities. A
token can read itself and earlier tokens, but not later ones.

Subtracting the row max keeps the largest exponent at `exp(0) = 1`, so FP32 does
not overflow. The softmax result is the same because subtracting the same number
from every score cancels out in the ratio.

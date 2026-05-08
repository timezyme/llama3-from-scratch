## Step 10: RoPE

RoPE is needed because attention needs word order. It rotates Q and K by token
position so attention can compare both what a token says and where it appears.

### Main idea

Q, K, and V by themselves do not say whether a token came first, second, or
last. Without RoPE, the model could compare token content, but it would not get
the right position signal.

RoPE adds that order by rotating pairs of numbers inside Q and K. The rotation
angle depends on the token position. So when attention later compares Q and K,
the score includes both content and position.

For each 128-number head, RoPE makes 64 pairs. In Llama 3, the first number in
each pair comes from the first half of the head, and the second number comes
from the second half. The code forms that second-half pair at
`kernel/rope.cu:126`.

Then the kernel applies the small 2D rotation:

```text
new_first  = first * cos - second * sin
new_second = first * sin + second * cos
```

That write happens in place at `kernel/rope.cu:137`.

### Where it fits

RoPE runs after Q/K/V projection and before attention. In `forward_step`, Q is
rotated at `src/inference_layer.cu:358`. K is rotated right after that, using
the newly written K rows in the KV cache. V is not rotated because V carries the
content that attention will mix after the Q/K scores are computed.

The sine and cosine values are prepared before the forward pass. The table uses
Llama 3's base value, `500000`, from `config.h:26`.

### Review question

What are the two Llama 3 RoPE details that are easy to get wrong?

**answer**

First, Llama 3 pairs the first half of each head with the second half:

```text
(0, 64), (1, 65), ...
```

It does not pair neighboring numbers like this:

```text
(0, 1), (2, 3), ...
```

Second, Llama 3 uses RoPE base `500000`, not `10000`.

Both mistakes are hard to catch because the kernel still runs and the numbers
can look reasonable. But the model is using the wrong positions, so the final
tokens can be wrong.

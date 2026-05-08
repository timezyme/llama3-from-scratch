## Step 8: RMSNorm Kernel

This kernel rescales one token's numbers so they are not too big or too small, then applies learned weights.

For one row `x`:

```text
rms = sqrt(mean(x_i^2) + epsilon)
y_i = (x_i / rms) * gamma_i
```

`gamma` is the learned scale for each column.

Here, `epsilon` is a tiny safety value. This project sets it to `1e-5`
(`config.h:14`) to avoid dividing by zero.

### Main idea

One CUDA block handles one 4,096-float row.

The kernel starts at `kernel/rmsnorm.cu:96` in `rmsnorm_kernel`.

The kernel does four things:

1. Threads add up squares from the row.
2. Shared memory reduces the partial sums into one total.
3. The code computes the row RMS at `kernel/rmsnorm.cu:138`, with `epsilon` inside
   the square root.
4. Threads write `input[i] / rms * gamma[i]` at `kernel/rmsnorm.cu:144`.

The `gamma` multiply is required. It is a learned weight, not an optional
cleanup step.

### Where it fits

Inside `forward_step`, RMSNorm first runs before Q/K/V at
`src/inference_layer.cu:315`. The same function calls it again before the
feed-forward network, or FFN, and once at the end before logits.

### Review question

What does RMSNorm do for one row, and what two details must be right?

**answer**

It computes the row's root mean square, divides each value by it, then applies
`gamma`.

The kernel uses one block per row. Threads sum squares, reduce the total in
shared memory, compute the RMS, and write the scaled output.

`epsilon` must stay inside the square root so the divisor stays nonzero. `gamma`
must be applied because it is the learned scale the model expects.

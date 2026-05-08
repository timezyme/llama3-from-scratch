## Step 7: Matmul Kernel

A matmul multiplies two matrices:

```text
C = A x B
```

In this project, matmul is the GPU workhorse. It is used for Q, K, V, O, gate,
up, down, and the attention matmuls.

The kernel does not know what Q or K means. It only knows matrix shapes.

### Main idea

The main idea is **reuse**.

Reading from GPU global memory, or HBM, is expensive. So the kernel does not
keep rereading the same matrix values from HBM.

Instead, each CUDA block copies a small tile of `A` and `B` into shared memory.
Then the threads reuse that tile while they build a tile of `C`.

The FP32 kernel starts at `kernel/matmul.cu:157` in `matmul_kernel`. Its main
reuse loop starts at `kernel/matmul.cu:231`: load the next tile, compute with
the current tile, sync, then swap.

Each thread keeps its running sums in registers. That is the fastest storage
the thread has.

### Small example

For Q projection:

```text
X_norm: [rows, 4096]
W_Q:    [4096, 4096]
Q:      [rows, 4096]
```

The kernel sees this as:

```text
A: [M, K]
B: [K, N]
C: [M, N]
```

So `M = rows`, `K = 4096`, and `N = 4096`.

Inside `forward_step`, the resident path calls this matmul code for Q/K/V at
`src/inference_layer.cu:323`. The same layer loop later uses matmul for O,
gate, up, and down.

### BF16 weights

The BF16-weight version starts at `kernel/matmul.cu:358` in
`matmul_bf16_weight_kernel`.

It stores weights as BF16 to save memory, widens them to FP32 inside the
kernel, and still accumulates in FP32.

### The exception

The final `lm_head` does not use this GPU matmul kernel in this repo.

Only the last hidden vector is needed for logits, so the code uses the CPU loop
at `src/inference_layer.cu:145` in `compute_lm_head_logits`.

### Review question

For one `128 x 128` output tile, one value from `A` helps compute 128 output
values. One value from `B` also helps compute 128 output values.

In a simple non-tiled kernel, how many HBM reads could that become? What does
tiling save?

**answer**

Without tiling, the same `A` value could be read 128 times from HBM. The same
is true for a `B` value.

With tiling, each value is read once from HBM into shared memory, then reused
from shared memory.

So the math is the same, but HBM traffic is much lower. That is why tiled
matmul is faster.

### Review answer

If someone asks what this step does, say:

> This is the GPU matrix multiply used throughout the decoder. The key idea is
> tiling: load small pieces of `A` and `B` from HBM into shared memory, then
> reuse them while each thread accumulates results in registers. The BF16 path
> saves memory by storing weights as BF16, but still accumulates in FP32.

---

## Step 7: Matmul Kernel (TA-SCRUTINY: M1 Mandatory)

**File:** `kernel/matmul.cu:157-341` (FP32 kernel), lines 358-504 (BF16-weight variant)
**Where in the pipeline:** This is the workhorse of the entire model. Every projection in the forward pass (Q, K, V, O, gate, up, down) is a matmul — that's 7 per decoder layer x 32 layers = **224 matmul calls per forward pass**. Getting this kernel right is the M1 midway check-in centerpiece.

### High-level picture

Matrix multiply: C = A x B, where A is your activations (e.g., [s, 4096]) and B is a weight matrix (e.g., [4096, 4096]). The naive way — each thread computes one element of C by dot-producting a row of A with a column of B — is catastrophically slow because every thread reads the same data from slow global memory (called **HBM**, High-Bandwidth Memory) over and over.

The fix is **tiling with shared memory**. The GPU has a small, fast scratchpad per thread block called shared memory (~10ns access vs ~300ns for HBM). The strategy: load a chunk of A and B into shared memory once, then let all 256 threads in the block reuse those values many times for their computations.

### The three M1 mandatory optimizations

These are explicitly required by the assignment and a TA *will* ask about them:

**1. Tiling** (lines 72-74). The output matrix C is divided into 128x128 tiles. Each thread block owns one tile and walks across the K dimension in 16-wide slabs. This limits how much data is in flight at once.

**2. Shared-memory reuse** (lines 165-166). Two shared arrays `smA[BM][BK+1]` and `smB[BK][BN+1]` hold the current tile of A and B. Each element loaded from HBM into shared memory gets reused 128 times — once by every thread that needs it along its row or column. That's the key multiplier that turns a memory-bound kernel into a compute-bound one.

**3. Coalesced HBM access** (lines 194-227). When 32 threads in a warp read consecutive memory addresses, the GPU merges those into a single wide transaction. The load pattern is designed so `tid + i * NUM_THREADS` maps to consecutive global addresses, giving maximum bandwidth from HBM.

### The four phases

```
PHASE A (line 159): Allocate shared memory + zero 64 register accumulators per thread
PHASE B (line 186): Prefetch tile 0 into shared buffer 0
PHASE C (line 244): Main loop — for each K-tile:
    (1) Prefetch NEXT tile into alternate buffer
    (2) Compute on CURRENT tile: 8x8 outer products into registers
    (3) __syncthreads(), swap buffers
PHASE D (line 328): Write 64 accumulators per thread back to C
```

### Double-buffering

The kernel has **two** copies of shared memory (`smA[2]`, `smB[2]`). While threads compute on buffer `cur`, the GPU simultaneously loads the next tile into buffer `nxt`. This overlaps memory latency with computation — by the time the compute is done, the next tile is already loaded. The `__syncthreads()` + swap at the end of each iteration (line 320-321) ensures everyone is done reading before the buffer gets reused.

### Per-thread register accumulation

Each thread owns an 8x8 grid of accumulators (`acc[TM][TN]` = 64 floats) that live in **registers** — the fastest memory on the GPU. The inner loop (lines 298-317) pulls values from shared memory into `a_reg[8]` and `b_reg[8]`, then does an 8x8 outer product. That's 64 FMAs (Fused Multiply-Adds) from just 16 shared-memory reads. Across the whole K dimension: 16 depths x 64 FMAs = 1,024 FMAs per thread per tile.

### Two kernels, three entry points

| Kernel                                 | Weight dtype                | Used by              |
| -------------------------------------- | --------------------------- | -------------------- |
| `matmul_kernel` (line 157)             | FP32                        | Path 1 (streaming)   |
| `matmul_bf16_weight_kernel` (line 358) | BF16 widened to FP32 inline | Paths 2-4 (resident) |

| Entry point                           | line                                                    | What              |
| ------------------------------------- | ------------------------------------------------------- | ----------------- |
| `gpu_matmul` (525)                    | Host pointers, allocates GPU memory, copies H2D and D2H | M1 grader tests   |
| `gpu_matmul_device` (611)             | Device pointers, no copies                              | FP32 forward pass |
| `gpu_matmul_device_bf16_weight` (638) | Device pointers, BF16 weights                           | BF16 forward pass |

The BF16 kernel is structurally identical — same phases A-D, same tile sizes. The only difference: loading B uses `load_bf16_quad()` (line 122) which reads 4 BF16 values via a single 64-bit load and widens them to FP32 before they enter shared memory. Once in `smB`, the compute loop doesn't know the weights came from BF16.

### New concepts

- **HBM (High-Bandwidth Memory)**: The GPU's main DRAM. Large (24 GB on L4) but slow (~300ns latency). This is where your weight matrices and activation tensors live.
- **Shared memory**: Per-block on-chip scratchpad. Tiny (~48-100 KB) but fast (~10ns). Each thread block gets its own private copy.
- **FMA (Fused Multiply-Add)**: `a*b + c` in one clock cycle on a CUDA core. The fundamental arithmetic operation — every dot product is a chain of FMAs.
- **Bank conflicts**: Shared memory has 32 banks. If multiple threads access the same bank simultaneously, accesses serialize. The `+1` padding on `smA[BM][BK+1]` (line 165) shifts columns across banks to prevent this.

### TA-scrutiny items

This is the **highest-scrutiny file in the project**. Expect questions on:
- Why tiling helps (reuse ratio: each HBM load feeds 128 reuses)
- Why double-buffering (overlap load + compute)
- Why `+1` padding (bank conflict avoidance)
- Why `float4` loads for B (128-bit coalesced reads)
- Why accumulation is always FP32 even when weights are BF16 (precision of running sums)

---

**TA-style question:**

Each element loaded into `smA` from HBM is reused 128 times (once per column of the output tile). Each element in `smB` is reused 128 times (once per row). Without shared memory, how many times would each element of A need to be re-read from HBM to compute the same 128x128 tile of C? What does this say about the "arithmetic intensity" improvement from tiling?

**answer**

Without shared memory, each element of A would be re-read from HBM once for every output element in its row of the tile. One element of A at position `[r, k]` contributes to all 128 output columns in row `r` — so it gets read 128 times from HBM instead of once.

Same for B: each element `B[k, c]` contributes to all 128 output rows in column `c` — 128 HBM reads instead of one.

With tiling, you load each element once from HBM into shared memory, then read it 128 times from shared memory. That's a **128x reduction in HBM traffic** for the same number of FMAs. In arithmetic intensity terms: the naive kernel does ~2 FLOPs per byte loaded from HBM (one multiply, one add, for one element of C). The tiled kernel does ~256 FLOPs per byte (128 reuses x 2 FLOPs). That shifts the kernel from memory-bandwidth-bound to compute-bound — exactly what you want, because the GPU has far more FMA throughput than memory bandwidth.

---
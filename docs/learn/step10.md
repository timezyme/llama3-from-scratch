---

## Step 10: RoPE (Rotary Position Embeddings)

**File:** `kernel/rope.cu:101-137` (kernel), lines 181-192 (table precomputation)
**Where in the pipeline:** Right after Q/K/V projections, before attention. Applied in-place to Q and K — **not** to V. This is how the model knows token positions.

### High-level picture

A transformer has no built-in sense of word order. If you scrambled the token positions, the raw Q/K/V projections would be identical. RoPE fixes this by **rotating** each Q and K vector by an angle that depends on the token's position in the sequence. After rotation, the dot product `Q_i * K_j` naturally encodes the *distance* between positions i and j.

For each head vector of 128 dimensions, RoPE groups them into 64 pairs and applies a 2D rotation to each pair:

```
pair index i (0..63):
  theta_i = 1 / 500000^(2i/128)          ← frequency
  angle   = position * theta_i            ← angle depends on position

  x_new[i]      =  x[i] * cos(angle) - x[i+64] * sin(angle)
  x_new[i+64]   =  x[i] * sin(angle) + x[i+64] * cos(angle)
```

That's a standard 2x2 rotation matrix `[cos, -sin; sin, cos]` applied to the pair `(x[i], x[i+64])`.

### The two TA-scrutiny pitfalls (Part 2 section 4)

**1. Pairing is (i, i + h_d/2), NOT (2i, 2i+1)** (line 123-124).

The original RoPE paper pairs even/odd indices: `(x[0], x[1])`, `(x[2], x[3])`, etc. Llama 3 uses "rotate_half": pair the first half with the second half: `(x[0], x[64])`, `(x[1], x[65])`, etc. Using the wrong pairing produces output that looks numerically reasonable but is silently wrong.

**2. Base is 500,000, NOT 10,000** (line 186).

The original RoPE paper uses base 10,000. Llama 3 uses 500,000. The base controls how quickly the rotation frequencies decay across dimensions — a larger base means slower decay, which helps the model handle longer sequences. Wrong base = wrong frequency scale = wrong positional encoding, even if everything else is correct.

### How the kernel works

One thread per `(row, head, pair_index)` triple. No shared memory, no reductions — fully data-parallel. Each thread:
1. Unpacks its flat index into row/head/pair (lines 114-118)
2. Computes position: `pos = row % q_seq` (line 118) — this resets positions at batch boundaries
3. Looks up precomputed cos/sin from the table (lines 127-129)
4. Applies the 2D rotation in-place (lines 135-136)

### Precomputed tables

The cos/sin tables are built once on the CPU (`precompute_rope_table`, line 181) and uploaded to GPU memory. The kernel just reads from them — no trig at runtime. The table is small: `seq_len * 64 * 2` floats (cos + sin).

### Why V is not rotated

V carries the *content* of what a token says. Position information only needs to flow through the Q*K dot product (which determines *how much* attention one token pays to another). V is weighted by the attention scores but doesn't participate in the score computation itself.

---

**TA-style question:**

The precomputed table has one row per position, and each row has 64 entries (one per pair). The `theta` values decrease exponentially: `theta_0 = 1/500000^0 = 1.0`, while `theta_63 = 1/500000^(126/128)` which is extremely small. What does this mean for the high-index pairs vs. the low-index pairs — which pairs rotate fast and which rotate slowly — and why is that useful?

**answer**

Low-index pairs (i near 0) have `theta` close to 1.0, so `angle = position * theta` grows quickly. These pairs rotate fast — they change a lot between adjacent positions. They encode **fine-grained, local** position information ("is this token 1 or 2 positions away?").

High-index pairs (i near 63) have `theta` close to zero, so `angle` barely changes between positions. These pairs rotate slowly — they encode **coarse, long-range** position information ("is this token roughly in the first half or second half of the sequence?").

Why this is useful: the model gets position signals at multiple scales simultaneously. Some attention heads can use the fast-rotating dimensions to care about exact local word order (important for grammar), while other heads can use the slow-rotating dimensions to attend to tokens far away (important for understanding context). It's like having both a fine ruler and a wide map in the same representation.

The large base (500,000 vs 10,000) stretches these frequencies out, making the slow dimensions rotate even more slowly — which is specifically why Llama 3 can handle longer contexts than models using the original 10,000 base.

---

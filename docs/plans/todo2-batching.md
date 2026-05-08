# Implementation Plan: TODO #2 — Batching Support (B>1)

## Project Requirement Alignment (`docs/llm_part1.md`, `docs/llm_part2.md`)

- **Batching is optional, +5% credit.** Confirmed in `docs/llm_part1.md` §3.1.1 line 90 ("Both batching and KV cache support are considered optional extensions and may receive up to 5% additional credit") and `docs/llm_part2.md` §2.1 line 50 ("You are not required to implement batching or KV caching. Both are optional extensions"). This plan stays inside the bonus envelope and does not retrofit required functionality.
- **`tests/test.cpp` and `tests/test_api.h` are read-only** (DO NOT CHANGE THIS, `docs/llm_part1.md` §2 line 28). The plan does not modify them. M1 grading flows entirely through `TestAPI::tokenize/detokenize/get_embeddings(const vector<int>&)/matmul`; the single-vector `get_embeddings` path is preserved bit-for-bit, so the M1 7-test grading harness is unaffected.
- **Per-sequence length cap = 1000 tokens** (`docs/llm_part1.md` §3.1.1 line 90). Per-batch memory budget for `KVCache[B, 1000, 1024 floats * 32 layers * 2 (K+V)]` ≈ B × 256 MiB. For B=2 the cache costs ~512 MiB; comfortably under the L4's 22 GiB.
- **Architecture constants used in this plan match spec**: `d=4096`, `h=32`, `h_k=8`, `h_d=128`, `kv_dim=h_k*h_d=1024`, `d_ff=14336`, `V=128256` (`docs/llm_part2.md` §1 line 11).
- **Greedy decoding via argmax stays unchanged** (`docs/llm_part2.md` §3.1 line 161-165). Batched output extends the existing greedy path per batch element; no sampling/beam search.
- **C++/CUDA only, no external compute libs** (`docs/llm_part1.md` §2 line 30, line 36). All edits stay in existing kernel files; no new external dependencies.
- **Reference verification** (`docs/llm_part2.md` §3.1 line 174 + `reference.py`) was already satisfied by the existing M2-3 tests for B=1. The new parity test verifies *batching consistency* against the validated B=1 path; correctness vs `reference.py` is transitively preserved.
- **Mixed-length prompt batching is NOT in spec.** The spec assumes a single sequence (`docs/llm_part1.md` §3.1.1 line 90, "operates on a single input sequence (prompt)"). Mixed-length batching with proper pad-aware attention masking is therefore **explicitly out of scope** for this TODO; the parity test uses equal-length prompts. See "Out of Scope" below.

## Live Seam Findings (deltas from user's draft)

Verified against the live tree. The draft is structurally correct; below are the deltas the plan must address.

1. **KVCache layout**: `include/kv_cache.h:43-56` exposes `k(layer)`, `v(layer)`, `k_at(layer,row)`, `v_at(layer,row)` — there is **no `b` arg** to add; it's a brand-new dimension. Today the per-layer device buffer is a single `[max_len, kv_dim]` block (`src/kv_cache.cu:23-38`). Proposed `[B, max_len, kv_dim]` is fine but the **stride-on-B** must be passed everywhere `cache.k(layer)` is read, because the per-head host loop in `forward_step` (`src/inference.cu:303-306`) does a single `cudaMemcpy` of the layer's cached K/V (`bytes_Xkv_full = kv_seq * kv_dim * sizeof(float)`) to host. With B>1 this becomes `B * kv_seq * kv_dim * sizeof(float)` worth of payload pulled per layer. **This is the biggest cost multiplier** and must be in the risk register. Mitigation: add a `k_batch(layer, b) / v_batch(layer, b)` accessor that returns the per-batch slice base `d_K_[layer] + b * max_len * kv_dim`; otherwise the stride formula gets inlined at every call site (a recipe for the same int-swap bug class as `k_at`).
2. **`get_embeddings` is on `ModelWeights`, not `LlamaDumpLoader`** at the call site. `include/model_weights.h:79` declares `float *ModelWeights::get_embeddings(const std::vector<int> &)`, which delegates to `LlamaDumpLoader::get_embeddings` (`src/loader.cpp:323`). Both call sites in inference (`src/inference.cu:435` and `src/inference.cu:497, 514`) go through `weights.get_embeddings(...)`. The user's "overload `get_embeddings(vector<vector<int>>)`" is right but lands in **`ModelWeights` first**, with an inner pad-aware copy. The loader's existing single-vector path can stay untouched (zero risk of regressing M1).
3. **`forward_step` is the high-blast-radius file**. Seq-time-only buffer sizing today is `q_seq * d` (`src/inference.cu:190-222`). Adding B requires every `bytes_X`, `bytes_Xkv_full`, `bytes_ffn` to scale by `B`, plus the **last-row extraction at `src/inference.cu:371-373`** which currently grabs row `q_seq - 1`; with batching it must grab row `q_seq - 1` of **each** batch (B last-rows out, not 1).
4. **RoPE table is host-precomputed and indexed by position only** (`kernel/rope.cu:91-102`). Good: the table itself does not change. **The kernel's index decomposition** is `(idx / half_hd / num_heads) → pos` (`kernel/rope.cu:48-52`). For batched activations `[B, q_seq, num_heads, head_dim]` flattened to leading row dim `[B*q_seq, ...]`, `pos` must be derived as `(flat_row % q_seq) + len_before`, not `flat_row`. The host already passes a sliced `d_cos_step = d_cos_full + len_before * half_hd` (`src/inference.cu:227-228`), so to keep that working **with B**, the kernel needs a new `q_seq` arg so it can do `pos = row % q_seq` internally. Add `int q_seq` to `gpu_rope` signature, or pass `int batch, int q_seq` for clarity.
5. **`causal_mask_kernel` is per `[s,s]` only** (`kernel/attention.cu:40-48`). Today it's invoked **once per head per layer per batch element** inside `run_attention_heads` (`src/inference.cu:130-131`). Since the per-head loop already runs on a `[q_seq, kv_seq]` slice, **the cleanest seam is to keep `causal_mask_kernel` as-is and just loop B at the call site** — the per-head loop already has ~B*32-head iterations, adding the outer B is a one-line nest. **Do not modify the kernel.**
6. **CLI parsing** (`main.cpp:38-55`) is single-positional today. The user proposes `--prompt p1 --prompt p2 ...`; this fits cleanly because the existing parser treats any non-`--max-tokens` arg as the positional prompt. Proposal: switch to `std::vector<std::string> prompts;` and `--prompt P` repeated → push to vector; default to one prompt if none given. Single-prompt CLI behavior must match today byte-for-byte (no regression on the existing smoke output).
7. **Test registry pattern** (`tests/test_m2m3.cpp:2116-2168`) is a `std::map<string, TestFunc> build_registry()`. New tests register with one line each (`r["batched_b2_parity"] = test_batched_b2_parity;`) — confirmed clean drop-in.
8. **Makefile**: no new `.cu`/`.cpp` files needed if all changes stay in existing kernel files and `inference.cu`. **No SOURCES/OBJECTS edits required**.
9. **TODO.md item 2 entry** (`docs/todos/TODO.md:27-34`) lists "kernel signatures gain `int batch` and stride args" but the live `rmsnorm` and `softmax` already grid as **one block per row**. If we stack along leading row dim (`B*q_seq` rows), **rmsnorm and softmax need zero changes**. The TODO entry is overly broad; the actual minimum-correct change list is tighter.
10. **LoC**: user's ~30 for loader is **closer to ~60** because the pad-and-stack helper lives in `model_weights.cpp` and needs lengths-vector return. Total honest count: **~400-500 LoC**.

## Requirements

- B>1 forward pass: stack activations along leading row dim (`[B*s, d]`) end to end.
- Validation: B=2 of "same prompt twice" must equal two independent B=1 runs of the same prompt to within max abs diff < 1e-3 (final hidden state and argmax token).
- Existing M2-3 suite (28 tests) passes unchanged with B=1 path.
- CLI accepts repeated `--prompt P`; single-prompt behavior is byte-for-byte unchanged.
- **Not in scope**: perf gains (TODO #8), attention dispatch fusion, beam search, dynamic batching, padding-aware masking beyond zeroing pad rows.

## Phase 1 — KVCache shape change (low blast radius, validates B=1 equivalence)

**Files**:
- `include/kv_cache.h:19-67` (add `batch_`, `b` arg to accessors, recompute strides)
- `src/kv_cache.cu:23-48` (resize allocations to `B * max_len * kv_dim`)

**Signature changes (C++ default-arg-correct ordering)**:
- Constructor: `KVCache::KVCache(int max_seq_len, int batch = 1)`. **Param order is `(max_seq_len, batch)`, not `(batch, max_seq_len)`** — defaults must be trailing in C++. This preserves the existing `KVCache(S_MAX)` call sites at `src/inference.cu:437` and `src/inference.cu:488` unchanged.
- Accessors: `k_at(int layer, int row, int b = 0)` and `v_at(int layer, int row, int b = 0)`. **Param order is `(layer, row, b)`, not `(layer, b, row)`** — same trailing-default rule. Existing `k_at(layer, row)` calls keep working at B=1.
- New per-batch slice accessors: `float *k_batch(int layer, int b) const { return d_K_[layer] + (size_t)b * max_len_ * kv_dim(); }` and likewise for `v_batch`. **These are what the per-head host loop in `forward_step` calls**; without them every read site must inline the stride formula. Returns the `[max_len, kv_dim]` slice base for batch `b`. The existing `k(layer) / v(layer)` accessors keep working as `k_batch(layer, 0)` equivalents and the rename is optional.
- Add `int batch() const`.
- Internal stride: `d_K_[layer] + (b * max_len_ + row) * kv_dim()` — batch is the slowest-varying axis so B=1 access is contiguous as before.

**Cascading-signature mitigation (cross-cutting)**: `forward_step(...)` and `gpu_rope(...)` likewise gain new args, and the same trailing-default rule applies:
- `forward_step(float *h_emb, int q_seq, ModelWeights &w, KVCache &cache, float *d_cos, float *d_sin, DeviceModelWeights *resident, int batch = 1)` — `batch` is appended to the existing parameter list, so all current callers continue to compile.
- `gpu_rope(float *d_QorK, float *d_cos_step, float *d_sin_step, int seq_len, int num_heads, int head_dim, int q_seq = -1)` — `q_seq` is appended; sentinel `-1` means "use seq_len" (i.e., B=1 path). The kernel does `int actual_q_seq = (q_seq < 0) ? seq_len : q_seq;` and decodes `pos = (row % actual_q_seq)`.
- Once Phase 4 lands and all internal callers pass explicit values, the defaults can be removed in a final cleanup commit. Tests that hit these signatures externally keep their B=1 calls compiling throughout.

**Verification (independent)**:
```
make ARCH=sm_89 tests_m2m3 && ./bin/tests_m2m3 full_forward_kv_cache_one_token_parity
```
Must pass with the existing `KVCache(S_MAX)` call (default `batch=1` activates). Confirms zero regression at B=1. **Do not write `KVCache(1, S_MAX)`** — that allocates 1 token across S_MAX batches (signature is `(max_seq_len, batch)`).

**Risk**: Low. Pure layout change; no kernel touched.

## Phase 2 — Batched embedding lookup

**Files**:
- `include/model_weights.h:79` (add overload)
- `src/model_weights.cpp` (~60 LoC: pad to max length, zero pad rows, return lengths)

**Signature**: `float *get_embeddings_batched(const std::vector<std::vector<int>> &batched_ids, std::vector<int> &out_lens, int &out_smax)` — returns `[B, s_max, d]` row-major; pad rows past `lens[b]` are zeroed.

**Verification**: New unit test `embedding_batched_padding` — verifies (a) shape, (b) row b=0 first `lens[0]` rows == `get_embeddings({ids[0]})` rows, (c) pad rows are bitwise zero.

**Risk**: Low. Single-vector path unchanged (M1 grading tests untouched).

## Phase 3 — Kernel signatures: RoPE only

**File**: `kernel/rope.cu:38-85`

**Changes**:
- `rope_kernel(... int seq_len, int q_seq, int num_heads, int head_dim)` — `seq_len` becomes `B*q_seq`; new `q_seq` arg lets the kernel compute `pos = (idx / half_hd / num_heads) % q_seq`. Cos/sin still indexed by `pos` from the sliced `d_cos_step`.
- `gpu_rope(...)` host signature gains `int q_seq` (or equivalently `int batch`; pick one and update call sites).
- Update `kernel/kernels.cuh:50-51` declaration.

**Verification**: existing `rope_fixture_q` and `rope_fixture_k` must still pass with `gpu_rope(..., q_seq=s, num_heads=..., head_dim=...)` where `s` and `B*s` collapse to `seq_len=s` for B=1.

**Risk**: Medium — RoPE arithmetic is the most subtle. The B=1 fixture tests are the immediate safety net; the new B=2 parity test is the second.

**No changes** to `kernel/rmsnorm.cu`, `kernel/swiglu.cu`, `kernel/residual.cu`, `kernel/matmul*.cu`, or `kernel/attention.cu`. They all grid over rows or elements; stacking along leading row dim is transparent.

## Phase 4 — `forward_step` batching

**File**: `src/inference.cu:173-383` (the critical region)

**Changes**:
- `forward_step(...)` gains `int batch` (or accept `B` via `q_seq` totals; **prefer explicit `int batch`** for readability).
- All buffer sizings: `bytes_X = batch * q_seq * d * sizeof(float)`, etc.
- KV cache writes: per-batch write slot is `cache.k_at(layer, len_before, b)` (not `(layer, b, len_before)` — args are `(layer, row, b)` per Phase 1). The K/V projection writes want a contiguous slot per batch. Two options:
  - (a) Loop B for the projection writes (clean but B more matmul launches).
  - (b) Treat the cache as `[B, max_len, kv_dim]` and write `[B*q_seq, kv_dim]` rows in one shot, with rows interleaved by batch — **breaks the per-batch contiguous K/V layout** the per-head loop assumes.
  - **Choose (a)** — explicit per-batch projection write, B is small (2-4) and TODO #8 is the proper place for fusion.
- KV cache reads (per-head loop input): use `cache.k_batch(layer, b)` and `cache.v_batch(layer, b)` (Phase 1 accessors) as the per-batch slice base; **do not** call `cache.k(layer)` directly under B>1 (it returns batch 0's slice only).
- Per-head host loop (`src/inference.cu:97-144`): wrap in outer `for (int b = 0; b < batch; ++b)` and pass per-batch slices of h_Q, h_K, h_V; each batch independently runs the existing 32-head loop. `attn_concat` becomes `[B, q_seq, d]`, written per-batch.
- Last-token extraction (`src/inference.cu:371-373`): copy out **B last rows** — `last_hidden[b * d ... (b+1)*d]` from `d_Xnorm + (b * q_seq + q_seq - 1) * d`. **`forward_step`'s return type changes from `std::vector<float>` (size `d`) to `std::vector<float>` (size `B*d`)** — flattened `[B, d]` row-major. All callers (`generate_next_token_impl` line 447, `generate_tokens_impl` lines 498-499 and 515-516) must be updated to index `last_hidden.data() + b * d` when feeding `compute_lm_head_logits`. The B=1 path stays correct because `B*d == d` when `batch=1`.
- `compute_lm_head_logits` gets a `b` index (caller passes `last_hidden.data() + b * d`) or is called B times. Note: this function is CPU-only today (`src/inference.cu:155-166`); see Risk Register entry.

**Mixed-length prompts: explicit rejection at the API boundary** (revised after review). The earlier draft claimed "zeroed pad embeddings make K/V projections zero, so attention contributes zero V" — this is **false**:
1. Pad rows go through RMSNorm + Q/K/V projections + residual + FFN. After residual addition with non-zero hidden state from prior layers, pad-row activations are not zero past layer 0.
2. Even if pad K/V *were* zero, softmax over `[s_real_0, ..., s_real_n, 0, 0]` still distributes probability mass to the pad columns; the post-softmax weighted sum is mathematically wrong vs. an unbatched run of the shorter prompt.
3. `last_hidden` extraction would copy row `q_seq - 1` (the pad-tail row) for the shorter prompt, not row `lens[b] - 1` — wrong logits.

Mixed-length batching with proper pad-aware attention masking and per-batch last-row extraction is **out of scope** (`docs/llm_part2.md` §2.1 line 50: batching is optional, +5%). Therefore the batched API **must reject unequal tokenized lengths**:

```cpp
// In get_embeddings_batched and generate_tokens_resident batched overload:
int s = static_cast<int>(batched_ids[0].size());
for (size_t b = 1; b < batched_ids.size(); ++b) {
    if (static_cast<int>(batched_ids[b].size()) != s) {
        throw std::runtime_error(
            "batched inference requires equal tokenized prompt lengths "
            "(mixed-length batching is out of scope for TODO #2)");
    }
}
```

This makes the API contract honest: B>1 is supported only when all prompts tokenize to the same length. The parity test trivially satisfies this. CLI usage with `--prompt p1 --prompt p2` either picks equal-length prompts or hits the explicit error message.

**Verification (independent)**:
```
./bin/tests_m2m3 full_forward_kv_cache_one_token_parity   # B=1 unchanged
./bin/tests_m2m3 full_forward_resident_one_token_parity   # B=1 unchanged
./bin/tests_m2m3 full_forward_medium_prompt               # B=1 unchanged
```

**Risk**: HIGH — easy to miscount strides. The B=2 parity test (Phase 6) is the definitive net.

## Phase 5 — `main.cpp` CLI

**File**: `main.cpp:28-82`

**Changes**:
- Replace `std::string prompt` with `std::vector<std::string> prompts;`.
- New `--prompt P` flag: appends to vector (repeatable).
- Backward compat: if no `--prompt` flags and exactly one positional, use it as a single-element vector (existing behavior).
- `generate_tokens_resident` overload that takes `std::vector<std::string>` and returns `std::vector<std::vector<int>>` (one per batch element).
- **Pre-check prompt lengths via raw `tok.encode(prompt).size()` so the CLI can fail fast** before invoking the batched API. **Do not call `apply_chat_template` from `main.cpp` — it lives in `src/inference.cu`'s anonymous namespace (line 52) and has internal linkage, so the call would fail to link.** The chat template wraps each prompt with a fixed-length prefix/suffix (~10 tokens at `src/inference.cu:55-68`), so equal raw-encode lengths imply equal post-template lengths:
  ```cpp
  if (prompts.size() > 1) {
      BPETokenizer tok(TOKENIZER_PATH);
      auto e0 = tok.encode(prompts[0]);
      int s0 = static_cast<int>(e0.size());
      for (size_t b = 1; b < prompts.size(); ++b) {
          int sb = static_cast<int>(tok.encode(prompts[b]).size());
          if (sb != s0) {
              std::fprintf(stderr,
                  "Error: --prompt args tokenize to different lengths "
                  "(b=0 -> %d tokens, b=%zu -> %d tokens). "
                  "Mixed-length batching is out of scope for this build.\n",
                  s0, b, sb);
              return 1;
          }
      }
  }
  ```
- The batched API itself (`generate_tokens_resident` overload, Phase 4) also performs the equal-length check on the post-`apply_chat_template` lengths and throws `std::runtime_error` if they diverge — that's the canonical guard. The CLI pre-check is a UX nicety so the error message is structured rather than an uncaught exception.
- Print per-prompt outputs.

**Verification**: `./bin/llm "The capital of France is"` (single positional) must produce identical output to the current implementation.

**Risk**: Low. Argument parser is small and the existing positional path is preserved.

## Phase 6 — New parity tests (revised: distinct prompts, multi-token, test seam)

**Why distinct prompts**: a same-prompt-twice B=2 test cannot detect cross-batch contamination. If the per-head loop accidentally reads `h_K` for batch 0 when computing batch 1, identical inputs produce identical outputs and the bug hides. Use **two distinct prompts that happen to tokenize to the same length** so each batch slot must be treated independently.

**Test seam for hidden-state comparison**: `forward_step` lives in an anonymous namespace at `src/inference.cu` and `generate_tokens_resident` returns only token IDs. To compare hidden states with a 1e-3 gate we need a public seam. Add to `include/inference.h`:

```cpp
// Test-only: returns generated tokens plus the final-step last-row hidden
// state(s) for each batch element. `last_hidden` is captured AFTER the last
// generated token's forward pass — i.e., it is the post-final-norm hidden for
// position (prompt_len + max_new_tokens - 2) for each batch slot. Single-step
// callers (max_new_tokens=1) get the prefill's last-token hidden.
// Layout: tokens.size() == B, each tokens[b].size() == max_new_tokens.
//         last_hidden.size() == B * d, row-major [B, d].
struct GenerateDebugResult {
    std::vector<std::vector<int>> tokens; // size B
    std::vector<float> last_hidden;        // size B * d (final step only)
};

GenerateDebugResult generate_tokens_resident_debug(
    ModelWeights &weights, DeviceModelWeights &resident,
    const std::vector<std::string> &prompts, int max_new_tokens);
```

This is the minimum addition needed to write the parity tests. The non-debug path stays unchanged. Test 1 (max_new_tokens=1) consumes `last_hidden`; Test 2 (max_new_tokens=4) only asserts on `tokens` and ignores `last_hidden`, so capturing only the final step is sufficient.

**File**: `tests/test_m2m3.cpp` — append two new tests and register them in `build_registry()`:

### Test 1: `batched_b2_distinct_parity` (single-token / prefill only)
Distinct equal-length prompts, max_tokens=1.
1. Hardcode two candidate prompts and **assert at the start of the test** that their raw tokenized lengths match — fail loud (not skip) if a future tokenizer change breaks this so the bug is visible. Use `tok.encode(p).size()` rather than `apply_chat_template(tok, p)` because the latter lives in `src/inference.cu`'s anonymous namespace (line 52) and is not linkable from `tests/test_m2m3.cpp`. Equal raw lengths imply equal post-chat-template lengths because the template adds a fixed-length wrapper (~10 tokens):
   ```cpp
   const std::string pA = "What is two plus two";
   const std::string pB = "Why does the sun rise";
   BPETokenizer tok(TOKENIZER_PATH);
   auto rA = tok.encode(pA);
   auto rB = tok.encode(pB);
   if (rA.size() != rB.size()) {
       std::printf("FAIL batched_b2_distinct_parity: prompt token lengths "
                   "diverged (A=%zu, B=%zu) — pick a new pair\n",
                   rA.size(), rB.size());
       return 1;
   }
   ```
2. Run B=1 baseline for prompt A → tokens `tA[]`, hidden `hA[d]`.
3. Run B=1 baseline for prompt B → tokens `tB[]`, hidden `hB[d]`.
4. Run B=2 batched `[A, B]` → `tokens[2][]`, `last_hidden[2*d]`.
5. Assert `tokens[0] == tA && tokens[1] == tB` (proves no cross-batch token leak).
6. Assert max-abs-diff(`last_hidden[0:d]`, `hA`) < 1e-3 and max-abs-diff(`last_hidden[d:2d]`, `hB`) < 1e-3.
7. **Negative control**: also run B=2 with `[A, A]` and assert that `last_hidden[0:d] == last_hidden[d:2d]` to catch a different bug class (broken-but-symmetric indexing).

### Test 2: `batched_b2_multitoken_parity` (decode loop)
Same prompts as test 1, but `max_new_tokens=4`. This exercises the KV-cache **read** path under B>1 (decode steps re-read prior K/V from the cache; per-batch read-stride bugs only surface here).
1. B=1 baseline for A → `tA[4]`. B=1 baseline for B → `tB[4]`.
2. B=2 batched `[A, B]` → `tokens[2][4]`.
3. Assert `tokens[0] == tA && tokens[1] == tB` for all 4 steps.

**Verification**:
```
./bin/tests_m2m3 batched_b2_distinct_parity
./bin/tests_m2m3 batched_b2_multitoken_parity
```

## Phase 7 — Full regression

```
./tools/test_l4.sh                                              # quick lane: M1 + fast M2-3
./bin/tests_m2m3 batched_b2_distinct_parity
./bin/tests_m2m3 batched_b2_multitoken_parity
./bin/llm --prompt "What is two plus two" --prompt "Why does the sun rise"   # both 5 words; verify token counts match before relying
./bin/llm --prompt "Hello" --prompt "The capital of France is"  # MUST exit with the unequal-length error from Phase 5
./tools/test_l4.sh --full                                       # final gate
```

**Note**: the equal-length CLI smoke probe still requires confirming the **tokenized** lengths match (word count is not token count under BPE). Run `python tools/token_show.py "..."` once on each candidate to verify lengths match before declaring the smoke test stable. The mixed-length probe is a *negative* test: it must produce the explicit error message from Phase 5, not a forward pass.

## Risk Register

| Risk | Failure mode | Caught by |
|---|---|---|
| RoPE position decomposition wrong for B>1 | Wrong tokens generated | `rope_fixture_q/k` (B=1 unchanged) + `batched_b2_distinct_parity` |
| Per-head loop strides wrong across batches (cross-batch K/V leak) | Batch 1 reads batch 0's K/V silently | `batched_b2_distinct_parity` — distinct prompts, so a leak makes outputs match the wrong B=1 baseline |
| KV-cache **read** stride wrong on decode steps | Decode token 2+ wrong, token 1 right | `batched_b2_multitoken_parity` (max_tokens=4) |
| Symmetric-but-broken indexing (e.g., always reads b=0) | Same-input-twice would still match | `[A, A]` negative control in `batched_b2_distinct_parity` |
| KVCache `[B, max_len, kv_dim]` buffer overruns | CUDA OOM or memcpy past end | Explicit `batch_*max_len_*kv_dim` allocation + asserts in `k_at/v_at` |
| Last-token extraction copies wrong row per batch | Wrong logits → wrong token | Hidden-state 1e-3 diff in `batched_b2_distinct_parity` (requires Phase 6 test seam) |
| Mixed-length prompts silently produce wrong output | Garbage for shorter prompt | API-boundary length check throws; CLI prints explicit error (Phases 4 & 5) |
| Per-layer H2D/D2H now `B*` larger | **Slowdown, not correctness fail** — accept | TODO #8 references this |
| `gpu_rope` signature break cascades | Compile errors across `tests/test_m2m3.cpp` rope tests | Defaulted args during Phase 3, compile-fail-fast on cleanup commit |
| Silent param-swap when calling `k_at`/`KVCache` ctor | Wrong slot or 1-token cache; tests still link | New-signature examples in plan stay self-consistent (Phase 1 line 58 fix); `k_batch(layer, b)` accessor removes inline-stride duplication |
| `compute_lm_head_logits` is CPU-only and called B× per generation step | **Test runtime grows ~B×; not a correctness fail** | Phase 6 tests at B=2 / max_tokens=4 ⇒ 8 invocations of ~525M FMAs each; acceptable for parity tests, blocks scaling. TODO #7 (GPU GEMV) is the proper fix |

## Out of Scope (handed off to TODO #8 / future)

- Per-layer attention H2D/D2H elimination (TODO #8 — the slowdown will be worse with B>1, but fusion is item 8's job).
- Mixed-length batching with proper pad-aware attention masking and per-batch `lens[b]-1` last-row extraction. **The batched API throws and the CLI prints an error if prompts tokenize to different lengths** (Phases 4 & 5).
- Beam search / dynamic batching / continuous batching.
- Per-batch `KVCache` lifecycle (e.g., one batch finishing early); current design holds B in lockstep.
- **Batching the no-cache single-token path** (`generate_next_token` / `generate_next_token_resident`). Only the resident KV-cached path (`generate_tokens_resident`) gets the batched overload. The no-cache path is what M1's `bin/llm "prompt"` smoke uses; keeping it B=1 preserves byte-for-byte compatibility with the existing single-token output and the M1 grading harness.

## Estimated Complexity / LoC

| File | Edit type | LoC delta |
|---|---|---|
| `include/kv_cache.h` | edit | ~25 |
| `src/kv_cache.cu` | edit | ~30 |
| `include/model_weights.h` | edit | ~8 |
| `src/model_weights.cpp` | edit | ~70 (incl. equal-length check) |
| `kernel/rope.cu` | edit | ~12 |
| `kernel/kernels.cuh` | edit | ~3 |
| `src/inference.cu` | **edit (heavy)** | ~220 (incl. `generate_tokens_resident_debug`) |
| `include/inference.h` | edit | ~25 (debug seam + batched overload) |
| `main.cpp` | edit | ~55 (incl. CLI length check) |
| `tests/test_m2m3.cpp` | edit | ~180 (two parity tests + helpers) |
| **Total** | | **~625 LoC** |

User's estimate: 300-400. Verified count: ~625 after adding the test seam, the equal-length API/CLI checks, and the second parity test.

## Post-Draft Validation

This section captures the four checks that should run on any plan before it's declared ready. The first draft of this plan failed all four; this version is the fix.

1. **Mental compile of every new signature.**
   - `KVCache(int max_seq_len, int batch = 1)` — defaults trailing ✓; verification text uses `KVCache(S_MAX)` (not `KVCache(1, S_MAX)`) ✓
   - `k_at(int layer, int row, int b = 0)` — defaults trailing ✓; Phase 4 example uses `(layer, len_before, b)`, not the swapped `(layer, b, len_before)` ✓
   - `k_batch(int layer, int b)` and `v_batch(int layer, int b)` — non-defaulted (only used under B>1) ✓; consumed by per-head loop (Phase 4) ✓
   - `gpu_rope(..., int q_seq = -1)` — defaults trailing, sentinel for B=1 ✓
   - `forward_step(..., int batch = 1)` — defaults trailing ✓; return size widens from `d` to `B*d` (Phase 4) and is bit-stable at B=1 ✓
   - `GenerateDebugResult` and `generate_tokens_resident_debug(...)` — declared in `include/inference.h` so test code can link against it ✓

2. **Two-layer math trace of every correctness claim.**
   - Pad rows: traced through residual + RMSNorm + Q/K/V; **not zero past layer 0**. Conclusion: cannot rely on pad zeroing for correctness — reject mixed-length at the API.
   - Softmax over `[real_scores, 0, 0]`: pad columns get nonzero attention mass — reject mixed-length.
   - `last_hidden` extraction at row `q_seq - 1`: wrong row for shorter prompts in mixed-length batch — reject mixed-length.
   - Memory budget for `KVCache[B, max_len, kv_dim] * NUM_LAYERS * 2 (K+V) * sizeof(float)`: per-batch = `2 * 32 * 1024 * 1024 * 4` = 256 MiB. B=2 ⇒ 512 MiB. L4 has 22 GiB ⇒ comfortably under budget.
   - `forward_step` return-size change: `vector<float>(B*d)`. At B=1, `B*d == d` ⇒ existing callers continue to work bit-for-bit.

3. **Adversarial test design — what bug class would still pass?**
   - Same-prompt-twice would pass even if batch 1 reads batch 0's K/V. **Fixed**: distinct equal-length prompts in `batched_b2_distinct_parity`.
   - Symmetric-but-broken indexing (always reads b=0) wouldn't be caught by distinct-prompts alone if both batches happen to land on b=0's slot. **Fixed**: `[A, A]` negative control asserts both slots equal each other.
   - Single-token (prefill-only) test misses KV-cache **read** bugs in the decode loop. **Fixed**: `batched_b2_multitoken_parity` with max_tokens=4.
   - Argmax-only comparison misses numerical drift. **Fixed**: 1e-3 hidden-state diff via the test seam.
   - Calling `apply_chat_template` from `tests/` or `main.cpp` would silently fail to link (anonymous namespace at `src/inference.cu:52`). **Fixed**: Phase 5 CLI and Phase 6 Test 1 use `tok.encode(p).size()` instead; the canonical post-template length check lives inside the batched API and throws.

4. **Code-path-to-test mapping (every new path has a test).**
   - Phase 1 KVCache shape change → exercised by `full_forward_kv_cache_one_token_parity` (B=1 baseline) + new tests (B=2).
   - Phase 2 batched embedding → new `embedding_batched_padding` unit test.
   - Phase 3 RoPE q_seq decomposition → existing `rope_fixture_q/k` (B=1) + `batched_b2_distinct_parity` (B=2).
   - Phase 4 forward_step batching → both new parity tests.
   - Phase 4 KV-cache read in decode loop → `batched_b2_multitoken_parity` (specifically Test 2).
   - Phase 5 CLI length check → Phase 7 negative-test smoke run (must produce explicit error).
   - Phase 6 test seam (`generate_tokens_resident_debug`) → consumed by both new tests.

## Files referenced

- `include/kv_cache.h`
- `src/kv_cache.cu`
- `include/loader.h`
- `src/loader.cpp`
- `include/model_weights.h`
- `include/inference.h`
- `src/inference.cu`
- `kernel/kernels.cuh`
- `kernel/rope.cu`
- `kernel/rmsnorm.cu`
- `kernel/attention.cu`
- `main.cpp`
- `tests/test_m2m3.cpp`
- `Makefile`
- `docs/todos/TODO.md`

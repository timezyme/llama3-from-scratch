# Coverage Gap Tests — Plan

## Why These Tests Exist

Three supplementary tests close real fault-localization gaps that survived the
M2-3 audit. None are required by the assignment. The existing 35 M2-3 tests
already prove the system correct end-to-end; these tests turn future
regressions into a single failing test name instead of a downstream parity
miss.

| Gap | Today's failure mode | New test name |
|---|---|---|
| `W_O` shape/transpose bug | downstream miss in `decoder_block_layer0_fixture` | `output_projection_fixture` |
| FFN sub-block (gate/up/SwiGLU/down) drift | folded into `decoder_block_layer0_fixture` with the residual | `ffn_block_isolated_fixture` |
| KV-cache corruption after step 1 | `full_forward_kv_cache_one_token_parity` only checks T=1 | `kv_cache_multi_step_parity` |

## Tests Considered and Rejected

- `rope_theta_unit` — `test_rope_manual` (test_m2m3.cpp:750) already proves
  base=500000 with hand-computed expected values. A test that calls
  `precompute_rope_table(base=500000)` and compares against the same formula
  is a tautology.
- `attention_scale_unit` — would hardcode `1/sqrt(8)` on both sides. The bug
  class "wrong scale at `src/inference_layer.cu:94`" cannot be caught by a
  unit test that sets the scale itself. Already exercised by
  `attention_output_full_fixture`.
- `batched_b4_distinct_parity` — `batched_b2_distinct_parity` already covers
  the b≥2 indexing path. Marginal additional coverage.
- `layer1_hidden_state_parity` — would catch per-layer-weight indexing bugs
  more directly, but `full_forward_hello`/`full_forward_medium_prompt` already
  exercise all 32 layers and would fail on the same bug class. Not worth a new
  fixture + python-script change for a class project.

## Non-Negotiable Guardrails

- Do not modify `tests/test.cpp`, `tests/test_api.h`, or `tests/test_api.cpp`.
- Do not rename any of the 35 existing M2-3 test names or change the order
  in which they appear in `--list`. The Phase 0 split is a pure relocation
  — `tools/test_l4.sh` greps test names, and renames break the script.
- Do not add wall-clock cost to the `--quick` lane beyond ~0.5 s total.
- Do not add new fixtures. Reuse files already produced by
  `tools/gen_m2m3_fixtures.py`.

## File Structure

`tests/test_m2m3.cpp` is 2411 lines today (~3x the 800-line soft limit). The
plan splits it along the existing registry phase boundaries **before**
adding the three new tests, so the new tests land in the right group from
the start instead of being relocated later. The split is detailed in
Phase 0 below.

What does **not** change after the split: binary name (`bin/tests_m2m3`),
registry test names, `--list` output, exit codes, `tools/test_l4.sh`. The
split is invisible to the driver script.

## Phase 0 — Split `tests/test_m2m3.cpp`

The split runs first. The three new tests are added in Phases 1–2 directly
into the appropriate split files; they never touch the monolith.

### File layout

All split files use the `.cpp` extension to match the current convention
and the existing `nvcc` compile rule. The `test_m2m3_` prefix scopes the
Makefile pattern rule away from M1's `tests/test.cpp` and
`tests/test_api.cpp`.

| New file | Source-line origin in monolith | Approx LOC |
|---|---|---|
| `tests/test_m2m3_helpers.h`         | new (declarations only)                  | ~80 |
| `tests/test_m2m3_helpers.cpp`       | leaf + heavy helpers, see "Symbol moves" | ~260 |
| `tests/test_m2m3_matmul.cpp`        | lines 130–414  (Phase 0 registry)        | ~290 |
| `tests/test_m2m3_rmsnorm_proj.cpp`  | lines 415–749  (Phase 1 registry)        | ~340 |
| `tests/test_m2m3_rope_attn.cpp`     | lines 750–1198 (Phase 2 registry)        | ~450 |
| `tests/test_m2m3_decoder_full.cpp`  | lines 1199–1820 (Phase 3 + Phase 4)      | ~620 |
| `tests/test_m2m3_kv_batch.cpp`      | lines 1821–2298 (Phase 5)                | ~480 |
| `tests/test_m2m3_main.cpp`          | lines 2299–end (registry + `main`)       | ~120 |

`tests/test_m2m3.cpp` is **deleted** after the split.

The line ranges above are orientation hints — they cover the *tests* that
move to each file. Helpers that fall inside those ranges
(e.g., `load_fixture` at 415, `compute_lm_head_logits` at 436,
`run_attention_heads` at 1556, `run_forward_pass` at 1621) are extracted
to `test_m2m3_helpers.cpp` instead, per "Symbol moves" below.

### Symbol moves

`tests/test_m2m3_helpers.h` (declarations + small inline constants):

- `inline constexpr float EPSILON = 1e-2f;` (was line 54)
- `inline constexpr int PASS = 0;`          (was line 55)
- `inline constexpr int FAIL = 3;`          (was line 56)
- `inline constexpr int USAGE_ERROR = 2;`   (was line 57)
- `#define CUDA_CHECK(expr) ...`            (was line 71, macro stays)
- `using TestFunc = std::function<int()>;`  (was line 2304)
- `using Registry = std::map<std::string, TestFunc>;`  (new alias)
- declarations of `check_cuda`, `fill_deterministic`, `float_to_bf16_bits`,
  `bf16_bits_to_float_host`, `compare`, `max_abs_diff`, `load_fixture`,
  `compute_lm_head_logits`, `run_attention_heads`, `run_forward_pass`
- declarations of group registrars:
  `void register_phase0(Registry &);` … `void register_phase5(Registry &);`
- forward declaration consumed only by `test_m2m3_kv_batch.cpp` (used by
  the new `kv_cache_multi_step_parity`):
  `extern std::vector<int> apply_chat_template(const BPETokenizer &,`
  `const std::string &);`

`tests/test_m2m3_helpers.cpp` (implementations; drop `static`):

- `check_cuda` (line 63), `fill_deterministic` (78), `float_to_bf16_bits`
  (84), `bf16_bits_to_float_host` (90), `compare` (98), `max_abs_diff`
  (116), `load_fixture` (415), `compute_lm_head_logits` (436),
  `run_attention_heads` (1556), `run_forward_pass` (1621).

Local-only helpers stay `static` and travel with their group:

- `run_bf16_weight_matmul_case` (line 207) → `test_m2m3_matmul.cpp`.
- `expect_out_of_range` (2052), `assert_prompt_lengths_match` (2216),
  `same_tokens` (2230), `run_debug_generation` (2247) →
  `test_m2m3_kv_batch.cpp`.

Each per-group file ends with:

```cpp
void register_phaseN(Registry &r) {
    r["test_name_a"] = test_test_name_a;
    r["test_name_b"] = test_test_name_b;
    // ... only the names from this group's phase
}
```

`test_m2m3_main.cpp` aggregates the registrars:

```cpp
Registry build_registry() {
    Registry r;
    register_phase0(r);
    register_phase1(r);
    register_phase2(r);
    register_phase3(r);  // includes Phase 3 kernel smoke + Phase 4
    register_phase5(r);
    return r;
}
```

### Makefile changes

1. Replace `M2M3_TEST_OBJECTS` (Makefile:150–160) so it lists the eight
   split objects instead of `$(BUILD_DIR)/test_m2m3.o`:

   ```make
   M2M3_TEST_OBJECTS := $(BUILD_DIR)/test_m2m3_main.o \
                        $(BUILD_DIR)/test_m2m3_helpers.o \
                        $(BUILD_DIR)/test_m2m3_matmul.o \
                        $(BUILD_DIR)/test_m2m3_rmsnorm_proj.o \
                        $(BUILD_DIR)/test_m2m3_rope_attn.o \
                        $(BUILD_DIR)/test_m2m3_decoder_full.o \
                        $(BUILD_DIR)/test_m2m3_kv_batch.o \
                        $(BUILD_DIR)/model_weights.o \
                        ... (existing inference + kernel objects unchanged)
   ```

2. Replace the explicit rule at Makefile:167–168 with a pattern rule
   scoped by the `test_m2m3_` prefix (does not collide with `test.o` or
   `test_api.o`, which keep their `g++` rules at lines 135–139):

   ```make
   $(BUILD_DIR)/test_m2m3_%.o: tests/test_m2m3_%.cpp | $(BUILD_DIR)
       $(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
   ```

### Phase 0 verification (run before any Phase 1/2 work)

- `make ARCH=sm_89 tests_m2m3` builds clean.
- `./bin/tests_m2m3 --list` prints the same 35 names in the same order as
  before the split.
- `./tools/test_l4.sh --full` still reports `M2-3 fail=0  M2-3 ran=35`.
- `./tools/test_l4.sh --quick` still reports the same quick-lane count.

If any of those four checks fails, fix Phase 0 before touching Phases 1–2.

### Effort estimate

~2 hours for a careful phase-by-phase split. Mechanical work — the only
substantive judgement calls are which helpers go to the shared header
(everything used by ≥2 groups) vs. which stay local (everything used by
exactly one group). Both buckets are listed above.

## Test Time Budget

| Test | Cost driver | Approx wall-time | Lane |
|---|---|---|---|
| `output_projection_fixture` | one matmul, s=3, [3,4096]·[4096,4096] | <100 ms | quick |
| `ffn_block_isolated_fixture` | 3 matmuls + SwiGLU, s=3, d_ff=14336 | <300 ms | quick |
| `kv_cache_multi_step_parity` | KV-cached T=4 vs 4 full forwards (s≈13..16) | ~15 s | perf |

Quick-lane delta: **~0.4 s** across two tests. Perf-lane delta: **~15 s**.

Note on the multi-step `s` values: the KV-cached path
(`generate_tokens(weights, prompt, T)`) calls `apply_chat_template` internally
(src/inference_chat.cu:38–55), which prepends 10 wrapper tokens around the
encoded prompt. For prompt "Hello world" (~3 encoded tokens), the actual
prefill length is ~13, so the four reference forwards run at s≈13, 14, 15,
16 — not 3..6. Total wall-time is ~15 s on L4, dominated by attention's
O(s²) scaling.

## Phase 1 — Quick-lane tests

### 1.1 `output_projection_fixture`
- **File**: `tests/test_m2m3_rope_attn.cpp`, appended after
  `test_attention_output_full_fixture`. Registered in `register_phase2`
  next to the other Phase 2 entries.
- Inputs: `tests/data/m2m3/attn_output_full.bin` + layer-0 `o_proj` weight
  from the dump dir (both already produced by `gen_m2m3_fixtures.py`).
- Body: upload attn_concat to device; call `gpu_matmul` to compute
  `attn_out = attn_concat @ W_o^T`; compare against
  `tests/data/m2m3/o_proj_layer0.bin`.
- Naming follows `q_projection_fixture` / `k_projection_fixture` /
  `v_projection_fixture`.

### 1.2 `ffn_block_isolated_fixture`
- **File**: `tests/test_m2m3_decoder_full.cpp`, appended after
  `test_swiglu_manual` and before `test_decoder_block_layer0_fixture`.
  Registered in `register_phase3`.
- Inputs: `tests/data/m2m3/post_attn_rmsnorm_layer0.bin` + layer-0
  `gate_proj`, `up_proj`, `down_proj` weights (already produced).
- Body: device pipeline `gpu_matmul (gate) → gpu_matmul (up) →
  gpu_swiglu → gpu_matmul (down)`; compare against
  `tests/data/m2m3/swiglu_layer0.bin`.
- Naming gotcha to call out in a one-line comment in the test body:
  `swiglu_layer0.bin` is the *post-`down_proj` FFN output*, not the SwiGLU
  activation. `gen_m2m3_fixtures.py:244` saves `ffn_out` under that name for
  historical reasons.

## Phase 2 — Perf-lane test

### 2.1 `kv_cache_multi_step_parity`
- **File**: `tests/test_m2m3_kv_batch.cpp`, appended after
  `test_full_forward_kv_cache_one_token_parity`. Registered in
  `register_phase5`.
- Prompt: `"Hello world"`. Generate `T=4` tokens with KV caching, and
  separately run 4 full forwards over the growing sequence. Compare
  next-token IDs at every step.
- Why this and not just T=2: a position-id update bug or a write-offset bug
  that takes effect after step 1 would still pass T=2 if the bug compounds
  monotonically. T=4 is cheap enough to use as a safety margin.

#### Reference-path construction (must match KV-cached preprocessing)

The KV-cached path applies the chat template internally
(src/inference_chat.cu:38–55). The reference path **must apply the same
template**, otherwise the two paths run on different token sequences and the
comparison is invalid. There is no public API that takes a pre-tokenized
sequence and runs one forward pass, so the reference is built using the
post-split `run_forward_pass` helper (declared in
`tests/test_m2m3_helpers.h`, formerly `test_m2m3.cpp:1621`) the same way
`full_forward_medium_prompt` does (formerly `test_m2m3.cpp:2114–2124`),
but with chat template applied.

The required `extern std::vector<int> apply_chat_template(...)` declaration
already lives in `tests/test_m2m3_helpers.h` after Phase 0, so the new
test just `#include`s the header — no per-file extern.

Test body skeleton:

```cpp
const std::string prompt = "Hello world";
const int T = 4;

// --- KV-cached path ---
ModelWeights cached_weights(DUMP_DIR);
auto cached = generate_tokens(cached_weights, prompt, T);  // size T

// --- Reference path: T separate full forwards over the chat-templated prefix ---
BPETokenizer tok(TOKENIZER_PATH);
ModelWeights ref_weights(DUMP_DIR);
ref_weights.load_global();
auto tokens = apply_chat_template(tok, prompt);  // ~13 tokens for "Hello world"

std::vector<int> reference;
reference.reserve(T);
for (int step = 0; step < T; ++step) {
    std::unique_ptr<float[]> h_emb(ref_weights.get_embeddings(tokens));
    int next = run_forward_pass(h_emb.get(),
                                static_cast<int>(tokens.size()), ref_weights);
    reference.push_back(next);
    tokens.push_back(next);  // grow sequence for next iteration
}

// --- Compare ---
if (cached != reference) { /* print both, FAIL */ }
```

Notes:
- `generate_tokens` returns the newly generated tokens (not the prompt
  prefix), so `cached.size() == T` and the per-step comparison is
  `cached[i] == reference[i]`.
- `run_forward_pass` returns the argmax of the last position, which matches
  the greedy step taken by `generate_tokens`.
- Two `ModelWeights` instances are used because each path mutates internal
  state during forward; this matches the pattern in
  `test_full_forward_kv_cache_one_token_parity` (formerly
  `test_m2m3.cpp:2147–2151`, post-split lives in `test_m2m3_kv_batch.cpp`).

## Phase 3 — Wiring

1. Per-group registrar updates (no central `build_registry` to touch — that
   moved to `test_m2m3_main.cpp` in Phase 0 and just calls each group's
   `register_phaseN`):

   - In `test_m2m3_rope_attn.cpp`'s `register_phase2`, append:
     `r["output_projection_fixture"] = test_output_projection_fixture;`
   - In `test_m2m3_decoder_full.cpp`'s `register_phase3`, append (next to
     `r["decoder_block_layer0_fixture"] = ...`):
     `r["ffn_block_isolated_fixture"] = test_ffn_block_isolated_fixture;`
   - In `test_m2m3_kv_batch.cpp`'s `register_phase5`, append (after the
     existing `r["full_forward_kv_cache_one_token_parity"] = ...` line):
     `r["kv_cache_multi_step_parity"] = test_kv_cache_multi_step_parity;`

   No new comment block; the new tests slot into their group's existing
   phase boundary.

2. `tools/test_l4.sh:134`: extend the perf lane's `BUILD_TARGETS` from
   `'all'` to `'all tests_m2m3'`. The current perf lane builds only the main
   binary, so today it cannot run any `tests_m2m3` test.
3. `tools/test_l4.sh:161`: extend the quick-lane filter to exclude the new
   perf test:
   `grep -Ev '^  (full_forward_|layer_streaming_smoke$|kv_cache_multi_step_parity$)'`
4. `tools/test_l4.sh` perf lane (lines 227–229 today): wrap a `run_kv_parity`
   helper around the new test, propagate its exit code, and emit a summary
   line so a regression cannot silently exit 0. The current branch is:

   ```sh
   perf)
       run_kv_perf
       ;;
   ```

   Replace with:

   ```sh
   perf)
       run_kv_perf
       run_kv_parity
       echo
       echo \"=== SUMMARY: lane=perf  kv_perf=PASS  kv_parity=\$KV_PARITY_STATUS ===\"
       [ \$KV_PARITY_FAIL -eq 0 ]
       ;;
   ```

   Add `run_kv_parity` next to `run_kv_perf` (around test_l4.sh:176):

   ```sh
   run_kv_parity() {
       echo
       echo '=== PERF: KV cache multi-step parity (T=4) ==='
       set +e
       OUT=\$(./bin/tests_m2m3 kv_cache_multi_step_parity 2>&1)
       STATUS=\$?
       set -e
       printf '%s\n' \"\$OUT\"
       if [ \$STATUS -ne 0 ] || ! printf '%s' \"\$OUT\" | grep -q '^PASS'; then
           KV_PARITY_FAIL=1
           KV_PARITY_STATUS=FAIL
       else
           KV_PARITY_FAIL=0
           KV_PARITY_STATUS=PASS
       fi
   }
   ```

   Initialise `KV_PARITY_FAIL=0` and `KV_PARITY_STATUS=SKIP` next to the
   existing `M1_FAIL=0` / `M2_FAIL=0` / `M2_COUNT=0` block (test_l4.sh:209–211)
   so the summary line still renders if `run_kv_parity` is never reached
   (e.g., if a future lane variant skips it).

## Verification Shape

Phase 0 has its own gate (see "Phase 0 verification" above). The checks
below run after Phases 1–3 are complete.

1. Local CPU mandatory harness untouched: `make clean && make tests &&
   ./bin/tests 1` still passes.
2. L4 build: `make ARCH=sm_89 all tests tests_m2m3`. The built binary now
   links eight split objects instead of `test_m2m3.o`; verify
   `ls bin/tests_m2m3` exists and `./bin/tests_m2m3 --list | wc -l` returns
   `38` (35 existing + 3 new).
3. L4 quick lane: `./tools/test_l4.sh --quick` — pass count = old + 2.
   Wall-time delta < 1 s.
4. L4 perf lane: `./tools/test_l4.sh --perf` — runs `run_kv_perf` (existing
   audit), then `run_kv_parity` (new), then prints a `SUMMARY` line with both
   statuses. Exit code is non-zero iff the parity test failed. Now also
   requires `tests_m2m3` to be built (see Phase 3 step 2).
5. L4 full lane: `./tools/test_l4.sh --full` — all 35 + 3 = 38 M2-3 tests
   pass.

## Risks and Open Questions

- **Phase 0 split is the largest source of risk.** It touches 35 existing
  tests by relocation (no behaviour change), the Makefile, and the registry
  wiring. The Phase 0 gate above (build clean, `--list` matches, `--full`
  reports 35/35) catches any linkage or registration mistake before Phases
  1–2 layer new tests on top.
- **Helpers ODR**: moving `compare`, `load_fixture`, etc. from `static` to
  external linkage means they must be defined exactly once (in
  `test_m2m3_helpers.cpp`). If two split files accidentally redefine a
  helper, the linker will catch it; flagged here so reviewers expect a
  multiple-definition error rather than a silent wrong-symbol pick.
- **Pattern rule scope**: the new `tests/test_m2m3_%.cpp` rule must not
  match `tests/test.cpp` or `tests/test_api.cpp` (M1 harness, compiled by
  `g++` not `nvcc`). The `test_m2m3_` prefix is what enforces this — do
  not loosen it to `tests/test_%.cpp`.
- **Perf-lane build target change** is the only test-l4 infrastructure
  change. It adds ~30 s to the perf lane's build phase (compiling the eight
  split objects in parallel is roughly the same wall-time as the old
  monolith) plus ~15 s for the parity test itself. The perf lane already
  runs as an audit, not the inner loop, so this is acceptable.
- **Fixture availability**: all three new tests reuse fixtures already
  produced by `tools/gen_m2m3_fixtures.py`. No new fixture file or
  python-script change is needed.

# `src/inference.cu` Split Plan — Carve 904-line File Into Sub-800-line Modules

## Goal

Split `src/inference.cu` (904 lines, over the project's 800 max-per-file rule) into smaller, topically-cohesive translation units without changing the public `include/inference.h` API, the forward-pass numerics, the telemetry timer hierarchy, or the linker outputs of `bin/llm` and `bin/tests_m2m3`. This is a pure file-split refactor; behavior is bit-identical.

After this plan: `src/inference.cu` becomes a thin facade that owns only the public `generate_*` entry points; the chat template, kernel-launch helpers, the per-step decoder block, and the three orchestration paths (single-token, single-prompt KV-decode, batched KV-decode) live in dedicated `.cu` files; and a project-internal header `src/inference_internal.h` carries the shared private signatures. No file exceeds ~500 lines after the split (well under the project's 800-line max).

**Intentional behavior changes:** none — bit-identical refactor.

| Path | Before | After |
|---|---|---|
| every public symbol in `inference.h` | defined in `src/inference.cu` | unchanged signatures, redistributed across new TUs (translation units) |
| forward-pass numerics | as-is | as-is (same kernel order, same `cudaMalloc`/`cudaFree` ordering, same telemetry hierarchy) |
| linker outputs | `bin/llm`, `bin/tests_m2m3` | unchanged binaries (same public API surface, same observable behavior; V8 records the expected +9 cross-TU helper symbols that gain external linkage) |

**Required-test invariant:** the professor-facing `bin/tests` harness is unaffected (it does not link `inference.o`); the M2-3 internal harness `bin/tests_m2m3` links the three new objects alongside `build/inference.o` (the facade is kept; the public `generate_*` symbols stay defined there) and remains green on the quick lane (30/30, with the full registry of 35 named tests unchanged); the L4 quick lane (`./tools/test_l4.sh`) stays green; an 8-token CLI smoke produces the same Sacramento answer.

---

## Must-Read Docs (in this order)

1. **`src/inference.cu`** (current, 904 lines) — identify the seams. Five natural blocks: (i) chat template + special-token IDs (`apply_chat_template`, lines 47-94); (ii) kernel-launch helpers (`AttentionScratch`, `run_attention_heads`, `compute_lm_head_logits`, lines 96-210); (iii) per-step decoder block (`forward_step`, lines 212-522); (iv) RoPE/resident-weights/validation utilities (`alloc_rope_tables`, `load_resident_layers`, `validate_equal_lengths`, lines 524-589); (v) the three orchestrator implementations (`generate_tokens_resident_batched_impl`, `generate_next_token_impl`, `generate_tokens_impl`, lines 591-843); plus the public-entry facade (`generate_*`, `decode_token`, lines 845-904).
2. **`include/inference.h`** (68 lines) — the API surface that must not change. Public symbols: `generate_next_token`, `generate_tokens`, `generate_next_token_resident`, `generate_tokens_resident` (single-prompt + batched overloads), `generate_tokens_resident_debug`, `decode_token`, plus the `GenerateDebugResult` struct.
3. **`Makefile`** — the inference rule (line 88) and both object lists: `MAIN_CUDA_OBJECTS` (line 50) and `M2M3_TEST_OBJECTS` (line 141). The new `.cu` files must be added to both lists.
4. **`main.cpp`** and **`tests/test_m2m3.cpp`** — confirm they only call the public API. (Do not include the internal header.)
5. **`docs/plans/main-cpp-cleanup.md`** — format/voice template; this plan matches its structure.
6. **`CLAUDE.md`** (project) — file-size guidance ("200-400 typical, 800 max") and the "When adding new source files, update SOURCES, OBJECTS, and TEST_OBJECTS" reminder.

**Files this plan touches:** `src/inference.cu` (rewrite as thin facade); `src/inference_chat.cu` (new); `src/inference_layer.cu` (new); `src/inference_loop.cu` (new); `src/inference_internal.h` (new, project-internal); `Makefile` (add four `.cu` rules and update both object lists); `docs/CODEMAPS/architecture.md` (one-line module map update); `CLAUDE.md` (gitignored project-context doc — lines 86 and 98 reference `src/inference.cu` specifically and need updating after the split) — nothing else.

---

## Discussion Highlights

- **Why split this way.** Five candidate seams exist (chat, kernel-launch helpers, per-step decoder, orchestrators, facade). Putting `forward_step` and the per-head attention helper together in one TU is non-negotiable: `forward_step` calls `run_attention_heads` 32×B×32 times in the hot path and they share the `AttentionScratch` struct. Putting the three orchestrators together in one TU is right because they share the `EOT_ID`/`S_MAX` constants, the RoPE-table allocation pattern, the `compute_lm_head_logits` argmax recipe, and the `cudaFree(d_cos)`/`cudaFree(d_sin)` epilogue. Chat-template wrapping is a natural standalone — it is pure host code, references no CUDA APIs, and is called once at the top of every orchestrator. The result is four `.cu` files of roughly 70 / 460 / 290 / 100 lines (chat / layer / loop / facade), all well under the 800-line cap (`inference_layer.cu` is the largest because moving `forward_step` verbatim along with the attention helpers and `compute_lm_head_logits` totals ~434 lines of code plus includes/macro overhead).
- **Public API stays in `src/inference.cu`.** The thin facade keeps every public entry function (`generate_next_token`, `generate_tokens`, the four `_resident` variants, `generate_tokens_resident_debug`, `decode_token`) defined in `inference.cu`. Each is a one-liner that delegates to the `*_impl` function in `inference_loop.cu`. This preserves the symbol-to-TU mapping that downstream callers (and any debugger backtraces) expect, and it keeps `inference.cu` recognizable as "the file that implements `inference.h`".
- **Internal header is project-internal, not in `include/`.** `src/inference_internal.h` declares the cross-file private signatures: `apply_chat_template`, `compute_lm_head_logits`, `alloc_rope_tables`, `load_resident_layers`, `validate_equal_lengths`, `forward_step`, the three `*_impl` orchestrator entrypoints, and the `EOT_ID`/`S_MAX` constants. Putting it under `src/` (not `include/`) signals "do not include from outside this module". Tests and `main.cpp` continue to include only `include/inference.h`.
- **Constants placement.** `S_MAX` is used by the orchestrators (cap check and RoPE table size). `EOT_ID` is used by the orchestrators (stop condition + finished-slot fill). `BEGIN_OF_TEXT`/`START_HEADER`/`END_HEADER`/`NEWLINE_NEWLINE`/`USER_TOKEN`/`ASSISTANT_TOKEN` are only used by `apply_chat_template`. Decision: chat-template-only constants stay file-local in `inference_chat.cu` (anonymous namespace); `S_MAX` and `EOT_ID` move into `inference_internal.h` as `inline constexpr int` (C++17 inline variables — no ODR (One Definition Rule) violation when the header is included by multiple TUs).
- **`AttentionScratch` ownership.** The struct is consumed by `run_attention_heads` (in `inference_layer.cu`) and constructed/destructed inside `forward_step` (also `inference_layer.cu`). It does not need to be in the internal header; it lives in `inference_layer.cu`'s anonymous namespace. Only `run_attention_heads`'s signature (which takes `const AttentionScratch&`) needs care: since it is only called from `forward_step` (same TU), it can be entirely file-local. The internal header only needs to declare `forward_step`.
- **`decode_token` collision.** There are two `decode_token` callers in the spec: the public `decode_token(int)` (used by `main.cpp` for printing), and a hypothetical helper. Looking at the file, only the public one exists — the static `BPETokenizer` lives inside the public function (inference.cu:901-903). Decision: `decode_token` stays in the facade `src/inference.cu` since it owns the public-API definitions. No collision.
- **Telemetry hierarchy preservation.** The `Stopwatch` timer names (`step.prefill`, `step.decode`, `layer.total`, `layer.attn_pre`, `layer.attn_heads`, `layer.post_attn_and_ffn`, `layer.load_disk_to_host`, `layer.h2d_weights`, `layer.unload`, `lm_head.cpu`, `generate.total`, `weights.load_all_resident_bf16`) all live inside specific functions and are timed by RAII (Resource Acquisition Is Initialization) Stopwatches with no cross-TU interaction. The split does not change which function owns which timer, so `Stopwatch::print_summary()` output is byte-identical.
- **No circular includes.** `inference_chat.cu` includes `tokenizer.h` + `inference_internal.h`; `inference_layer.cu` includes `kernel/kernels.cuh` + `kv_cache.h` + `device_weights.h` + `inference_internal.h`; `inference_loop.cu` includes all of the above + `inference_internal.h`; `inference.cu` (facade) includes only `inference.h` + `inference_internal.h`. The internal header includes only the `<vector>`/`<string>` STL headers and forward-declares `ModelWeights`/`DeviceModelWeights`/`KVCache`/`BPETokenizer`. No cycles.
- **Symbol-count sanity.** `nm build/inference.o | grep -c " T "` on the current binary reports the public-entry count plus any non-static helpers. After the split, `nm build/inference.o build/inference_chat.o build/inference_layer.o build/inference_loop.o | grep -c " T "` should match (or differ only by the cross-TU helpers that lost their `static` linkage — these are intended). V8 below records the before/after counts.
- **Extension choice.** All four files are `.cu`. `inference_chat.cu` does not call any CUDA API at the moment, but keeping it `.cu` (a) lets us mirror the existing inference rule template at Makefile:88 instead of authoring a g++ rule, (b) avoids surprise if a future change wants `cudaMalloc` for token-ID upload, (c) matches CLAUDE.md guidance "Do not introduce `.cpp` files for CUDA code" (the project convention treats inference-side code as CUDA-side regardless of whether the current snippet calls runtime APIs). Build cost is identical to the current single-`nvcc` rule.
- **Linker order.** `MAIN_CUDA_OBJECTS` and `M2M3_TEST_OBJECTS` add objects in unspecified order from the linker's perspective; what matters is that every needed symbol is present once. The four new objects are appended (not interleaved among the kernel objects) to keep the diff readable. Order does not affect correctness here because there are no static-init order dependencies between TUs (the only `static` is the `BPETokenizer` inside the public `decode_token`, which uses function-local-static initialization).
- **Named namespace (optional, NOT used in this plan's baseline).** The cross-TU helpers (`apply_chat_template`, `compute_lm_head_logits`, `forward_step`, `alloc_rope_tables`, `load_resident_layers`, `validate_equal_lengths`, the three `*_impl` orchestrators) gain external linkage when moved out of the file-local anonymous namespace. To avoid global-namespace collisions and make the project-internal nature explicit at every call site, a future revision could wrap them in `namespace cs265 { namespace inference_internal { ... } }` (or a single-segment `namespace inference_internal`). Cost: a few extra lines of `namespace { ... }` wrapping per file plus qualified call sites in the facade. Benefit: zero collision risk if a future TU defines an unrelated `forward_step`, plus self-documenting "private" markers in callers. **The A1-A5 snippets below intentionally show plain global declarations to keep the baseline diff minimal; the names are unique enough today that collision risk is theoretical.** Add namespacing as a follow-up if the project grows additional helpers with similar names.

---

## Action Items

Mark each item with `[x]` when complete. Verification commands listed inline.

### Phase 1: Internal header

- [ ] **A1. Create `src/inference_internal.h` with shared private declarations.**
  - Forward-declare: `class BPETokenizer;`, `class ModelWeights;`, `class DeviceModelWeights;`, `class KVCache;`.
  - Include `<string>`, `<vector>`, and `"inference.h"` (for the `GenerateDebugResult` return type used by the batched orchestrator).
  - Declare `inline constexpr int S_MAX = 1024;` and `inline constexpr int EOT_ID = 128009;` (C++17 inline variables — single definition across all TUs that include this header).
  - Declare cross-TU helper signatures:
    ```
    std::vector<int> apply_chat_template(const BPETokenizer &tok,
                                         const std::string &prompt);

    std::vector<float> compute_lm_head_logits(const float *lm_head,
                                              const float *h_x_last);

    std::vector<float> forward_step(const float *h_input, int q_seq,
                                    ModelWeights &weights, KVCache &cache,
                                    const float *d_cos_full,
                                    const float *d_sin_full,
                                    DeviceModelWeights *resident_weights,
                                    int batch = 1);  // default preserved — current
                                                     // single-prompt callers in
                                                     // src/inference.cu:754,
                                                     // :816, :833 omit this arg

    void alloc_rope_tables(float **d_cos_out, float **d_sin_out);
    void load_resident_layers(DeviceModelWeights *resident_weights);
    int validate_equal_lengths(const std::vector<std::vector<int>> &batched_ids,
                               const char *context);

    int generate_next_token_impl(ModelWeights &weights,
                                 DeviceModelWeights *resident_weights,
                                 const std::string &prompt);

    std::vector<int> generate_tokens_impl(ModelWeights &weights,
                                          DeviceModelWeights *resident_weights,
                                          const std::string &prompt,
                                          int max_new_tokens);

    GenerateDebugResult generate_tokens_resident_batched_impl(
        ModelWeights &weights, DeviceModelWeights &resident_weights,
        const std::vector<std::string> &prompts, int max_new_tokens);
    ```
  - **Do NOT declare** `run_attention_heads`, `AttentionScratch` — those are file-local to `inference_layer.cu`. Do NOT declare `decode_token` — that is the public API in `include/inference.h`.
  - Top-of-file comment must include: "Project-internal — do not include from outside `src/inference*.cu`. In particular, do not include from `include/inference.h` (would create a cycle)."
  - Verify: `wc -l src/inference_internal.h` returns < 60 lines; `grep -c "^#include" src/inference_internal.h` returns 3 (`<string>`, `<vector>`, `"inference.h"`).

### Phase 2: Carve out `inference_chat.cu`

- [ ] **A2. Create `src/inference_chat.cu` with chat-template logic.**
  - Move from `src/inference.cu` (lines 47-94): the `BEGIN_OF_TEXT` / `START_HEADER` / `END_HEADER` / `NEWLINE_NEWLINE` / `USER_TOKEN` / `ASSISTANT_TOKEN` constants (keep file-local, anonymous namespace) and the `apply_chat_template` function. Drop `EOT_ID` here (it lives in the internal header).
  - Includes: `"inference_internal.h"`, `"tokenizer.h"`, `<string>`, `<vector>`.
  - The `apply_chat_template` function loses its file-local `namespace { }` wrapping (it must have external linkage so other TUs can call it). Keep the special-token-ID constants in an anonymous namespace.
  - Verify: `wc -l src/inference_chat.cu` returns ~60-80 lines; `grep -n "apply_chat_template" src/inference_chat.cu` shows the definition; `grep -n "EOT_ID" src/inference_chat.cu` returns 0 (it is now in the internal header).

### Phase 3: Carve out `inference_layer.cu`

- [ ] **A3. Create `src/inference_layer.cu` with the per-step decoder block.**
  - Move from `src/inference.cu`:
    - The `AttentionScratch` struct (lines 96-107) — keep in anonymous namespace (file-local).
    - `run_attention_heads` (lines 109-185) — keep in anonymous namespace (only `forward_step` calls it, same TU).
    - `compute_lm_head_logits` (lines 187-210) — give it external linkage (called from `inference_loop.cu`).
    - `forward_step` (lines 212-522) — give it external linkage.
    - The `CUDA_CHECK` macro (lines 39-45) — copy into this TU's top (file-local macro, identical body). The orchestrator TU also needs it (chat TU does not — it makes no CUDA calls); each TU that needs it defines its own copy, since the macro is `#undef`-able and tiny. (Alternative: hoist into `inference_internal.h` — see A6.)
  - Includes: `"config.h"`, `"device_weights.h"`, `"inference_internal.h"`, `"instrument.h"`, `"kernel/kernels.cuh"`, `"kv_cache.h"`, `<algorithm>`, `<cmath>`, `<cstdio>`, `<stdexcept>`, `<vector>`, `<cuda_runtime.h>`.
  - Verify: `wc -l src/inference_layer.cu` returns ~440-480 lines (the moved code spans `src/inference.cu` lines 106-539, ~434 lines, plus ~20-30 lines of new includes / macro / namespace wrapping); `nm build/inference_layer.o | grep "forward_step"` shows the symbol with `T` (text) marker; `nm build/inference_layer.o | grep "run_attention_heads"` shows nothing or a `t` (lower-case t = local) — confirming file-local linkage.

### Phase 4: Carve out `inference_loop.cu`

- [ ] **A4. Create `src/inference_loop.cu` with the three orchestrator implementations.**
  - Move from `src/inference.cu`:
    - `alloc_rope_tables` (lines 535-549) — external linkage.
    - `load_resident_layers` (lines 555-569) — external linkage.
    - `validate_equal_lengths` (lines 575-589) — external linkage.
    - `generate_tokens_resident_batched_impl` (lines 597-715) — external linkage (called from facade).
    - `generate_next_token_impl` (lines 721-765) — external linkage.
    - `generate_tokens_impl` (lines 778-843) — external linkage.
    - File-local `CUDA_CHECK` macro copy.
  - Includes: `"config.h"`, `"device_weights.h"`, `"inference.h"` (for the `GenerateDebugResult` struct return type — alternative would be to forward-declare, but the public header is small and idempotent), `"inference_internal.h"`, `"instrument.h"`, `"kv_cache.h"`, `"model_weights.h"`, `"tokenizer.h"`, `<algorithm>`, `<cstdio>`, `<memory>`, `<stdexcept>`, `<string>`, `<vector>`, `<cuda_runtime.h>`.
  - All three `*_impl` functions are declared in `inference_internal.h` (see A1) so the facade can call them.
  - Verify: `wc -l src/inference_loop.cu` returns ~270-310 lines; `nm build/inference_loop.o | grep -E "generate_(next_token|tokens)_impl|generate_tokens_resident_batched_impl"` shows three `T` symbols.

### Phase 5: Reduce `src/inference.cu` to a facade

- [ ] **A5. Rewrite `src/inference.cu` as a thin facade owning only the public entries.**
  - Keep only:
    - The eight public functions from `include/inference.h` (`generate_next_token`, `generate_tokens` single-prompt, `generate_next_token_resident`, `generate_tokens_resident` single-prompt, `generate_tokens_resident` batched, `generate_tokens_resident_debug`, `decode_token`).
    - File top-of-file comment block (lines 1-21) — preserve as-is for repo continuity.
  - Each `generate_*` function delegates: e.g. `int generate_next_token(ModelWeights &w, const std::string &p) { return generate_next_token_impl(w, nullptr, p); }`.
  - `decode_token` (lines 901-903) keeps its `static BPETokenizer tok(TOKENIZER_PATH);` — this is a function-local static, fully self-contained.
  - Includes: `"inference.h"`, `"inference_internal.h"`, `"tokenizer.h"` (for the `BPETokenizer` definition needed by `decode_token`).
  - Verify: `wc -l src/inference.cu` returns ~80-120 lines (down from 904); `grep -n "generate_next_token\b\|generate_tokens\b\|decode_token\b" src/inference.cu` shows seven public-entry definitions; `nm build/inference.o | grep " T " | wc -l` is small (just the public entries).

### Phase 6: Optional consolidation (defer if A1-A5 already lands)

- [ ] **A6. (Optional) Hoist the `CUDA_CHECK` macro into `inference_internal.h`.**
  - Two TUs (`inference_layer.cu`, `inference_loop.cu`) currently each define their own `CUDA_CHECK`. Consolidating into the internal header removes the duplicated macro and keeps definitions in lockstep.
  - Wrap with `#ifndef CUDA_CHECK / #define CUDA_CHECK(...) ... / #endif` to allow per-TU override if needed later.
  - **Defer this step** if it complicates the diff — duplicating a 6-line macro across two TUs is acceptable. Note in `docs/learnings.md` that this was an explicit decision.
  - Verify: if performed, `grep -c "define CUDA_CHECK" src/inference*.cu` returns 0; the header has the only definition.

### Phase 7: Build-system updates

- [ ] **A7. Update the `Makefile` for the new objects.**
  - Add three new object lines to `MAIN_CUDA_OBJECTS` (Makefile:50-54): `$(BUILD_DIR)/inference_chat.o`, `$(BUILD_DIR)/inference_layer.o`, `$(BUILD_DIR)/inference_loop.o`. Keep `$(BUILD_DIR)/inference.o` (the facade is still built).
  - Add the same three objects to `M2M3_TEST_OBJECTS` (Makefile:141-148). Keep `$(BUILD_DIR)/inference.o`.
  - Add three `nvcc` build rules below the existing inference rule (Makefile:88), using the same recipe template:
    ```
    $(BUILD_DIR)/inference_chat.o: $(SRC_DIR)/inference_chat.cu | $(BUILD_DIR)
    	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
    $(BUILD_DIR)/inference_layer.o: $(SRC_DIR)/inference_layer.cu | $(BUILD_DIR)
    	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
    $(BUILD_DIR)/inference_loop.o: $(SRC_DIR)/inference_loop.cu | $(BUILD_DIR)
    	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@
    ```
  - Verify (laptop): `make -n` is meaningless on a CPU box (the `all` target now errors out, as of `c7ea846`); instead `make tests` (CPU laptop) still runs without trying to compile the new `.cu` files (they are not in `TEST_OBJECTS`).
  - Verify (L4): `make clean && make` builds `bin/llm` linking all four inference objects; `make clean && make tests_m2m3` builds `bin/tests_m2m3` linking the same four.

### Phase 8: Documentation refresh

- [ ] **A8. Update `docs/CODEMAPS/architecture.md` and `CLAUDE.md` to match the new file layout.**
  - **`docs/CODEMAPS/architecture.md`** — locate the inference module entry (currently a single line referencing `src/inference.cu`) and replace with: `src/inference.cu` (facade), `src/inference_chat.cu` (chat template), `src/inference_layer.cu` (per-step decoder block + per-head attention), `src/inference_loop.cu` (orchestrator paths), `src/inference_internal.h` (private cross-TU declarations). Do not add the internal header to `include/` documentation — it is project-internal.
  - **`CLAUDE.md`** (project, gitignored) — two specific lines pin behavior to `src/inference.cu`:
    - `CLAUDE.md:86` — "...hands the prompt to `apply_chat_template` in `src/inference.cu`." Update to "in `src/inference_chat.cu`".
    - `CLAUDE.md:98` — "**`include/inference.h` / `src/inference.cu`** — Full 32-layer forward pass with Llama 3 Instruct chat template". Update to reference all four inference TUs (e.g., "**`include/inference.h` / `src/inference*.cu`** — Public API facade plus the chat template, per-layer decoder block, and orchestrator paths").
  - Verify: `grep -nE "inference" docs/CODEMAPS/architecture.md` shows all four `.cu` files plus the internal header; `grep -nE "src/inference" CLAUDE.md` returns no line that still names `src/inference.cu` as the sole owner of the chat template or the forward pass.

### Phase 9: Verification

- [ ] **V1. Quick lane (L4).** `./tools/test_l4.sh --quick` returns M1 7/7 + M2-3 30/30 (the quick lane runs 30 of the 35 registered M2-3 tests; it skips the four `full_forward_*` tests and `layer_streaming_smoke`). This is the primary green light.
- [ ] **V2. Full build sanity (L4).** `make clean && make ARCH=sm_89` produces `bin/llm`; `make tests_m2m3 ARCH=sm_89` produces `bin/tests_m2m3`.
- [ ] **V3. CPU build sanity (laptop).** `make clean && make tests` builds `bin/tests` and runs `./bin/tests 1` through `./bin/tests 7` green. The laptop never compiles the new `.cu` files (they are not in `TEST_OBJECTS`).
- [ ] **V4. Single-token CLI smoke (L4).** `./bin/llm "What is the capital of California?"` returns a valid argmax token (the spec's only mandated E2E check, `llm_part2.md:154`).
- [ ] **V5. Multi-token decode smoke (L4).** Capture the pre-split output to `/tmp/pre_split_output.txt` BEFORE doing A1-A8: `./bin/llm --max-tokens 8 "What is the capital of California?" > /tmp/pre_split_output.txt 2>&1`. After the split, run the same command into `/tmp/post_split_output.txt` and `diff` them. Differences should be limited to telemetry timestamps; the printed token IDs and the decoded "The capital of California is Sacramento." line must match.
- [ ] **V6. Batched B>1 smoke (L4).** `./bin/llm --prompt "A B C" --prompt "D E F" --max-tokens 4` runs the batched path and prints both completions. Compare output structure to pre-split.
- [ ] **V7. File-size check.** `wc -l src/inference*.cu src/inference_internal.h` confirms no file is over 800 lines (target: facade ~100, chat ~70, layer ~460, loop ~290, internal header ~60). All five files together should sum to ~980 lines (slightly above original 904 due to per-TU includes, namespace wrapping, and macro duplication; this is expected and acceptable).
- [ ] **V8. Symbol-count sanity (L4).** Capture `nm build/inference.o | grep -c " T "` from the pre-split build (one number, e.g. `8`). After the split, run `nm build/inference.o build/inference_chat.o build/inference_layer.o build/inference_loop.o | grep -c " T "` — the post-split number should equal the pre-split count plus the count of cross-TU helpers that lost `static` linkage (`apply_chat_template`, `compute_lm_head_logits`, `forward_step`, `alloc_rope_tables`, `load_resident_layers`, `validate_equal_lengths`, `generate_next_token_impl`, `generate_tokens_impl`, `generate_tokens_resident_batched_impl` = 9 new external symbols). Expected delta: post = pre + 9.
- [ ] **V9. M2-3 fixture coverage map.** Confirm the existing M2-3 tests (35 named tests; run `./bin/tests_m2m3 --list` for the canonical list) still exercise each new TU. Many M2-3 tests bypass the public API and call kernels or static helpers directly (e.g. `final_rmsnorm_fixture` invokes `gpu_rmsnorm` directly at `tests/test_m2m3.cpp:1855`; `lm_head_last_token_only` calls a *test-local* `compute_lm_head_logits` helper at `tests/test_m2m3.cpp:436`, not the inference-side one). Coverage is therefore split into "direct kernel/struct tests" (which keep working because the new files re-export the same kernels and structs) and "end-to-end tests" (which exercise the new `*_impl` orchestrators):
  - `inference_chat.cu` — exercised end-to-end by the full-forward-only tests (`full_forward_hello`, `full_forward_medium_prompt`, `full_forward_kv_cache_one_token_parity`, `full_forward_resident_one_token_parity`) and `batched_b2_distinct_parity`; all of these enter through a `generate_*` call that runs `apply_chat_template` first.
  - `inference_layer.cu` — exercised directly by `decoder_block_layer0_fixture` (per-layer fixture) and `attention_output_full_fixture` (attention slice); transitively by every full-forward test that goes through `forward_step`.
  - `inference_loop.cu` — exercised end-to-end by the four `full_forward_*` tests and `batched_b2_distinct_parity` (these are the only tests that enter through one of the three `*_impl` orchestrators). `kv_cache_bounds_checks`, `resident_layer0_weight_smoke`, and `layer_streaming_smoke` look related but are *not* coverage for this TU — they directly construct `KVCache`, `DeviceModelWeights`, and `ModelWeights` respectively and validate those classes; the orchestrator path is never invoked.
  - `inference.cu` (facade) — exercised by every test that calls a public `generate_*` (the full-forward subset above, plus `batched_b2_distinct_parity`). The kernel-/struct-level fixtures (`rmsnorm_fixture`, `q_projection_fixture`, etc.) bypass the facade and validate kernel-level correctness; they are unaffected by the split.
  - Run `./bin/tests_m2m3 --list` and confirm the test name list is unchanged from the pre-split run (35 tests in `--list`, same names; quick lane runs 30 of those by skipping `full_forward_*` and `layer_streaming_smoke`).
- [ ] **V10. No new include leaks.** `grep -r "inference_internal.h" --include="*.cpp" --include="*.h" --include="*.cu" .` should match only files under `src/inference*.cu`. `main.cpp` and `tests/test_m2m3.cpp` must NOT include it.
- [ ] **V11. Header-only public API check.** `grep -c "include.*inference.h\b" main.cpp tests/test_m2m3.cpp` returns at least one match per file; `grep -c "include.*inference_internal.h" main.cpp tests/test_m2m3.cpp` returns 0.

---

## Risks

- **HIGH (correctness):** the `forward_step` function is the heart of inference. Any subtle change to its `cudaMalloc`/`cudaFree` ordering, the per-layer Stopwatch hierarchy, or the `cudaDeviceSynchronize` placement risks introducing race conditions or numerical drift that V1 catches but is hard to diagnose post-hoc. Mitigation: A3 moves `forward_step` verbatim — no re-ordering, no inlining, no signature change. The 2-layer math trace (mental compile, V8) confirms identical instruction order.

- **HIGH (build-system):** if the four new objects are added to `MAIN_CUDA_OBJECTS` but missing from `M2M3_TEST_OBJECTS` (or vice versa), one binary builds and the other fails to link with "undefined reference to apply_chat_template" or similar. Mitigation: A7 explicitly lists both updates in the same step; V2 builds both binaries from clean.

- **MEDIUM (linker):** if a function is moved between TUs but its anonymous-namespace wrapping is dropped without giving it external linkage, the new TU's caller will see "undefined reference". Conversely, if the wrapping is kept and the caller is in a different TU, the same error appears. Mitigation: V8 symbol-count check explicitly tracks which 9 helpers gained external linkage. The plan above documents each one.

- **MEDIUM (include cycles):** `inference_internal.h` references `GenerateDebugResult` (from `inference.h`), so it must include the public header. If `inference.h` ever pulls in `inference_internal.h`, we get a cycle. Mitigation: explicitly forbid this in the internal header's top comment ("Do not include from `include/inference.h`"). V10 and V11 catch leaks.

- **MEDIUM (telemetry):** the `Stopwatch::print_summary()` static state is shared across TUs. If construction order across new TUs accidentally changes the order that timers register with the singleton, the printed summary order could shift (cosmetic, not numeric). Mitigation: timers are RAII-scoped inside specific functions, so the summary order depends on call order at runtime, which is unchanged. V5 diff comparison catches it.

- **LOW (extension choice):** `inference_chat.cu` does not currently call CUDA APIs; `nvcc` compiles it as device-host code at trivial cost. If the build time on the L4 box becomes a concern, switching it to `.cpp` is a one-line Makefile change later. Not blocking.

- **LOW (CUDA_CHECK macro):** if A6 is deferred and the macro is duplicated across two TUs, a future bug fix to the macro must touch both. Mitigation: comment in each duplicate definition pointing at the other; or just do A6 (low cost).

- **LOW (cosmetic diff):** the four new files plus the heavy diff to `src/inference.cu` plus the Makefile updates means the PR is large for what is a no-op refactor. Mitigation: the commit message and PR body explicitly call it a "pure file-split refactor — no behavior change" and list the V1/V5/V8 evidence.

---

## Estimated Complexity: MEDIUM

- A1 (internal header, ~50 lines): 10 min — purely declarations, no logic.
- A2 (chat-template TU, ~70 lines): 10 min — straight cut/paste with linkage adjustment.
- A3 (decoder-block TU, ~460 lines): 30 min — the largest TU; care needed to preserve every line of `forward_step` verbatim. Read-aloud check the moved code against the original.
- A4 (orchestrator TU, ~290 lines): 30 min — three function moves, plus header forward-declarations to coordinate with A1.
- A5 (facade reduction, ~100 lines): 15 min — keep top-of-file comment, replace body with seven thin delegators + `decode_token`.
- A6 (optional macro hoist): 5 min if performed, 0 if deferred.
- A7 (Makefile, ~12 lines added across two object lists + three new rules): 10 min — mechanical, but careful to update both `MAIN_CUDA_OBJECTS` and `M2M3_TEST_OBJECTS`.
- A8 (codemap doc + CLAUDE.md): 10 min — `docs/CODEMAPS/architecture.md` module map plus two CLAUDE.md lines (86 and 98).
- V1-V11 (verification): one L4 burst (~10 min for `--quick` lane) + one laptop CPU build check (~3 min) + symbol-count diff (~5 min).
- **Total wall:** ~90-120 min coding + ~20 min verification. Larger than the `main.cpp` cleanup (it was LOW) because the file is bigger and the moves cross more boundaries, but still single-PR-sized.

---

## Out of Scope

- Changing the `include/inference.h` API or any kernel signature.
- Touching `tests/test.cpp`, `tests/test_api.h`, `tests/test_api.cpp`, or `tests/test_m2m3.cpp` (the M2-3 test harness still includes only the public header).
- Adding new functionality (new generate variants, sampling, top-k, temperature, etc.).
- Optimizing the forward pass or kernels (no fusion, no rewrite of `forward_step`'s allocation pattern).
- Promoting `inference_internal.h` into `include/` for external consumption.
- Renaming any of the existing public functions.
- Splitting any file other than `src/inference.cu`. (`main.cpp` is the only other near-the-cap source; that was handled by the prior plan.)
- Switching kernel objects (matmul, rmsnorm, rope, etc.) to a different layout.
- Updating `docs/learnings.md` unless A6 is deferred and produces a learning entry.

---

**WAITING FOR CONFIRMATION**: Proceed with this plan? (yes / no / modify)

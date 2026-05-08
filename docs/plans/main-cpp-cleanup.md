# `main.cpp` Cleanup Plan — Strip Above-Spec Bloat

## Goal

Tighten `main.cpp` (and the Makefile rule that supports its CPU stub) so the CLI demo only carries what the project actually requires. The graded paths (`bin/tests`, `bin/tests_m2m3`) are untouched; the inference API, kernels, and decoder graph are untouched; this is a demo-binary and build-rule tightening.

After this plan: `bin/llm` is CUDA-only, `--interactive` is exclusively the bonus showcase (multi-token autoregressive decode), the no-arg "Hello world" fallback is gone, and every doc/comment that advertised the old build or CLI behavior is updated to match. The three output-print blocks are *not* consolidated — see A5 for why.

**Intentional behavior changes** (this plan is not pure cosmetic cleanup):

| Path | Before | After |
|---|---|---|
| no-CUDA `make` | builds a stub `bin/llm` that errors at runtime | refuses to link, exits 1 with a build-time error |
| `./bin/llm` (no args) | runs with default `"Hello world"` prompt | prints usage and exits 1 |
| `./bin/llm --interactive` (no `--max-tokens`) | enters REPL, defaults to `max_tokens=1` | exits 1 with "requires --max-tokens >= 2" |
| `make run` | runs `./bin/llm` (relied on the default prompt) | runs `./bin/llm "Hello world"` explicitly |
| README.md / CLAUDE.md | document CPU-fallback build | document CUDA-required build |

No new inference behavior, no kernel work, no `inference.h` API changes.

**Required-test invariant:** the professor-facing M1 harness remains buildable and behaviorally unchanged. `make tests` must continue to build `bin/tests` on both CUDA and CPU-only machines, and the `TestAPI` surface (`tests/test_api.h`) remains untouched.

---

## Must-Read Docs (in this order)

1. **`docs/assignment/llm_part1.md`** — Part 1 spec. Key: §3.1 deliverables (tokenize, embed, matmul); line 90 ("batching and KV caching are optional, +5% each").
2. **`docs/assignment/llm_part2.md`** — Part 2 spec. Key: line 7 ("By the end of this part, your system will accept a prompt and **generate a token**"). The single-token greedy path is the only end-to-end requirement.
3. **`CLAUDE.md`** (project) — Build commands, file ownership constraints (`tests/test.cpp` and `tests/test_api.h` are read-only — neither is touched here).
4. **`main.cpp`** (current state) — the file under the knife.
5. **`Makefile`** — verify the `CUDA_ENABLED` guard pattern (line 29-33) and how `tests_m2m3` already gates on it (line 129+) — we mirror that for `bin/llm`.

Files this plan touches: `main.cpp`, `Makefile`, `README.md`, `CLAUDE.md`, `docs/CODEMAPS/architecture.md`, `docs/generated/development-guide.md`. Nothing else.

---

## Discussion Highlights

- **Grading does not invoke `main.cpp`.** M1 grading runs `bin/tests` (its own object list: `test.o`, `test_api.o`, `tokenizer_bpe.o`, `loader.o`, `matmul.o`). M2-3 internal tests run `bin/tests_m2m3`. `bin/llm` is purely a demo binary — `main.cpp` simplifications cannot break grading.
- **Mandatory tests are protected.** This plan may change the default `all`/`bin/llm` build behavior, but it must not change the professor-facing `make tests` build graph or the `TestAPI` implementation path. The required harness depends on `tests/test.cpp`, `tests/test_api.cpp`, `tests/test_api.h`, `src/tokenizer_bpe.cpp`, `src/loader.cpp`, and either `kernel/matmul.cu` or `kernel/matmul_cpu.cpp` depending on CUDA availability.
- **The CPU stub in `main.cpp` is dead weight.** Without `nvcc`, `bin/llm` already prints `"Error: inference requires CUDA..."` and exits 1. The `#ifdef CUDA_ENABLED` block exists only to keep the no-CUDA build linkable. Mirroring the `tests_m2m3` pattern in the Makefile (refuse to build `bin/llm` without CUDA) lets us delete the stub entirely. CPU-only `bin/tests` (M1) still builds because its object list does not include `main.o`.
- **`--interactive` only earns its keep with `max_tokens > 1`.** The REPL pays a multi-second warmup to keep weights resident; if `max_tokens == 1`, every prompt returns one greedy token and the user might as well run `./bin/llm "prompt"`. Rejecting `--interactive` when `max_tokens == 1` is a correctness-clarifying gate, not a feature change. The demo script (`scripts/demo-start.sh:20`) defaults to 32, so this does not regress the TA demo.
- **The `"Hello world"` default-prompt fallback is invented.** The spec never asked for an arg-less mode. Removing it means `./bin/llm` with no args prints usage and exits non-zero — exactly what every other CLI does.
- **Output printing stays as-is.** Path-1 (single token), path-2 (multi-token), and path-3 (batch B>1) look similar but have distinct byte-level formatting. A5 documents why the helper refactor was dropped.
- **`tools/test_l4.sh` runs on L4** (CUDA always present), so the CUDA test lanes are unaffected.

---

## Action Items

Mark each item with `[x]` when complete. Verification commands listed inline.

### Code cleanup

- [ ] **A1. Make `bin/llm` CUDA-only in the Makefile *and update the docs that advertise CPU-fallback*.**
  - In `Makefile`, wrap only the `all:` / `bin/llm` rule and the `bin/llm`-only dependencies inside an `ifeq ($(CUDA_ENABLED),1)` / `else` block, mirroring the `tests_m2m3` pattern (line 129+). The `else` branch should produce a clear error: `@echo "ERROR: bin/llm requires CUDA (nvcc not found)"; @exit 1`.
  - **Do not move or wrap the mandatory test harness.** Keep `tests`, `TEST_MATMUL`, `TEST_OBJECTS`, `$(BIN_DIR)/tests`, `$(BUILD_DIR)/test.o`, `$(BUILD_DIR)/test_api.o`, `$(BUILD_DIR)/tokenizer_bpe.o`, `$(BUILD_DIR)/loader.o`, and `$(BUILD_DIR)/matmul_cpu.o` available outside the CUDA-only `bin/llm` guard. `make tests` must remain valid on a CPU-only box.
  - **Do not touch professor-owned/test-surface files:** `tests/test.cpp`, `tests/test_api.h`, and `tests/test_api.cpp`. Also do not alter `src/tokenizer_bpe.cpp`, `src/loader.cpp`, or `kernel/matmul_cpu.cpp` as part of this cleanup; preserving their current behavior preserves the mandatory TestAPI checks.
  - Remove `MATMUL_OBJECT := $(BUILD_DIR)/matmul_cpu.o; OBJECTS += $(MATMUL_OBJECT)` from the no-CUDA branch (lines 56-59) since `OBJECTS` is now CUDA-only. The `TEST_MATMUL` selector for the M1 `tests` target (lines 105-109) still picks `matmul_cpu.o` correctly.
  - **Doc updates (same commit as the Makefile change):**
    - `README.md:5` — reword "A CPU matmul fallback builds when `nvcc` is unavailable." Replace with: "Building `bin/llm` requires `nvcc`; the M1 `bin/tests` target still builds CPU-only via the `matmul_cpu` fallback."
    - `README.md:18` — change comment "Build (CPU-only if nvcc is not found)" to "Build (requires nvcc for `bin/llm`; `bin/tests` builds CPU-only)".
    - `CLAUDE.md:35` — replace the sentence "Without CUDA: CPU matmul fallback only, `bin/llm` prints an error." with "Without CUDA: `make` (default target `bin/llm`) fails fast at link time; only the M1 `bin/tests` target still builds (using the CPU `matmul_cpu` fallback)."
    - `docs/CODEMAPS/architecture.md:63` — replace "no -> kernel/matmul_cpu.cpp linked, bin/llm prints error on run" with "no -> bin/llm refuses to link; bin/tests still builds via kernel/matmul_cpu.cpp".
    - `docs/generated/development-guide.md:10` — replace "nvcc (optional — CPU fallback available)" with "nvcc (required for bin/llm; bin/tests builds CPU-only)".
  - Verify on L4 (CUDA on): `make` builds `bin/llm`; `make tests` builds `bin/tests`; `make tests_m2m3` builds `bin/tests_m2m3`.
  - Verify locally (CUDA off): `make tests` still succeeds; `make` (default `all`) prints the error and exits 1; the three doc lines no longer claim CPU-fallback for `bin/llm`.
  - Verify mandatory-harness dry run locally (CUDA off): `make -n tests` shows only `g++` compilation/linking for `test.o`, `test_api.o`, `tokenizer_bpe.o`, `loader.o`, and `matmul_cpu.o`; it must not try to build `main.o`, `inference.o`, `device_weights.o`, `kv_cache.o`, or any `.cu` object.

- [ ] **A2. Delete the `#ifndef CUDA_ENABLED` branch from `main.cpp`.**
  - Remove the entire `#ifndef CUDA_ENABLED ... #else ... #endif` scaffolding around `int main(...)` (lines 53-60 and the trailing `#endif` on line 244).
  - Move the `#include "inference.h"` and `print_usage()` helper out of the surviving `#ifdef CUDA_ENABLED` block (currently lines 29-50) into the top-level header includes — they are unconditional now.
  - Verify: `grep -c CUDA_ENABLED main.cpp` returns `0`. `make` on L4 still produces a working `bin/llm`.

- [ ] **A3. Gate `--interactive` on `max_tokens > 1` *and update the docs/comments that advertise bare `--interactive`*.**
  - Inside the `if (interactive)` block (currently main.cpp:105), add an early check after the prompts/positional check: `if (max_tokens < 2) { std::fprintf(stderr, "Error: --interactive requires --max-tokens >= 2 (single-token mode does not benefit from resident weights; use ./bin/llm \"prompt\" instead)\n"); return 1; }`.
  - Update `print_usage()` to document the constraint: `--interactive    REPL mode (requires --max-tokens >= 2); ...`.
  - **Doc/comment updates (same commit):**
    - `main.cpp:18` — change the file-header usage comment from `./bin/llm --interactive [--max-tokens N]        (REPL on stdin)` to `./bin/llm --interactive --max-tokens N    (REPL on stdin, N >= 2)`. Drop the brackets that imply `--max-tokens` is optional.
    - `README.md:12` — the "Live demo" line currently says `./bin/llm --interactive`. Change to `./bin/llm --interactive --max-tokens 32` (matching what `scripts/demo-start.sh` actually invokes) so the README and the script agree.
    - `CLAUDE.md:30` — change `./bin/llm --interactive [--max-tokens N]  # REPL: load weights once, prompt loop on stdin` to `./bin/llm --interactive --max-tokens N  # REPL (N >= 2): load weights once, prompt loop on stdin`.
  - Verify: `./bin/llm --interactive` (no `--max-tokens`) prints the error and exits 1; `./bin/llm --interactive --max-tokens 1` prints the same error; `./bin/llm --interactive --max-tokens 8` enters the REPL as before; `scripts/demo-start.sh` (defaults to `--max-tokens 32`) still works; `grep -nE 'interactive\s*\[--max-tokens|interactive\s*$' main.cpp README.md CLAUDE.md` returns no remaining bare-`--interactive` advertising.

- [ ] **A4. Drop the `"Hello world"` default-prompt fallback *and update `make run` so it stays useful*.**
  - Remove the `if (prompts.empty() && positional_prompts.empty()) { prompts.push_back("Hello world"); }` branch (currently main.cpp:156-158).
  - When both intake forms are empty in non-interactive mode, print usage and exit 1 (call `print_usage(argv[0]); return 1;`).
  - **`Makefile:95-96` — the `run` target currently runs `./bin/llm` with no args, which would now fail.** Update the rule to: `./$(BIN_DIR)/$(TARGET) "Hello world"` so `make run` still produces a smoke-test inference. (CLAUDE.md:22 documents `make run`; preserving its observable behavior keeps that doc accurate without a separate edit.)
  - Verify: `./bin/llm` (no args) prints usage and exits 1; `make run` still runs a single-token inference on "Hello world" and exits 0.

- [ ] **A5. ~~Consolidate the three output-print blocks~~ — DROPPED.**
  - **Why dropped (response to plan review F2):** the three stanzas (main.cpp:209-214, 222-225, 232-240) are not redundant — they are three distinct output formats:
    - Path-1: `Generated token: %d` (integer ID, no `:` after `tokens`), `Decoded text:    ` (4 spaces), `Full output:     ` (5 spaces). No indent.
    - Path-2: `Generated %zu tokens:` (trailing `:`!), same `Decoded text:    ` / `Full output:     ` alignment, no indent.
    - Path-3: `Output [%zu]:` header, then 2-space indent on every line, `Generated %zu tokens` (no `:`), `Decoded text: ` (1 space after colon), `Full output:  ` (2 spaces).
  - The Path-3 batched single-token edge case (`--max-tokens 1` with B>1) currently prints `Generated 1 tokens`, not `Generated token: %d`. Any helper that uses `ids.size() == 1` to switch to the integer-ID format would change that path-3 edge case.
  - A faithful helper would need an explicit `Mode { SINGLE, MULTI, BATCHED }` parameter and three branches reproducing each format string verbatim. That helper would be roughly the same line count as the three stanzas it replaces, with added dispatch complexity. Net negative.
  - **The duplication isn't real.** "Generated / Decoded text / Full output" are the same words but different format strings (alignment, punctuation, indent). Leaving the three stanzas as-is is the correct call.
  - If the user wants this revisited later, the right scope is a separate plan that explicitly accepts a *unified* output format (breaking byte-identity) — not this cleanup.

### Verification on L4

- [ ] **V1. Quick lane.** `./tools/test_l4.sh` (default lane: M1 + fast M2-3). Should remain green.
- [ ] **V2. Single-token smoke.** `./bin/llm "What is the capital of California?"` produces a sensible argmax token (the spec's only mandated E2E check, `llm_part2.md:154`).
- [ ] **V3. Multi-token KV-cached decode.** `./bin/llm --max-tokens 8 "What is the capital of California?"` produces 8 tokens with the unchanged decode path.
- [ ] **V4. Interactive REPL.** `./bin/llm --interactive --max-tokens 8` enters the REPL, accepts a prompt, generates ≥2 tokens. `./bin/llm --interactive` (no `--max-tokens`) prints the new error and exits 1.
- [ ] **V5. Batched B>1.** `./bin/llm --prompt "A B C" --prompt "D E F" --max-tokens 4` runs the batched path and prints both completions.
- [ ] **V6. CPU-only build (laptop).** `make tests && ./bin/tests 1` still works on a box without `nvcc`. `make` alone fails fast with the new Makefile error.
- [ ] **V7. `make run` regression check.** `make run` builds `bin/llm` and runs `./bin/llm "Hello world"` (single-token inference) and exits 0. Catches finding F1 from the plan review.
- [ ] **V8. Doc-claim consistency.** Run:
  ```
  grep -nE "CPU.*fallback|CPU.?only|nvcc.*not.*found|nvcc.*optional|prints? error on run" \
      README.md CLAUDE.md docs/CODEMAPS/architecture.md docs/generated/development-guide.md
  ```
  No surviving line should claim `bin/llm` builds without CUDA. `bin/tests` is still allowed (and documented) to build CPU-only — distinguish in any updated phrasing.
- [ ] **V9. `--interactive` doc consistency.** Run `grep -nE "interactive\s*\[--max-tokens|interactive\s*$|interactive[^-]*REPL on stdin" main.cpp README.md CLAUDE.md`. No surviving line should advertise bare `--interactive` without `--max-tokens N` (N >= 2). Catches finding F1 from the latest plan review.
- [ ] **V10. Mandatory professor-test harness preservation.** On a CPU-only checkout, run `make clean && make tests && ./bin/tests 1`. Then run `make -n tests` and confirm the dry-run graph contains no `main.o`, `inference.o`, `device_weights.o`, `kv_cache.o`, or CUDA kernel object required only by `bin/llm`/`tests_m2m3`. On L4, `make tests` must also remain green before running any demo-binary checks.

---

## Risks

- **HIGH (process):** the CLAUDE.md ground rule that grading runs through `tests/test_api.h` must hold. If the autograder ever invokes `bin/llm`, this plan would regress it. Mitigation: the spec (`llm_part1.md:48`, `llm_part2.md:174`) only references the test API and `reference.py` numerical comparison — `bin/llm` is for human inspection.
- **MEDIUM:** the Makefile change has to keep the M1 `tests` target buildable on no-CUDA boxes. The `TEST_MATMUL` selector handles this, but A1's Makefile edit must be careful not to delete the `matmul_cpu.o` build rule (Makefile line 72-73) or accidentally pull `main.o`/CUDA-only objects into `TEST_OBJECTS`. Mitigation: V6 and V10 verify CPU-only `make tests` still works and that the dry-run graph stays on the mandatory TestAPI path.
- **MEDIUM:** if an unknown CPU-only autograder habit runs bare `make` before `make tests`, this plan intentionally makes bare `make` fail. That is outside the documented mandatory TestAPI harness, but it is the only plausible grading-procedure collision. Mitigation: keep this risk explicit; if avoiding it matters more than deleting the stub, do not perform A1 and only do the `main.cpp` cleanup that is CUDA-build-only.
- **LOW:** anyone with muscle memory typing `./bin/llm` (no args) loses the smoke-test fallback. Mitigation: usage text now prints; one-line workaround is `./bin/llm "Hello world"`.
- **LOW:** the demo script uses `--max-tokens 32`, so A3's gate does not affect it. If a future user runs `--interactive --max-tokens 1` they will hit a hard error — exactly the desired behavior.

---

## Estimated Complexity: LOW

- A1 (Makefile guard + 5 doc lines): ~10 Makefile lines + 5 short doc edits across `README.md`, `CLAUDE.md`, `docs/CODEMAPS/architecture.md`, `docs/generated/development-guide.md`. Mechanical.
- A2 (delete CPU branch in main.cpp): ~10 lines deleted.
- A3 (interactive gate + 3 doc/comment updates): ~5 lines added in main.cpp + 3 short edits to `main.cpp:18`, `README.md:12`, `CLAUDE.md:30`.
- A4 (drop default prompt + 1 Makefile line): ~3 lines deleted in main.cpp + 1 line edited in `Makefile:96`.
- ~~A5~~ DROPPED — see entry above for rationale.
- Total wall: ~30 min coding (smaller than the previous estimate now that A5 is gone) + one L4 burst (`./tools/test_l4.sh` quick lane + V2-V5 smoke). Reversible — single commit, easy revert if anything regresses.

---

## Out of Scope

- Changing the `inference.h` API or any kernel.
- Touching `tests/test.cpp`, `tests/test_api.h`, or `tests/test_m2m3.cpp`.
- Adding new features (sampling, temperature, top-k, etc.).
- Removing the resident-weights or batched-decode paths — both are working bonus features and are kept.
- Renaming `bin/llm` or changing the build target name.

---

**WAITING FOR CONFIRMATION**: Proceed with this plan? (yes / no / modify)

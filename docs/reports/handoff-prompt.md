# Handoff Prompt for Next Session

Copy everything below the `---` line into a fresh Claude Code session.

---

I'm preparing for a CS265 TA code-review interview for my from-scratch Llama 3 8B inference engine (C++17 + CUDA, no PyTorch at runtime). I need help with two things in this session:

1. **Interview prep** — walk me through likely TA questions, drill me on parts of the code I should expect to explain line-by-line, and help me practice live demo flow.
2. **Verify all spec requirements are met** — re-audit the codebase against the actual class spec (`docs/llm_part1.md` and `docs/llm_part2.md`) so I don't show up missing something.

## Context you need to read first (in this order)

1. `docs/llm_part1.md` — class spec Part 1 (M1 deliverables, grading rubric).
2. `docs/llm_part2.md` — class spec Part 2 (M2 + M3, including §3.1 Step 6 = the conclusive end-to-end test).
3. `CLAUDE.md` — project-level instructions, build commands, file-ownership constraints. **Note: `tests/test.cpp` and `tests/test_api.h` are read-only (graders may replace them).**
4. `README.md` — see "Notes for graders" section.
5. `docs/code-review-prep.md` — my prep doc with: demo runbook, design-decision register (9 entries), 16 anticipated TA questions with prepared answers. **This is the primary doc for interview prep.**
6. `docs/plans/submission-wrap-up.md` — the most recent submission-wrap-up plan (already executed; A1–A10 done).

## Current state of the project

- On `main` branch. Recent commits: `1d22f30` (interactive REPL + demo scripts), `8a77742` (untrack docs/), `7563b72` (TODO #2 B>1 batching, +5% bonus).
- Working tree clean.
- M1 7/7 PASSED on L4 (sm_89, FP32 path).
- The conclusive E2E test passes: `bin/llm "What is the capital of California?"` → `"The capital of California is Sacramento."`
- TODO #1 (KV cache, +5% bonus): COMPLETE.
- TODO #2 (B>1 batching, +5% bonus): COMPLETE. Validated by `batched_b2_distinct_parity` (B=1 vs B=2 within 1e-3 hidden-state diff).

## L4 demo flow (use these scripts, not raw gcloud)

The VM is `cs265-l4`; the demo scripts auto-detect the zone via `gcloud compute instances list`, so it does not need to match `PREFERRED_ZONE`. Personal config in `.l4-config.env` (gitignored).

```bash
./scripts/demo-start.sh           # ~5 min before TA: start VM, SSH in, drop into REPL
                                  # (default --max-tokens 32)
./scripts/demo-stop.sh            # after demo: stop VM
```

`demo-start.sh` SSHs into the VM and runs `./bin/llm --interactive --max-tokens 32`. Cold resident BF16 weight load is ~165s (one-time). Each prompt after warmup is ~3 seconds.

## Important constraints / gotchas

- **BF16 path is for the bonus only.** Required M1 graded matmul uses FP32. Per the class discussion board, TAs relax epsilon for BF16 — flag this in the interview if asked about precision.
- **L4 has 24 GB VRAM.** FP32 residency (32 GB) doesn't fit; that's why the bonus path uses BF16. Don't let the TA assume it's a free choice.
- **`docs/` is gitignored** (per repo policy). Plans, prep docs, learnings stay local. Only README and source are tracked.
- **Stop the VM after every burst.** SPOT ~$0.20–0.30/hr, STANDARD ~$0.72/hr (set in `.l4-config.env` via `PROVISIONING_MODEL`). `demo-stop.sh` handles it. If you've captured a custom image (`tools/create_custom_image.sh`), prefer **delete** over stop — the image is the source of truth and re-provision rebuilds in ~60s.

## What I want from you in this session

Start by reading the 6 docs above (in order). Then:

1. Ask me 5–10 likely TA questions from `docs/code-review-prep.md` §3 to drill me. Push back on weak answers; correct me when I'm wrong.
2. Pick one CUDA kernel (`kernel/matmul.cu` is the most likely target) and walk me through it line-by-line, checking my understanding at each block. Catch me on any line I can't explain.
3. Audit the codebase against the spec: open `docs/llm_part1.md` and `docs/llm_part2.md` and confirm every required functionality has corresponding implementation. Surface anything missing or undertested.
4. If you find anything broken, propose the smallest fix and ask before changing code.

Do NOT run the L4 burst again unless I ask — it costs money and 5+ minutes. The conclusive E2E was already verified in the most recent commit's prep work.

Be terse. Don't restate this prompt. Don't add filler. If you have a recommendation, state it and ask once — don't list options.

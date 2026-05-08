# CS265 LLM Inference Engine — Documentation Index

**Generated:** 2026-03-29 | **Updated:** 2026-05-08 | **Scan Level:** Exhaustive | **Mode:** Initial Scan

## Project Overview

- **Type:** Monolith — C++17/CUDA LLM inference engine
- **Primary Language:** C++17 + CUDA
- **Architecture:** Pipeline-based (Tokenizer → Weight Loader → GPU Operators → CUDA Kernels → Decode)
- **Model:** Meta Llama 3 8B Instruct
- **Status:** Milestones 1-3 complete on L4 (sm_89) — 7 M1 tests + 38 M2-3 tests passing, CLI inference validated against PyTorch. KV cache + resident BF16 and B>1 batching bonus credit shipped.

## Quick Reference

- **Tech Stack:** C++17, CUDA, GNU Make, Python (tooling)
- **Entry Points:** `main.cpp` (CLI inference), `tests/test.cpp` (M1 harness), `tests/test_m2m3_main.cpp` (M2-3 harness)
- **Build:** `make` (release) / `make tests` (M1) / `make tests_m2m3` (M2-3, CUDA required)
- **Architecture Pattern:** Pipeline-based inference with link-time GPU/CPU dispatch

## Generated Documentation

- [Project Overview](./generated/project-overview.md) — Executive summary, tech stack, architecture summary
- [How This Project Meets The Assignment](./generated/requirements-implementation.md) — Plain-English walkthrough of the implemented requirements
- [Source Tree Analysis](./generated/source-tree-analysis.md) — Annotated directory tree, critical directories, entry points
- [Development Guide](./generated/development-guide.md) — Prerequisites, build commands, testing, GCP workflow
- [CUDA Kernel Documentation](./generated/hardware-documentation.md) — Tiled GEMM kernel architecture, optimization details
- [CODEMAPS/](./CODEMAPS/) — Token-lean architecture reference (pipeline, kernels, data format, dependencies)

## Existing Documentation

- [README.md](../README.md) — Project overview, quick start, GCP setup, model download
- [CLAUDE.md](../CLAUDE.md) — AI assistant project instructions and context
- [Development Journal](./JOURNAL.md) — Chronological development log
- [Assignment Part 1](./assignment/llm_part1.md) — Course assignment specification (Part 1)
- [Assignment Part 2](./assignment/llm_part2.md) — Course assignment specification (Part 2)
- [Mid Check-in](./reports/mid-checkin.md) — Mid-project check-in report
- [Milestone 1 Report](./reports/Milestone1-Report.md) — Milestone 1 completion report
- [Code Review Prep](./reports/code-review-prep.md) — Code review preparation notes
- [Handoff Prompt](./reports/handoff-prompt.md) — Session handoff prompt
- [Milestone 1 Spec](./plans/milestone1_spec.md) — Milestone 1 specification
- [Milestone 2-3 Plan](./plans/milestone2-3-plan.md) — Milestone 2-3 implementation plan
- [Milestone 2-3 Remaining](./plans/milestone2-3-remaining.md) — Milestone 2-3 remaining work tracker
- [Diagrams](./diagrams/) — Mermaid sources + rendered PNGs (pipeline, KV cache)
- [Notes](./notes/) — Working notes

## Getting Started

1. Install prerequisites: g++ (C++17), Python 3, GNU Make, CUDA toolkit (for GPU builds)
2. Set up Python venv and download model weights (see [Development Guide](./development-guide.md))
3. Run `make` to build the main binary, `make tests` to build M1 tests, `make tests_m2m3` to build M2-3 tests
4. For GPU testing, follow the GCP workflow in the development guide
5. Run `./bin/tests <1-7>` to verify each M1 test passes
6. Run `./bin/tests_m2m3 --list` to see M2-3 tests, `./bin/tests_m2m3 <name>` to run individual tests
7. Run `./bin/llm "prompt"` for single-token CLI inference (CUDA required)

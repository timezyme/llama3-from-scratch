# Source Tree Analysis

**Generated:** 2026-03-29 | **Updated:** 2026-05-08 | **Scan Level:** Exhaustive

## Annotated Directory Tree

```
CS265-llm-starter-main/
├── main.cpp                      # [ENTRY POINT] CLI single-token inference
├── config.h                      # Llama 3 8B architecture constants
├── Makefile                      # Build system: auto-detects nvcc, CUDA/CPU matmul toggle
├── README.md                     # Project overview, setup guide, GCP instructions
├── CLAUDE.md                     # AI assistant context and project instructions
├── .env                          # Harvard GitLab credentials (gitignored from sharing)
├── .gitignore                    # Excludes assets/, build/, bin/, .venv/
│
├── include/                      # C++ header files
│   ├── prelude.h                 # Common STL imports and type aliases (float_t, size_t)
│   ├── tokenizer.h               # LLMTokenizer interface + BPETokenizer class declaration
│   ├── loader.h                  # LlamaDumpLoader class: binary weight loading interface
│   ├── milifloat.h               # bf16_to_float() and half_to_float() inline converters
│   ├── operator.cuh              # AbstractOperator base class (stub, throws on all methods)
│   ├── model_weights.h           # LayerWeights, GlobalWeights, ModelWeights: per-layer + resident BF16
│   ├── kv_cache.h                # Device-side per-layer K/V buffers (multi-token / batched decode)
│   ├── instrument.h              # Header-only Stopwatch and probe_vram telemetry
│   ├── device_weights.h           # DeviceModelWeights: GPU-resident BF16 weight manager
│   └── inference.h               # Forward-pass entry points: generate_next_token, generate_tokens
│
├── src/                          # C++ implementation files
│   ├── tokenizer_bpe.cpp         # [229 lines] BPE tokenizer: vocab loading, encode/decode, merge loop
│   ├── loader.cpp                # Weight loader: binary header parsing, BF16/FP16/FP32 conversion
│   ├── model_weights.cpp         # Weight loading with transpose-at-load for all 32 layers
│   ├── device_weights.cu         # GPU-resident BF16 weight upload/free
│   ├── inference.cu              # Public API facade (forward_pass entry points)
│   ├── inference_internal.h      # Cross-TU private declarations for inference_*.cu
│   ├── inference_chat.cu         # Chat template formatting
│   ├── inference_layer.cu        # Single decoder block (attention + FFN)
│   ├── inference_loop.cu         # Autoregressive decode orchestrator (single / KV / batched)
│   └── kv_cache.cu               # cudaMalloc/cudaFree per-layer K/V buffers
│
├── kernel/                       # GPU compute kernels
│   ├── kernels.cuh               # Kernel function signatures (all GPU operators)
│   ├── matmul.cu                 # [CUDA] Double-buffered tiled GEMM with shared memory
│   ├── matmul_cpu.cpp            # [CPU FALLBACK] i-k-j triple loop for non-CUDA builds
│   ├── rmsnorm.cu                # [CUDA] Row-wise RMSNorm with shared-memory reduction
│   ├── rope.cu                   # [CUDA] Rotary position embeddings + host table precomputation
│   ├── attention.cu              # [CUDA] Scale, causal mask, and softmax kernels
│   ├── swiglu.cu                 # [CUDA] SwiGLU activation kernel
│   └── residual.cu               # [CUDA] In-place residual addition kernel
│
├── tests/                        # Test suite
│   ├── test.cpp                  # [READ-ONLY] Test harness with 7 tests (grading)
│   ├── test_api.h                # [READ-ONLY] TestAPI class interface
│   ├── test_api.cpp              # TestAPI implementation: tokenize, get_embeddings, matmul
│   ├── test_m2m3_main.cpp        # M2-3 test driver (dispatches by name)
│   ├── test_m2m3_helpers.h       # Shared helpers and registry type
│   ├── test_m2m3_helpers.cpp     # Tolerance comparisons and fixture I/O
│   ├── test_m2m3_matmul.cpp      # Phase 0: matmul/loader parity (5 tests)
│   ├── test_m2m3_rmsnorm_proj.cpp # Phase 1: RMSNorm + Q/K/V projections (7 tests)
│   ├── test_m2m3_rope_attn.cpp   # Phase 2: RoPE, GQA, mask, softmax (8 tests)
│   ├── test_m2m3_decoder_full.cpp # Phase 3-4: residual, SwiGLU, decoder block (7 tests)
│   ├── test_m2m3_kv_batch.cpp    # Phase 5: final norm, lm_head, KV cache, batching (11 tests)
│   └── data/                     # Binary test fixtures (M1 from upstream, M2-3 from gen_m2m3_fixtures.py)
│       ├── test2_tokenize.bin
│       ├── test3_embedding.bin
│       ├── test4_embedding.bin
│       ├── test5_matmul.bin
│       ├── test6_matmul.bin
│       ├── test7_matmul.bin
│       └── m2m3/                 # Generated golden fixtures for M2-3 phases
│
├── tools/                        # Python + shell utilities
│   ├── dumper.py                 # Safetensors -> binary dump (280-byte header + payload)
│   ├── llama3_downloader.py      # Download Llama 3 8B from HuggingFace (gated)
│   ├── gen_token_model.py        # tokenizer.json -> BPE rank file (token.model)
│   ├── gen_m2m3_fixtures.py      # Generates M2-3 golden test fixtures via NumPy forward pass
│   ├── verify_reference.py       # Compares reference.py vs our NumPy logic on real weights
│   ├── token_show.py             # Reference HF tokenizer for verification
│   ├── provision_l4.sh           # GCP L4 VM provision (SPOT/STANDARD; uses custom image if available)
│   ├── test_l4.sh                # Push source, build, run M1+M2-3 lane on L4
│   └── create_custom_image.sh    # Snapshot warm L4 boot disk into reusable image
│
├── scripts/                      # Demo + local convenience scripts
│   ├── demo-start.sh             # Auto-detect zone, start L4, drop into bin/llm --interactive
│   ├── demo-stop.sh              # Auto-detect zone, stop the L4 VM
│   └── run_tests.sh              # Local test runner
│
├── docs/                         # Documentation and reports
│   ├── index.md                  # Documentation index
│   ├── JOURNAL.md                # Development journal
│   ├── learnings.md              # Project-specific knowledge and gotchas
│   ├── walkthrough-progress.md   # Code-review walkthrough progress tracker
│   ├── RUNBOOK-L4.md             # GCP L4 provisioning + test + demo runbook
│   ├── CODEMAPS/                 # Token-lean architecture refs (architecture/kernels/data/dependencies)
│   ├── generated/                # This file lives here; auto-regenerated docs
│   ├── assignment/               # Course assignment specs (llm_part1/2 .md + .pdf)
│   ├── reports/                  # mid-checkin.md, Milestone1-Report.md, code-review-prep.md, handoff-prompt.md
│   ├── plans/                    # Implementation plans (inference-cu-split, todo2-batching, submission-wrap-up, ...)
│   ├── learn/                    # Read-aloud cue cards step1..step18 + walkthrough-summary
│   ├── tests/                    # Per-test deep-dive notes
│   ├── diagrams/                 # Mermaid sources + rendered PNGs (pipeline, residuals, key areas)
│   └── notes/                    # Working notes
│
├── assets/                       # [GITIGNORED] Model weights and tokenizer
│   └── llama3/
│       ├── token.model           # BPE vocabulary file (base64 + rank format)
│       └── dump/                 # Binary weight dumps (291 tensors)
│
├── build/                        # [GITIGNORED] Compiled object files and deps
├── bin/                          # [GITIGNORED] Output binaries (llm, tests)
└── .venv/                        # [GITIGNORED] Python virtual environment
```

## Critical Directories

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `include/` | Public C++ headers defining all interfaces | `tokenizer.h`, `loader.h`, `operator.cuh`, `model_weights.h`, `inference.h` |
| `src/` | Core implementation (tokenizer, weight loader, inference) | `tokenizer_bpe.cpp`, `loader.cpp`, `model_weights.cpp`, `device_weights.cu`, `inference*.cu` |
| `kernel/` | GPU compute layer (CUDA + CPU fallback) | `matmul.cu`, `rmsnorm.cu`, `rope.cu`, `attention.cu`, `swiglu.cu`, `residual.cu`, `matmul_cpu.cpp` |
| `tests/` | Grading test suite + M2-3 tests + student implementations | `test_api.cpp` (editable), `test.cpp` (read-only), `test_m2m3_*.cpp` |
| `tools/` | Python preprocessing and fixture generation | `dumper.py`, `llama3_downloader.py`, `gen_m2m3_fixtures.py` |

## Entry Points

| Binary | Entry File | Purpose |
|--------|-----------|---------|
| `bin/llm` | `main.cpp` | CLI inference (single-token, multi-token, interactive, batched) |
| `bin/tests` | `tests/test.cpp` | Run M1 test suite (pass test ID 1-7) |
| `bin/tests_m2m3` | `tests/test_m2m3_main.cpp` | Run M2-3 test suite (38 tests, CUDA required) |

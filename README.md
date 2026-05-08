# Llama 3 From Scratch

From-scratch Llama 3 8B Instruct inference in C++17/CUDA. No ML framework dependencies at runtime. Built on the [CS265 MLSys starter project](https://github.com/qtwang/CS265-mlsys-project).

The pipeline runs all 32 decoder layers: BPE tokenization, embedding lookup, RMSNorm, RoPE positional encoding, grouped-query attention, SwiGLU FFN, and output projection via a separate lm_head weight matrix. CUDA kernels handle matrix multiplication (double-buffered tiled GEMM), normalization, and activation functions. Building `bin/llm` requires `nvcc`; the M1 `bin/tests` target still builds CPU-only via the `matmul_cpu` fallback. A Python toolchain downloads and converts the model weights offline.

## Notes for graders

- **Required path uses FP32** (M1 grading tests via `bin/tests` 1..7, single-token inference). All 7 M1 tests pass on L4 (sm_89).
- **Multi-token path uses BF16** for resident weights so all ~14.5 GB of Llama 3 8B can stay on L4's 24 GB VRAM. FP32 residency would need 32 GB. Per the discussion-board policy on BF16, please apply the relaxed-epsilon allowance to any internal M2-3 tests whose tolerances were written for FP32 reference values.
- **Conclusive end-to-end test (`docs/assignment/llm_part2.md` §3.1 Step 6)**: `./bin/llm --max-tokens 8 "What is the capital of California?"` produces `"The capital of California is Sacramento."` (token-by-token argmax decode + EOT). Verified on L4 in this branch.
- **Live demo:** `./scripts/demo-start.sh` brings up the L4, SSHes in, and drops into an interactive REPL (`./bin/llm --interactive --max-tokens 32`). Resident BF16 weights load once (~3 min); each subsequent prompt answers in ~3s. After the demo, `./scripts/demo-stop.sh` stops the VM.
- **Extensions shipped**: KV cache + resident weights, and B>1 batched generation. Both ship with internal parity tests; see `tests/test_m2m3_kv_batch.cpp` (`batched_b2_distinct_parity`, etc.).

## Quick start

```bash
# Build (requires nvcc for bin/llm; bin/tests builds CPU-only)
make                    # release build → bin/llm
make tests              # M1 test binary → bin/tests
make tests_m2m3         # M2-3 test binary → bin/tests_m2m3 (CUDA required)

# Run a test (requires model assets)
./bin/tests 1

# Single-token inference (CUDA required)
./bin/llm "The capital of France is"
```

## Prerequisites

- C++17 compiler and Python 3.10+ (macOS or Linux)
- CUDA toolkit with `nvcc` for GPU builds
- Hugging Face account with [Llama 3 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) access approved
- HF token (generate at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)), placed in `.env` as `HUGGINGFACE_TOKEN=hf_...`

## Build commands

```bash
make                    # release build (-O2) → bin/llm
make BUILD=debug        # debug build (-g -O0)
make tests              # M1 test binary → bin/tests
make tests_m2m3         # M2-3 test binary → bin/tests_m2m3 (CUDA required)
make run                # build and run bin/llm
make clean              # remove build/ and bin/
```

## L4 test lanes

CUDA verification runs on the configured GCP L4 VM through `tools/test_l4.sh`.
The default lane is intentionally quick; the full lane is a final gate.

```bash
./tools/test_l4.sh          # quick lane: M1 + fast M2-3
./tools/test_l4.sh --unit   # M1 only
./tools/test_l4.sh --perf   # KV-cache performance/audit
./tools/test_l4.sh --full   # final full regression gate
```

The `--perf` lane runs the resident 8-token KV-cache path and fails if the log
shows old per-layer streaming timers.

## GCP GPU VM setup

The full L4 provisioning + test + demo workflow is automated. See [`docs/RUNBOOK-L4.md`](./docs/RUNBOOK-L4.md) for details.

```bash
cp .l4-config.env.example .l4-config.env       # set VM_NAME, PREFERRED_ZONE, etc.
echo "HUGGINGFACE_TOKEN=hf_..." > .env

./tools/provision_l4.sh                        # create VM, install drivers, download weights, run dumper
./tools/test_l4.sh --quick --no-stop           # rsync source, build, run quick test lane, leave VM up
./tools/create_custom_image.sh                 # snapshot the warm disk for ~60s future boots (optional)
```

After capturing the custom image, future `provision_l4.sh` runs boot in ~60s with everything pre-baked. The demo wrapper scripts auto-detect the VM zone:

```bash
./scripts/demo-start.sh 32                     # start VM, SSH in, drop into bin/llm --interactive --max-tokens 32
./scripts/demo-stop.sh                         # stop the VM after the demo
```

## Test results

All 7 M1 tests and 38 M2-3 tests pass on a GCP L4 (g2-standard-4, sm_89). CLI inference produces correct tokens validated against PyTorch.

### Milestone 1 (tokenizer, embeddings, matmul)

| Test | Description | Result |
|------|-------------|--------|
| `./bin/tests 1` | Tokenize "Hello world" → `[128000, 9906, 1917]` | PASSED |
| `./bin/tests 2` | Tokenize long sentence (binary fixture) | PASSED |
| `./bin/tests 3` | Embedding lookup (fixture 1) | PASSED |
| `./bin/tests 4` | Embedding lookup (fixture 2) | PASSED |
| `./bin/tests 5` | Matmul, seq_len=1 | PASSED |
| `./bin/tests 6` | Matmul, seq_len=10 | PASSED |
| `./bin/tests 7` | Matmul, seq_len=100 | PASSED |

### Milestones 2-3 (CUDA kernels, full inference)

38 tests covering RMSNorm, RoPE, attention (scale, causal mask, softmax, GQA), SwiGLU, residual add, BF16-weight matmul parity, B>1 batched parity, KV-cache bounds, single-layer forward pass, and full 32-layer inference. Run with:

```bash
./bin/tests_m2m3 --list   # list all test names
./bin/tests_m2m3 <name>   # run a specific test
```

## Numerical validation against the grading reference

`reference.py` is the PyTorch forward pass the TAs use to grade this project (pulled from the upstream repo). `tools/verify_reference.py` runs both `reference.py` and the NumPy logic from `gen_m2m3_fixtures.py` on the real Llama-3 weights and compares per-operator outputs (RMSNorm, QKV, RoPE, attention, FFN, final hidden, logits, argmax). When they agree within tolerance, our CUDA kernels transitively match the grading reference.

```bash
python3 tools/verify_reference.py --prompt hello    # <BOS> Hello world         -> next token 0
python3 tools/verify_reference.py --prompt medium   # <BOS> The capital of France is -> next token 12366
```

Both prompts agree with `reference.py` to a per-op max abs diff < 1e-5 in layer 0 and produce the same argmax token.

## Project structure

```
main.cpp                 # CLI entry point (single-token / multi-token / interactive / batched)
config.h                 # Llama 3 8B architecture constants
reference.py             # PyTorch grading reference (used by TAs)
include/
  prelude.h              # Common type aliases and STL imports
  tokenizer.h            # BPETokenizer interface
  loader.h               # LlamaDumpLoader (binary dump reader)
  milifloat.h            # BF16/FP16 → FP32 converters
  model_weights.h        # Per-layer and global weight management
  device_weights.h       # GPU-resident weight manager (BF16 device buffers)
  inference.h            # Forward-pass entry points
  kv_cache.h             # Device-side per-layer K/V buffers (multi-token decode)
  instrument.h           # Header-only Stopwatch + probe_vram telemetry
  operator.cuh           # AbstractOperator base class (scaffold)
src/
  tokenizer_bpe.cpp      # BPE tokenizer (encode/decode with special tokens)
  loader.cpp             # Weight loader (280-byte header + BF16/FP16/FP32 payload)
  model_weights.cpp      # Weight loading with transpose-at-load
  device_weights.cu      # GPU-resident BF16 weight upload/free
  inference.cu           # Public API facade (forward_pass entry points)
  inference_internal.h   # Cross-TU private declarations for inference_*.cu
  inference_chat.cu      # Chat template formatting
  inference_layer.cu     # Single decoder block (attention + FFN)
  inference_loop.cu      # Autoregressive decode orchestrator (single / KV / batched)
  kv_cache.cu            # cudaMalloc/cudaFree per-layer K/V buffers
kernel/
  kernels.cuh            # Host-callable kernel entry points
  matmul.cu              # Tiled GEMM (double-buffered shared memory, float4 loads, BF16-weight variant)
  matmul_cpu.cpp         # CPU fallback for non-CUDA builds
  rmsnorm.cu             # Row-wise RMSNorm with shared-memory reduction
  rope.cu                # Rotary position embeddings (rotate_full convention)
  attention.cu           # Scale, causal mask, numerically stable softmax
  swiglu.cu              # SwiGLU activation (SiLU(gate) * up)
  residual.cu            # In-place residual addition
tests/
  test.cpp               # M1 test harness (7 tests, read-only)
  test_api.h             # TestAPI interface (read-only)
  test_api.cpp           # TestAPI implementation (tokenize, embed, matmul)
  test_m2m3_main.cpp     # M2-3 test driver (dispatches by name)
  test_m2m3_helpers.h    # Shared helpers and registry type
  test_m2m3_helpers.cpp  # Tolerance comparisons and fixture I/O
  test_m2m3_matmul.cpp   # Phase 0: matmul/loader parity tests
  test_m2m3_rmsnorm_proj.cpp  # Phase 1: RMSNorm + Q/K/V projections
  test_m2m3_rope_attn.cpp     # Phase 2: RoPE, GQA, causal mask, softmax
  test_m2m3_decoder_full.cpp  # Phase 3-4: residual, SwiGLU, full decoder block
  test_m2m3_kv_batch.cpp      # Phase 5: final norm, lm_head, KV cache, B>1 batching
tools/
  llama3_downloader.py   # Download weights from Hugging Face
  dumper.py              # Safetensors → binary dump (280-byte header + payload)
  gen_token_model.py     # tokenizer.json → BPE rank file (token.model)
  gen_m2m3_fixtures.py   # Generate golden test fixtures via NumPy
  verify_reference.py    # Compare reference.py vs our NumPy logic on real weights
  token_show.py          # Token inspection utility
  provision_l4.sh        # GCP L4 VM provision (SPOT/STANDARD; uses custom image if available)
  test_l4.sh             # Push source, build, run M1+M2-3 lane on L4
  create_custom_image.sh # Snapshot warm L4 boot disk for fast (~60s) re-provision
scripts/
  demo-start.sh          # Auto-detect zone, start L4, SSH into bin/llm --interactive --max-tokens N
  demo-stop.sh           # Auto-detect zone, stop the L4 VM
  run_tests.sh           # Local test runner
```

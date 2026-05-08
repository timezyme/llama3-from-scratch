# Development Guide

**Generated:** 2026-03-29 | **Scan Level:** Exhaustive

## Prerequisites

| Requirement | Details |
|------------|---------|
| **C++ Compiler** | g++ with C++17 support |
| **CUDA Toolkit** | nvcc (required for bin/llm; bin/tests builds CPU-only) |
| **Python 3** | For tooling (dumper, downloader) |
| **GNU Make** | Build system |
| **GCP Account** | For GPU testing on L4 (`g2-standard-4`, sm_89) |
| **HuggingFace Account** | For downloading Llama 3 8B (gated model) |

## Local Development Setup

### 1. Clone the Repository

```bash
git clone <repo-url>
cd CS265-llm-starter-main
```

### 2. Python Environment (for tooling only)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install huggingface_hub safetensors numpy torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Download Model Weights

```bash
source .venv/bin/activate
HF_TOKEN=<your-token> python tools/llama3_downloader.py --out ./assets/llama3/
```

### 4. Generate Token Model and Dump Weights

```bash
python tools/gen_token_model.py    # tokenizer.json -> assets/llama3/token.model
python tools/dumper.py             # safetensors -> assets/llama3/dump/ (291 tensors)
```

## Build Commands

| Command | Description | Output |
|---------|-------------|--------|
| `make` | Release build (`-O2`) | `bin/llm` |
| `make BUILD=debug` | Debug build (`-g -O0`) | `bin/llm` |
| `make tests` | Build M1 test binary | `bin/tests` |
| `make tests_m2m3` | Build M2-3 test binary (CUDA required) | `bin/tests_m2m3` |
| `make run` | Build and run main binary | — |
| `make clean` | Remove `build/` and `bin/` | — |

The Makefile auto-detects `nvcc`. Without it, `bin/llm` refuses to link; `bin/tests` still builds, using the CPU matmul fallback (`kernel/matmul_cpu.cpp`) in place of the CUDA kernel.

## Running Tests

### M1 Tests

```bash
# Build and run a single test
make tests
./bin/tests 1    # Test ID 1-7

# Run all tests
for i in 1 2 3 4 5 6 7; do ./bin/tests $i; done
```

#### Test Descriptions

| ID | What It Tests | Tolerance |
|----|--------------|-----------|
| 1 | Tokenize sentence → verify token count | Exact match |
| 2 | Tokenize sentence → compare with binary reference | Exact match |
| 3 | Embeddings for first 5 tokens → compare with reference | `EPSILON = 1e-2` |
| 4 | Embeddings for last 5 tokens → compare with reference | `EPSILON = 1e-2` |
| 5 | Matrix multiply (small) → compare with reference | `EPSILON = 1e-2` |
| 6 | Matrix multiply (medium) → compare with reference | `EPSILON = 1e-2` |
| 7 | Matrix multiply (large) → compare with reference | `EPSILON = 1e-2` |

**Note:** Tests require model weights at `./assets/llama3/` and dumped weights at `./assets/llama3/dump/`. Tests 5-7 require CUDA (or CPU fallback) for matrix multiplication.

### M2-3 Tests (CUDA required)

```bash
make tests_m2m3
./bin/tests_m2m3 --list   # list all test names
./bin/tests_m2m3 <name>   # run a specific test

# Run all M2-3 tests
for t in $(./bin/tests_m2m3 --list); do ./bin/tests_m2m3 $t; done
```

38 tests covering RMSNorm, RoPE, attention (scale, causal mask, softmax, GQA), SwiGLU, residual add, BF16-weight matmul parity, B>1 batched parity, KV-cache bounds, single-layer forward pass, and full 32-layer inference.

## GCP GPU Testing

CUDA kernels are tested on a GCP L4 VM. See [`RUNBOOK-L4.md`](RUNBOOK-L4.md) for the full provisioning workflow (including the custom-image fast-boot path), recurring test lanes, demo lifecycle, and tear-down.

Use the lanes intentionally:

```bash
./tools/test_l4.sh          # quick lane: M1 + fast M2-3 (default dev loop)
./tools/test_l4.sh --unit   # M1 only
./tools/test_l4.sh --perf   # KV-cache performance/audit
./tools/test_l4.sh --full   # final full regression gate
./tools/test_l4.sh --no-stop  # leave VM running for follow-up work
```

The full lane is not the development loop; it includes expensive full-forward and loader lifecycle checks. For the code-review demo, use `./scripts/demo-start.sh` and `./scripts/demo-stop.sh` instead of raw gcloud — both auto-detect the VM zone.

## Adding New Source Files

When adding new `.cpp` or `.cu` files, update these Makefile variables:

1. `SOURCES` — add the source file path
2. `OBJECTS` — add the corresponding `.o` in `$(BUILD_DIR)/`
3. `TEST_OBJECTS` — add to test build if needed for tests
4. `CUDA_KERNEL_OBJECTS` — add if the file is a CUDA kernel used by the main binary
5. `M2M3_KERNEL_OBJECTS` — add if the file is a CUDA kernel used by the M2-3 test binary

Add a compile rule following the existing pattern:
```makefile
$(BUILD_DIR)/newfile.o: src/newfile.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
```

## Important Constraints

- `tests/test.cpp` and `tests/test_api.h` are **read-only** (grading depends on them)
- Never overwrite local implementations with upstream stubs (upstream `test_api.cpp` has empty stubs)
- Model weights at `./assets/llama3/` are gitignored
- The upstream repo is at `https://code.harvard.edu/mir593/CS265-llm-starter.git` — cherry-pick only `test.cpp`, `test_api.h`, and `tests/data/` from it

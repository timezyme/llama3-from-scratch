# Llama 3 From Scratch

From-scratch Llama 3 8B Instruct inference in C++17/CUDA. No ML framework dependencies at runtime.

The pipeline runs all 32 decoder layers: BPE tokenization, embedding lookup, RMSNorm, RoPE positional encoding, grouped-query attention, SwiGLU FFN, and output projection via a separate lm_head weight matrix. CUDA kernels handle matrix multiplication (double-buffered tiled GEMM), normalization, and activation functions. A CPU matmul fallback builds when `nvcc` is unavailable. A Python toolchain downloads and converts the model weights offline.

## Guided tour

A 14-step visual walkthrough of the codebase lives in [`docs/presentation/`](docs/presentation/index.html). Open `index.html` in a browser and use the arrow keys to navigate.

## Quick start

```bash
# Build (CPU-only if nvcc is not found)
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

## GCP GPU VM setup

You need a GPU to run the CUDA kernels. These instructions use a GCP spot instance with an NVIDIA T4 (~$0.16/hr).

### Create the VM

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

gcloud compute instances create llama3-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --image-family=common-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-balanced \
  --metadata="install-nvidia-driver=True" \
  --scopes=default
```

### First-time VM setup

```bash
gcloud compute ssh llama3-gpu --zone=us-central1-a

# On the VM:
sudo apt-get update -qq && sudo apt-get install -y -qq build-essential python3-pip
pip3 install huggingface_hub safetensors numpy torch --index-url https://download.pytorch.org/whl/cpu

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Start / stop VM

```bash
gcloud compute instances start llama3-gpu --zone=us-central1-a
gcloud compute instances stop llama3-gpu --zone=us-central1-a    # ~$0.005/hr disk only
gcloud compute instances delete llama3-gpu --zone=us-central1-a  # zero cost
```

### Copy project to the VM

```bash
gcloud compute scp --recurse --zone=us-central1-a . llama3-gpu:~/llama3
gcloud compute scp --zone=us-central1-a src/loader.cpp llama3-gpu:~/llama3/src/loader.cpp
```

## Model download and weight dumping

### Download Llama 3 8B (~90 seconds on VM)

```bash
cd ~/llama3
HF_TOKEN=<your-token> python3 tools/llama3_downloader.py --out ./assets/llama3/
```

### Generate token.model

The Hugging Face download gives you `tokenizer.json` (GPT-style byte encoding), but the C++ tokenizer expects a raw-byte rank file. This script converts it:

```bash
python3 -c "
import json, base64

with open('assets/llama3/tokenizer.json') as f:
    data = json.load(f)
vocab = data['model']['vocab']

def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('¡'), ord('¬')+1)) + list(range(ord('®'), ord('ÿ')+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, cs))

b2u = bytes_to_unicode()
u2b = {chr(v): bytes([k]) for k, v in b2u.items()}

def token_to_bytes(token_str):
    raw = b''
    for ch in token_str:
        if ch in u2b:
            raw += u2b[ch]
        else:
            raw += ch.encode('utf-8')
    return raw

with open('assets/llama3/token.model', 'w') as out:
    for token_str, rank in sorted(vocab.items(), key=lambda x: x[1]):
        b64 = base64.b64encode(token_to_bytes(token_str)).decode('ascii')
        out.write(f'{b64} {rank}\n')

print(f'Wrote {len(vocab)} entries')
"
```

### Dump weights

```bash
python3 tools/dumper.py
# Outputs 291 tensors to assets/llama3/dump/
# Embeddings: assets/llama3/dump/embeddings.bin
# Layers:     assets/llama3/dump/layer_XX/
# Global:     assets/llama3/dump/global/
# Manifest:   assets/llama3/dump/manifest.json
```

## Building and testing on the VM

```bash
gcloud compute ssh llama3-gpu --zone=us-central1-a
cd ~/llama3
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

make clean && make && make tests && make tests_m2m3

# Run M1 tests
for i in 1 2 3 4 5 6 7; do ./bin/tests $i; done

# Run M2-3 tests
for t in $(./bin/tests_m2m3 --list); do ./bin/tests_m2m3 $t; done

# Single-token inference
./bin/llm "The capital of France is"
```

## Test results

All 7 M1 tests and 27 M2-3 tests pass on a GCP T4 (n1-standard-4). CLI inference produces correct tokens validated against PyTorch.

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

27 tests covering RMSNorm, RoPE, attention (scale, causal mask, softmax, GQA), SwiGLU, residual add, single-layer forward pass, and full 32-layer inference. Run with:

```bash
./bin/tests_m2m3 --list   # list all test names
./bin/tests_m2m3 <name>   # run a specific test
```

## Project structure

```
main.cpp                 # CLI entry point (single-token inference)
config.h                 # Llama 3 8B architecture constants
include/
  prelude.h              # Common type aliases and STL imports
  tokenizer.h            # BPETokenizer interface
  loader.h               # LlamaDumpLoader (binary dump reader)
  milifloat.h            # BF16/FP16 → FP32 converters
  model_weights.h        # Per-layer and global weight management
  inference.h            # Forward-pass entry points
  operator.cuh           # AbstractOperator base class (scaffold)
src/
  tokenizer_bpe.cpp      # BPE tokenizer (encode/decode with special tokens)
  loader.cpp             # Weight loader (280-byte header + BF16/FP16/FP32 payload)
  model_weights.cpp      # Weight loading with transpose-at-load
  inference.cu           # 32-layer forward pass with chat template
kernel/
  kernels.cuh            # Host-callable kernel entry points
  matmul.cu              # Tiled GEMM (double-buffered shared memory, float4 loads)
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
  test_m2m3.cpp          # M2-3 test harness (27 tests, CUDA required)
tools/
  llama3_downloader.py   # Download weights from Hugging Face
  dumper.py              # Safetensors → binary dump (280-byte header + payload)
  gen_m2m3_fixtures.py   # Generate golden test fixtures via NumPy
  token_show.py          # Token inspection utility
docs/
  presentation/          # Interactive guided tour of the codebase
  learnings.md           # Project-specific knowledge and gotchas
  Milestone1-Report.pdf  # Project report
```

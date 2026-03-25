# CS265 LLM Inference Project

From-scratch Llama 3 8B inference in C++17/CUDA. No ML framework dependencies at runtime.

**Milestone 1: Complete** — all 7 tests passing on GCP T4.

## Prerequisites

- **Local (macOS M4)**: Xcode command line tools (`xcode-select --install`), Python 3.10+
- **GPU testing**: GCP account with billing enabled, `gcloud` CLI installed
- **Model access**: Hugging Face account with Llama 3 8B Instruct access approved at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- **HF token**: Generate at https://huggingface.co/settings/tokens, place in `.env` as `HUGGINGFACE_TOKEN=hf_...`

## Local setup (macOS, no GPU)

```bash
# Install Python deps
pip install -r requirements.txt

# Build (CPU-only, auto-detects no nvcc)
make                    # release build → bin/llm
make BUILD=debug        # debug build
make tests              # test binary → bin/tests
make clean              # remove build/ and bin/

# Run tokenizer test (requires assets/llama3/token.model)
./bin/tests 1
```

The Makefile auto-detects `nvcc`. Without it, the CPU matmul fallback (`kernel/matmul_cpu.cpp`) is used.

## GCP GPU VM

### Auth and project setup

```bash
# Authenticate
gcloud auth login

# Set project (must have billing enabled)
gcloud config set project timezyme-document-processing
```

### Create VM (first time)

```bash
gcloud compute instances create cs265-gpu-test \
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

**Cost**: ~$0.16/hr (spot pricing). Stop when not in use.

### First-time VM setup

```bash
# SSH in
gcloud compute ssh cs265-gpu-test --zone=us-central1-a

# On the VM:
sudo apt-get update -qq && sudo apt-get install -y -qq build-essential
sudo apt-get install -y -qq python3-pip
pip3 install huggingface_hub safetensors numpy torch --index-url https://download.pytorch.org/whl/cpu

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Start / stop VM

```bash
# Start
gcloud compute instances start cs265-gpu-test --zone=us-central1-a

# Stop (preserves disk, ~$0.005/hr for disk only)
gcloud compute instances stop cs265-gpu-test --zone=us-central1-a

# Delete (zero cost, must recreate from scratch)
gcloud compute instances delete cs265-gpu-test --zone=us-central1-a
```

### Copy project to VM

```bash
# Full project (run from project root)
gcloud compute scp --recurse --zone=us-central1-a \
  . cs265-gpu-test:~/CS265

# Single file update
gcloud compute scp --zone=us-central1-a \
  src/loader.cpp cs265-gpu-test:~/CS265/src/loader.cpp
```

## Model download and weight dumping

### Download Llama 3 8B (on VM, ~90 seconds)

```bash
gcloud compute ssh cs265-gpu-test --zone=us-central1-a
cd ~/CS265
HF_TOKEN=<your-token> python3 tools/llama3_downloader.py --out ./assets/llama3/
```

### Generate token.model

The HF download gives you `tokenizer.json` (GPT-style byte encoding), but the C++ tokenizer needs a raw-byte rank file. Convert it:

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
# Layers: assets/llama3/dump/layer_XX/
# Global: assets/llama3/dump/global/
# Manifest: assets/llama3/dump/manifest.json
```

## Building and testing on VM

```bash
gcloud compute ssh cs265-gpu-test --zone=us-central1-a
cd ~/CS265
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Build
make clean && make tests

# Run grading test
./bin/tests 1    # tokenize "Hello world" → [128000, 9906, 1917]

# Build main binary
make
./bin/llm
```

## Test results (Milestone 1)

All 7 passing on GCP T4 (`n1-standard-4`).

| Test | Description | Result |
|------|-------------|--------|
| `./bin/tests 1` | Tokenize "Hello world" → `[128000, 9906, 1917]` | PASSED |
| `./bin/tests 2` | Tokenize long sentence (binary fixture) | PASSED |
| `./bin/tests 3` | Embedding lookup (fixture 1) | PASSED |
| `./bin/tests 4` | Embedding lookup (fixture 2) | PASSED |
| `./bin/tests 5` | Matmul, seq_len=1 | PASSED |
| `./bin/tests 6` | Matmul, seq_len=10 | PASSED |
| `./bin/tests 7` | Matmul, seq_len=100 | PASSED |

## Project structure

```
src/tokenizer_bpe.cpp    # BPE tokenizer (encode/decode)
src/loader.cpp           # Weight loader (mmap, BF16→FP32)
tools/dumper.py          # Safetensors → binary dump (Python)
kernel/matmul.cu         # CUDA tiled GEMM kernel
kernel/matmul_cpu.cpp    # CPU fallback (no GPU builds)
tests/test_api.cpp       # TestAPI implementation
tests/test.cpp           # Test harness (DO NOT MODIFY)
tests/test_api.h         # Test API header (DO NOT MODIFY)
include/loader.h         # LlamaDumpLoader declarations
include/tokenizer.h      # BPETokenizer declarations
include/milifloat.h      # BF16/FP16 → FP32 converters
include/operator.cuh     # AbstractOperator base class
kernel/kernels.cuh       # CUDA kernel signatures
config.h                 # Constants (paths, dims, epsilon)
```

## Read-only files (grading)

Don't modify: `tests/test.cpp`, `tests/test_api.h`, `tools/llama3_downloader.py`, `tools/token_show.py`

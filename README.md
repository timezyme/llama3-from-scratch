# Llama 3 From Scratch

From-scratch Llama 3 8B inference in C++17/CUDA. No ML framework dependencies at runtime.

The pipeline covers tokenization (BPE with greedy merge), weight loading (BF16/FP16/FP32 via mmap), and matrix multiplication (double-buffered tiled GEMM on CUDA, with a CPU fallback). A Python toolchain handles downloading and converting the model weights offline.

## Guided tour

There's a 14-step visual walkthrough of the codebase in [`docs/presentation/`](docs/presentation/index.html). Open `index.html` in a browser and use the arrow keys to navigate.

## Quick start

```bash
# Build (CPU-only if nvcc is not found)
make                    # release build → bin/llm
make tests              # test binary → bin/tests

# Run a test (requires model assets)
./bin/tests 1
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
make tests              # test binary → bin/tests
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

make clean && make tests
./bin/tests 1    # tokenize "Hello world" → [128000, 9906, 1917]

make
./bin/llm
```

## Test results

All 7 tests pass on a GCP T4 (n1-standard-4).

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
main.cpp                 # Entry point
config.h                 # Constants (paths, dims, epsilon)
include/
  prelude.h              # Common type aliases and STL imports
  tokenizer.h            # BPETokenizer interface
  loader.h               # LlamaDumpLoader declarations
  milifloat.h            # BF16/FP16 → FP32 converters
  operator.cuh           # AbstractOperator base class
src/
  tokenizer_bpe.cpp      # BPE tokenizer (encode/decode)
  loader.cpp             # Weight loader (mmap, BF16→FP32)
kernel/
  kernels.cuh            # CUDA kernel signatures
  matmul.cu              # Tiled GEMM kernel (double-buffered, shared memory)
  matmul_cpu.cpp         # CPU fallback for non-CUDA builds
tests/
  test.cpp               # Test harness (7 tests)
  test_api.h             # TestAPI interface
  test_api.cpp           # TestAPI implementation (tokenize, embed, matmul)
tools/
  llama3_downloader.py   # Download weights from Hugging Face
  dumper.py              # Safetensors → binary dump
  token_show.py          # Token inspection utility
docs/
  presentation/          # Interactive guided tour of the codebase
  Milestone1-Report.pdf  # Project report
```

<!-- Generated: 2026-05-02 | Updated: 2026-05-05 | Files scanned: Makefile + tools/ + scripts/ | Token estimate: ~450 -->

# Dependencies

## Runtime (C++)
| Dependency      | Purpose                    | Notes                          |
|-----------------|----------------------------|--------------------------------|
| CUDA Runtime    | GPU kernel execution       | Linked via -lcudart            |
| C++17 STL       | Containers, strings, I/O   | No external C++ libs           |

No ML frameworks at runtime. Zero external C++ dependencies beyond CUDA.

## Build
| Tool    | Version     | Purpose                                |
|---------|-------------|----------------------------------------|
| g++     | C++17       | Host code compilation                  |
| nvcc    | CUDA 12.8 (T4) / 12.9 (L4) | Kernel compilation (optional) |
| Make    | GNU         | Build orchestration                    |

Makefile vars: `ARCH ?= sm_75` (T4 default), `NVCCFLAGS ?= -std=c++17 -O2`,
`BUILD ?= release`. Override arch with `make ARCH=sm_89` for L4.

## Python Tooling (offline preprocessing only)
| Package          | Purpose                                |
|------------------|----------------------------------------|
| huggingface_hub  | Download Llama 3 8B from HuggingFace   |
| safetensors      | Read safetensors weight format         |
| numpy            | Fixture generation (gen_m2m3_fixtures) |
| torch (CPU)      | Dtype conversion in dumper.py          |
| transformers     | Reference tokenizer (token_show.py)    |

Installed in `.venv/`. Not needed at inference time.

### Python Tools
| Script                       | Purpose                                       |
|------------------------------|-----------------------------------------------|
| tools/llama3_downloader.py   | Pull safetensors from HuggingFace             |
| tools/dumper.py              | safetensors -> 280-byte-header binary dump    |
| tools/gen_token_model.py     | tokenizer.json -> BPE rank file (token.model) |
| tools/gen_m2m3_fixtures.py   | NumPy reference forward pass -> M2-3 fixtures |
| tools/verify_reference.py    | Compare reference.py vs C++ numerics          |
| tools/token_show.py          | Inspect tokenizer output                      |

## Shell Tooling
| Script                       | Purpose                                                            |
|------------------------------|--------------------------------------------------------------------|
| tools/provision_l4.sh        | L4 VM provision (SPOT or STANDARD; uses custom image if available) |
| tools/test_l4.sh             | Push source, build, run M1+M2-3 lane, optionally stop VM           |
| tools/create_custom_image.sh | Stop VM and snapshot the warm boot disk into a reusable image      |
| scripts/demo-start.sh        | Auto-detect zone, start VM, SSH into `bin/llm --interactive`       |
| scripts/demo-stop.sh         | Auto-detect zone, stop the VM after the demo                       |
| scripts/run_tests.sh         | Local test runner                                                  |

`PROVISIONING_MODEL` in `.l4-config.env` selects SPOT (default, ~$0.20–0.30/hr,
preemptible) vs STANDARD (~$0.72/hr, no preemption). After a successful warm
provision + build, `create_custom_image.sh` captures the disk so future
`provision_l4.sh` runs boot in ~60s. The image lives in the source disk's region;
keep `PREFERRED_ZONE` in that region to avoid cross-region image copy.

## External Services
| Service     | Purpose                     | Auth                |
|-------------|-----------------------------|---------------------|
| HuggingFace | Model weight download       | HF_TOKEN in .env    |
| GCP Compute | GPU testing (L4 spot)       | gcloud CLI          |
| Harvard Git | Upstream test fixtures      | GITHUB_TOKEN in .env|

## Target Hardware (current: L4)
| Property   | Value                                       |
|------------|---------------------------------------------|
| GPU        | NVIDIA L4 (Ada Lovelace, sm_89)             |
| VRAM       | 24 GB GDDR6                                 |
| Bandwidth  | 300 GB/s (lower than T4's 320 GB/s)         |
| Tensor cores | BF16/FP16/INT8/FP8 (sm_89)                |
| Instance   | GCP g2-standard-4, SPOT or STANDARD         |
| Cost       | SPOT ~$0.20–0.30/hr, STANDARD ~$0.72/hr     |
| Image      | common-cu129-ubuntu-2204-nvidia-580 (base); custom image after first warm provision |
| nvcc path  | /usr/local/cuda-12.9/bin/nvcc (no symlink)  |

L4 was selected over T4 because sm_89 has BF16 tensor cores (T4/sm_75 does not)
and 24 GB VRAM enables resident BF16 weights + batching. As of 2026-05-02,
resident BF16 weights are wired in production: the KV-cache perf gate is closed
(`docs/JOURNAL.md`), per-token decode runs on cached weights, and L4's BF16
advantage is realized. WMMA / tensor-core matmul (Phase D, TODO #3) is still
open. See `docs/RUNBOOK-L4.md` and `docs/learnings.md` for setup and gotchas.

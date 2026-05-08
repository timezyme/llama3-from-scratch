# Project Overview — CS265 LLM Inference Engine

**Generated:** 2026-03-29 | **Updated:** 2026-05-08 | **Scan Level:** Exhaustive

## Executive Summary

A from-scratch C++17 implementation of Llama 3 8B inference, built as a Harvard CS265 course project. The system implements the complete inference pipeline — BPE tokenization, binary weight loading with BF16/FP16/FP32 conversion, and GPU-accelerated matrix multiplication via custom CUDA kernels — without relying on ML framework dependencies at runtime.

**Milestones 1-3 are complete:** all 7 M1 tests and 38 M2-3 tests pass on a GCP L4 GPU instance (sm_89). CLI inference produces correct tokens validated against PyTorch. Bonus credit landed: TODO #1 (KV cache + resident BF16 weights) and TODO #2 (B>1 batching).

## Quick Reference

| Property | Value |
|----------|-------|
| **Language** | C++17, CUDA, Python (tooling) |
| **Build System** | GNU Make with nvcc auto-detection |
| **Target Hardware** | NVIDIA L4 GPU (GCP `g2-standard-4`, sm_89) |
| **Model** | Meta Llama 3 8B Instruct |
| **Architecture** | Pipeline-based inference engine |
| **Repository Type** | Monolith |
| **Entry Point** | `main.cpp` |
| **Test Binaries** | `bin/tests` (7 M1 tests), `bin/tests_m2m3` (38 M2-3 tests) |

## Technology Stack

| Category | Technology | Version / Details |
|----------|-----------|-------------------|
| Language | C++ | C++17 (`-std=c++17`) |
| GPU Compute | CUDA | nvcc (auto-detected) |
| Compiler | g++ | With `-Wall -Wextra -pedantic` |
| Build | GNU Make | Makefile with conditional CUDA/CPU targets |
| Model Format | Safetensors | Via Python dumper → custom binary format |
| Python Tooling | huggingface_hub, safetensors, numpy, torch (CPU) | In `.venv/` |
| Cloud | GCP Compute Engine | g2-standard-4 + L4 GPU, SPOT or STANDARD |

## Architecture Summary

```
Prompt -> Chat Template -> BPE Tokenizer (encode) -> Token IDs
    |
    v
Embedding Lookup (Weight Loader, BF16->FP32)
    |
    v
32x Decoder Layers:
    +-- RMSNorm -> Q/K/V Projection -> RoPE -> GQA Attention -> O Projection -> Residual Add
    +-- RMSNorm -> Gate/Up Projection -> SwiGLU Activation -> Down Projection -> Residual Add
    |
    v
Final RMSNorm -> lm_head (output projection) -> Argmax
    |
    v
BPE Tokenizer (decode) -> Output Text
```

## Key Dimensions

| Constant | Value | Defined In |
|----------|-------|------------|
| `EMBEDDING_DIM` | 4096 | `config.h` |
| `NUM_HEADS` | 32 | `config.h` |
| `NUM_KV_HEADS` | 8 | `config.h` |
| `HEAD_DIM` | 128 | `config.h` |
| `FFN_DIM` | 14336 | `config.h` |
| `VOCAB_SIZE` | 128256 | `config.h` |
| `NUM_LAYERS` | 32 | `config.h` |
| `RMS_NORM_EPSILON` | 1e-5f | `config.h` |
| `ROPE_BASE` | 500000 | `config.h` |
| Model Tensors | 291 | Dumped by `tools/dumper.py` |

## Current Status

- **Milestone 1**: Complete — BPE tokenizer, binary weight loader, CUDA tiled GEMM matmul, all 7 tests passing
- **Milestones 2-3**: Complete — full 32-layer forward pass, CUDA kernels (RMSNorm, RoPE, GQA attention, SwiGLU, residual, BF16-weight matmul), model weight management with transpose-at-load + resident BF16 path, CLI inference with Llama 3 Instruct chat template, all 38 M2-3 tests passing
- **Bonus credit shipped**: TODO #1 (KV cache + resident BF16 weights, +5%) and TODO #2 (B>1 batching, +5%); both ship with internal parity tests

## Links

- [CODEMAPS/](./CODEMAPS/) — Token-lean architecture reference
- [Source Tree Analysis](./source-tree-analysis.md)
- [Development Guide](./development-guide.md)
- [CUDA Kernel Documentation](./hardware-documentation.md)

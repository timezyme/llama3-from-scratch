#!/usr/bin/env python3
"""Verify reference.py matches gen_m2m3_fixtures.py numerically.

Runs the same forward pass using:
  1. reference.py (PyTorch)             - the grading reference
  2. gen_m2m3_fixtures.py logic (NumPy) - the project's CUDA-matched path

Compares per-operator outputs (RMSNorm, QKV, RoPE, attention, FFN,
final hidden, logits, argmax). If they agree within tolerance, the
CUDA kernels transitively match the grading reference.

Usage:
    python3 tools/verify_reference.py [--prompt hello|medium]
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import reference as ref  # noqa: E402

DUMP_DIR = REPO_ROOT / "assets" / "llama3" / "dump"

PROMPTS = {
    "hello":  [128000, 9906, 1917],                   # <BOS> Hello world
    "medium": [128000, 791, 6864, 315, 9822, 374],    # <BOS> The capital of France is
}

# Known-good next tokens produced by the NumPy reference (matches CUDA output).
EXPECTED_NEXT_TOKEN = {
    "hello":  0,
    "medium": 12366,
}


def read_dump(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        data = f.read()
    dtype_code = struct.unpack_from("<I", data, 256)[0]
    ndims = struct.unpack_from("<I", data, 260)[0]
    shape0 = struct.unpack_from("<Q", data, 264)[0]
    shape1 = struct.unpack_from("<Q", data, 272)[0]
    payload = data[280:]
    if dtype_code == 0:
        arr = np.frombuffer(payload, dtype=np.float32)
    elif dtype_code == 1:
        arr = np.frombuffer(payload, dtype=np.float16).astype(np.float32)
    elif dtype_code == 2:
        raw = np.frombuffer(payload, dtype=np.uint16)
        arr = np.frombuffer((raw.astype(np.uint32) << 16).tobytes(),
                            dtype=np.float32)
    else:
        raise ValueError(f"unknown dtype_code {dtype_code}")
    return arr.reshape(shape0) if ndims == 1 else arr.reshape(shape0, shape1)


def to_t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x).copy()).to(torch.float32)


def load_layer_weights(layer_idx: int) -> dict:
    prefix = f"model_layers_{layer_idx}_"
    layer_dir = DUMP_DIR / f"layer_{layer_idx:02d}"

    def load(name: str) -> np.ndarray:
        return read_dump(layer_dir / (prefix + name + ".bin"))

    return {
        "input_layernorm.weight":         load("input_layernorm_weight"),
        "self_attn.q_proj.weight":        load("self_attn_q_proj_weight"),
        "self_attn.k_proj.weight":        load("self_attn_k_proj_weight"),
        "self_attn.v_proj.weight":        load("self_attn_v_proj_weight"),
        "self_attn.o_proj.weight":        load("self_attn_o_proj_weight"),
        "post_attention_layernorm.weight": load("post_attention_layernorm_weight"),
        "mlp.gate_proj.weight":           load("mlp_gate_proj_weight"),
        "mlp.up_proj.weight":             load("mlp_up_proj_weight"),
        "mlp.down_proj.weight":           load("mlp_down_proj_weight"),
    }


def rmsnorm_np(x: np.ndarray, gamma: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * gamma


def silu_np(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def apply_rope_np(proj: np.ndarray, num_heads: int,
                  cos: np.ndarray, sin: np.ndarray,
                  head_dim: int = 128) -> np.ndarray:
    s = proj.shape[0]
    half = head_dim // 2
    out = proj.copy()
    for p in range(s):
        for h in range(num_heads):
            base = h * head_dim
            for i in range(half):
                c, sn = cos[p, i], sin[p, i]
                a = out[p, base + i]
                b = out[p, base + i + half]
                out[p, base + i] = a * c - b * sn
                out[p, base + i + half] = a * sn + b * c
    return out


def run_numpy_forward(token_ids: list[int]) -> dict:
    """Run forward pass using gen_m2m3_fixtures.py style NumPy ops."""
    s = len(token_ids)
    emb = read_dump(DUMP_DIR / "embeddings.bin")
    x = emb[token_ids].astype(np.float32)
    first_layer_intermediates = None

    half = ref.H_D // 2
    i_vals = np.arange(half, dtype=np.float32)
    theta = 1.0 / (ref.ROPE_BASE ** (2.0 * i_vals / ref.H_D))
    positions = np.arange(s, dtype=np.float32).reshape(-1, 1)
    angles = positions * theta.reshape(1, -1)
    cos_np = np.cos(angles).astype(np.float32)
    sin_np = np.sin(angles).astype(np.float32)

    for li in range(ref.N_LAYERS):
        w = load_layer_weights(li)

        x_norm = rmsnorm_np(x, w["input_layernorm.weight"])
        Q = x_norm @ w["self_attn.q_proj.weight"].T
        K = x_norm @ w["self_attn.k_proj.weight"].T
        V = x_norm @ w["self_attn.v_proj.weight"].T

        Q_rope = apply_rope_np(Q, ref.H, cos_np, sin_np, ref.H_D)
        K_rope = apply_rope_np(K, ref.H_K, cos_np, sin_np, ref.H_D)

        Qh = Q_rope.reshape(s, ref.H, ref.H_D)
        Kh = K_rope.reshape(s, ref.H_K, ref.H_D)
        Vh = V.reshape(s, ref.H_K, ref.H_D)

        heads_per_group = ref.H // ref.H_K
        head_outs = []
        for hi in range(ref.H):
            kvg = hi // heads_per_group
            q_hi = Qh[:, hi, :]
            k_g = Kh[:, kvg, :]
            v_g = Vh[:, kvg, :]
            S = (q_hi @ k_g.T) * (1.0 / np.sqrt(ref.H_D))
            mask = np.triu(np.ones((s, s), dtype=np.float32), k=1) * (-1e6)
            S = S + mask
            row_max = S.max(axis=-1, keepdims=True)
            expS = np.exp(S - row_max)
            alpha = expS / expS.sum(axis=-1, keepdims=True)
            head_outs.append(alpha @ v_g)
        O = np.concatenate(head_outs, axis=-1)
        attn_out = O @ w["self_attn.o_proj.weight"].T
        x = x + attn_out

        x_norm2 = rmsnorm_np(x, w["post_attention_layernorm.weight"])
        gate = x_norm2 @ w["mlp.gate_proj.weight"].T
        up = x_norm2 @ w["mlp.up_proj.weight"].T
        h = silu_np(gate) * up
        ffn = h @ w["mlp.down_proj.weight"].T
        x = x + ffn

        if li == 0:
            first_layer_intermediates = {
                "x_norm": x_norm.copy(),
                "Q": Q.copy(), "K": K.copy(), "V": V.copy(),
                "Q_rope": Q_rope.copy(), "K_rope": K_rope.copy(),
                "O": O.copy(),
                "attn_out": attn_out.copy(),
                "ffn": ffn.copy(),
                "decoder_out": x.copy(),
            }

    final_gamma = read_dump(DUMP_DIR / "global" / "model_norm_weight.bin")
    x_norm_final = rmsnorm_np(x, final_gamma)
    lm_head = read_dump(DUMP_DIR / "global" / "lm_head_weight.bin")
    logits = x_norm_final[-1, :] @ lm_head.T
    next_token = int(np.argmax(logits))

    return {
        "pre_final_norm": x,
        "final_hidden":   x_norm_final,
        "logits":         logits,
        "next_token":     next_token,
        "layer0":         first_layer_intermediates,
    }


def run_torch_forward(token_ids: list[int]) -> dict:
    """Run forward pass using reference.py (PyTorch)."""
    s = len(token_ids)
    emb = read_dump(DUMP_DIR / "embeddings.bin")
    x = to_t(emb[token_ids])
    first_layer_intermediates = None

    cos, sin = ref.precompute_rope_tables(ref.H_D, s)

    for li in range(ref.N_LAYERS):
        raw_w = load_layer_weights(li)
        w = {k: to_t(v) for k, v in raw_w.items()}

        x_norm = ref.rmsnorm(x, w["input_layernorm.weight"])
        Q, K, V = ref.qkv_projections(x_norm,
                                      w["self_attn.q_proj.weight"],
                                      w["self_attn.k_proj.weight"],
                                      w["self_attn.v_proj.weight"])

        Qr = Q.view(s, ref.H,   ref.H_D).transpose(0, 1)
        Kr = K.view(s, ref.H_K, ref.H_D).transpose(0, 1)
        Vr = V.view(s, ref.H_K, ref.H_D).transpose(0, 1)
        Qr, Kr = ref.apply_rope(Qr, Kr, cos, sin)

        O = ref.grouped_query_attention(Qr, Kr, Vr)
        attn_out = ref.attention_output_proj(O, w["self_attn.o_proj.weight"])
        x = ref.residual_add(x, attn_out)

        x_norm2 = ref.rmsnorm(x, w["post_attention_layernorm.weight"])
        ffn = ref.swiglu_ffn(x_norm2,
                             w["mlp.gate_proj.weight"],
                             w["mlp.up_proj.weight"],
                             w["mlp.down_proj.weight"])
        x = ref.residual_add(x, ffn)

        if li == 0:
            # Transpose RoPE outputs back to (s, H*H_D) for comparison
            Q_rope_flat = Qr.transpose(0, 1).reshape(s, ref.H * ref.H_D)
            K_rope_flat = Kr.transpose(0, 1).reshape(s, ref.H_K * ref.H_D)
            first_layer_intermediates = {
                "x_norm": x_norm.numpy(),
                "Q": Q.numpy(), "K": K.numpy(), "V": V.numpy(),
                "Q_rope": Q_rope_flat.numpy(),
                "K_rope": K_rope_flat.numpy(),
                "O": O.numpy(),
                "attn_out": attn_out.numpy(),
                "ffn": ffn.numpy(),
                "decoder_out": x.numpy(),
            }

    final_gamma = to_t(read_dump(DUMP_DIR / "global" / "model_norm_weight.bin"))
    lm_head = to_t(read_dump(DUMP_DIR / "global" / "lm_head_weight.bin"))
    logits = ref.output_layer(x, final_gamma, lm_head)
    next_token = ref.greedy_decode_one_token(logits)

    return {
        "pre_final_norm": x.numpy(),
        "final_hidden":   ref.rmsnorm(x, final_gamma).numpy(),
        "logits":         logits.numpy(),
        "next_token":     next_token,
        "layer0":         first_layer_intermediates,
    }


def diff(name: str, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    abs_err = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-8)
    rel_err = abs_err / denom
    return float(abs_err.max()), float(rel_err.max())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=["hello", "medium"], default="hello")
    args = parser.parse_args()

    token_ids = PROMPTS[args.prompt]
    expected_next = EXPECTED_NEXT_TOKEN[args.prompt]

    print(f"Prompt: {args.prompt}  tokens: {token_ids}  s={len(token_ids)}")
    print("=" * 72)

    print("\n[1/2] Running reference.py (PyTorch)...")
    torch_out = run_torch_forward(token_ids)
    print(f"  next_token = {torch_out['next_token']}")

    print("\n[2/2] Running gen_m2m3_fixtures.py logic (NumPy)...")
    numpy_out = run_numpy_forward(token_ids)
    print(f"  next_token = {numpy_out['next_token']}")

    print("\n" + "=" * 72)
    print("Per-operator max absolute / relative diff (layer 0):")
    print(f"  {'operator':<18} {'max_abs':>12} {'max_rel':>12}")
    for key in ["x_norm", "Q", "K", "V", "Q_rope", "K_rope",
                "O", "attn_out", "ffn", "decoder_out"]:
        a, r = diff(key, torch_out["layer0"][key], numpy_out["layer0"][key])
        print(f"  {key:<18} {a:>12.6e} {r:>12.6e}")

    print("\nEnd-to-end (after 32 layers):")
    print(f"  {'tensor':<18} {'max_abs':>12} {'max_rel':>12}")
    for key in ["pre_final_norm", "final_hidden", "logits"]:
        a, r = diff(key, torch_out[key], numpy_out[key])
        print(f"  {key:<18} {a:>12.6e} {r:>12.6e}")

    print("\n" + "=" * 72)
    print("Argmax next-token comparison:")
    print(f"  reference.py (torch):  {torch_out['next_token']}")
    print(f"  gen_fixtures (numpy):  {numpy_out['next_token']}")
    print(f"  stored fixture:        {expected_next}")

    status = 0
    if torch_out["next_token"] != expected_next:
        print(f"  FAIL: reference.py disagrees with stored fixture")
        status = 1
    if numpy_out["next_token"] != expected_next:
        print(f"  FAIL: numpy disagrees with stored fixture")
        status = 1
    if torch_out["next_token"] == numpy_out["next_token"] == expected_next:
        print("  PASS: all three agree")

    return status


if __name__ == "__main__":
    raise SystemExit(main())

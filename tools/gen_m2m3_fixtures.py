#!/usr/bin/env python3
"""Generate golden fixture files for Milestone 2-3 tests.

Reads the binary dump files, computes reference outputs using NumPy,
and writes them as raw FP32 binary files under tests/data/m2m3/.

Usage:
    python3 tools/gen_m2m3_fixtures.py

Requires: numpy (no PyTorch dependency).
"""

import os
import struct
import numpy as np

DUMP_DIR = "assets/llama3/dump"
OUT_DIR = "tests/data/m2m3"
EMBEDDING_DIM = 4096
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
RMS_NORM_EPSILON = 1e-5

# Prompt for fixture generation: "Hello world"
# Must match the tokenizer output: [128000, 9906, 1917]
PROMPT_TOKEN_IDS = [128000, 9906, 1917]
SEQ_LEN = len(PROMPT_TOKEN_IDS)


def read_dump(path):
    """Read a binary dump file (280-byte header + payload) -> numpy FP32."""
    with open(path, "rb") as f:
        data = f.read()
    # Parse header
    # [0..255] name, [256..259] dtype, [260..263] ndims, [264..271] shape0, [272..279] shape1
    dtype_code = struct.unpack_from("<I", data, 256)[0]
    ndims = struct.unpack_from("<I", data, 260)[0]
    shape0 = struct.unpack_from("<Q", data, 264)[0]
    shape1 = struct.unpack_from("<Q", data, 272)[0]
    payload = data[280:]

    if dtype_code == 0:  # FP32
        arr = np.frombuffer(payload, dtype=np.float32)
    elif dtype_code == 1:  # FP16
        arr = np.frombuffer(payload, dtype=np.float16).astype(np.float32)
    elif dtype_code == 2:  # BF16
        raw = np.frombuffer(payload, dtype=np.uint16)
        arr = np.frombuffer((raw.astype(np.uint32) << 16).tobytes(),
                            dtype=np.float32)
    else:
        raise ValueError(f"unknown dtype_code {dtype_code}")

    if ndims == 1:
        return arr.reshape(shape0)
    return arr.reshape(shape0, shape1)


def save_fixture(name, arr):
    """Save a numpy array as raw FP32 binary."""
    path = os.path.join(OUT_DIR, name)
    arr.astype(np.float32).tofile(path)
    print(f"  wrote {path}  shape={arr.shape}  "
          f"min={arr.min():.6f} max={arr.max():.6f}")


def rmsnorm(x, gamma, eps=RMS_NORM_EPSILON):
    """Row-wise RMSNorm: (x / RMS(x)) * gamma."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * gamma


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating fixtures for prompt tokens: {PROMPT_TOKEN_IDS}")
    print(f"Sequence length: {SEQ_LEN}")

    # ---------------------------------------------------------------
    # 1. Load embeddings for the prompt
    # ---------------------------------------------------------------
    print("\n--- Embeddings ---")
    emb_table = read_dump(os.path.join(DUMP_DIR, "embeddings.bin"))
    print(f"  embedding table shape: {emb_table.shape}")
    X = emb_table[PROMPT_TOKEN_IDS]  # [3, 4096]
    save_fixture("embeddings_hello.bin", X)

    # ---------------------------------------------------------------
    # 2. Layer 0 weights
    # ---------------------------------------------------------------
    layer_dir = os.path.join(DUMP_DIR, "layer_00")
    print("\n--- Layer 0 weights ---")

    gamma_in = read_dump(
        os.path.join(layer_dir, "model_layers_0_input_layernorm_weight.bin"))
    print(f"  input_layernorm shape: {gamma_in.shape}")

    W_q = read_dump(
        os.path.join(layer_dir, "model_layers_0_self_attn_q_proj_weight.bin"))
    W_k = read_dump(
        os.path.join(layer_dir, "model_layers_0_self_attn_k_proj_weight.bin"))
    W_v = read_dump(
        os.path.join(layer_dir, "model_layers_0_self_attn_v_proj_weight.bin"))
    print(f"  W_q shape: {W_q.shape}")
    print(f"  W_k shape: {W_k.shape}")
    print(f"  W_v shape: {W_v.shape}")

    # ---------------------------------------------------------------
    # 3. RMSNorm fixture
    # ---------------------------------------------------------------
    print("\n--- RMSNorm ---")
    X_norm = rmsnorm(X, gamma_in)
    save_fixture("rmsnorm_layer0.bin", X_norm)

    # ---------------------------------------------------------------
    # 4. Q, K, V projections: output = X_norm @ W^T
    # ---------------------------------------------------------------
    print("\n--- Projections ---")
    Q = X_norm @ W_q.T  # [3, 4096]
    K = X_norm @ W_k.T  # [3, 1024]
    V = X_norm @ W_v.T  # [3, 1024]

    save_fixture("q_proj_layer0.bin", Q)
    save_fixture("k_proj_layer0.bin", K)
    save_fixture("v_proj_layer0.bin", V)

    # ---------------------------------------------------------------
    # 5. RoPE (rotate_full convention, base=500000)
    # ---------------------------------------------------------------
    print("\n--- RoPE ---")
    s = SEQ_LEN
    half_hd = HEAD_DIM // 2  # 64

    # Precompute theta: theta_i = 1 / (base ^ (2*i / head_dim))
    i_vals = np.arange(half_hd, dtype=np.float32)
    theta = 1.0 / (500000.0 ** (2.0 * i_vals / HEAD_DIM))

    # Angles for each position: [s, half_hd]
    positions = np.arange(s, dtype=np.float32).reshape(-1, 1)
    angles = positions * theta.reshape(1, -1)
    cos_table = np.cos(angles)
    sin_table = np.sin(angles)

    def apply_rope(proj, num_heads):
        """Apply RoPE in-place to a projected tensor [s, num_heads * head_dim]."""
        out = proj.copy()
        for p in range(s):
            for h in range(num_heads):
                base_idx = h * HEAD_DIM
                for i in range(half_hd):
                    c = cos_table[p, i]
                    sn = sin_table[p, i]
                    q_first = out[p, base_idx + i]
                    q_second = out[p, base_idx + i + half_hd]
                    out[p, base_idx + i] = q_first * c - q_second * sn
                    out[p, base_idx + i + half_hd] = q_first * sn + q_second * c
        return out

    Q_rope = apply_rope(Q, NUM_HEADS)
    K_rope = apply_rope(K, NUM_KV_HEADS)

    save_fixture("q_rope_layer0.bin", Q_rope)
    save_fixture("k_rope_layer0.bin", K_rope)

    # ---------------------------------------------------------------
    # 6. Attention (GQA with causal mask)
    # ---------------------------------------------------------------
    print("\n--- Attention ---")
    scale = 1.0 / np.sqrt(HEAD_DIM)
    heads_per_group = NUM_HEADS // NUM_KV_HEADS  # 4

    # Reshape for per-head computation
    Q_heads = Q_rope.reshape(s, NUM_HEADS, HEAD_DIM)  # [s, 32, 128]
    K_heads = K_rope.reshape(s, NUM_KV_HEADS, HEAD_DIM)  # [s, 8, 128]
    V_heads = V.reshape(s, NUM_KV_HEADS, HEAD_DIM)  # [s, 8, 128]

    attn_outputs = []
    for head_i in range(NUM_HEADS):
        kv_group = head_i // heads_per_group
        Qi = Q_heads[:, head_i, :]  # [s, hd]
        Kg = K_heads[:, kv_group, :]  # [s, hd]
        Vg = V_heads[:, kv_group, :]  # [s, hd]

        # Scores: S = Q @ K^T * scale
        S = (Qi @ Kg.T) * scale  # [s, s]

        if head_i == 0:
            save_fixture("attn_scores_head0.bin", S.astype(np.float32))

        # Causal mask
        mask = np.triu(np.ones((s, s), dtype=np.float32), k=1) * (-1e6)
        S = S + mask

        # Stable softmax
        row_max = S.max(axis=-1, keepdims=True)
        exp_S = np.exp(S - row_max)
        alpha = exp_S / exp_S.sum(axis=-1, keepdims=True)

        # Weighted sum
        O_i = alpha @ Vg  # [s, hd]
        attn_outputs.append(O_i)

        if head_i == 0:
            save_fixture("attn_output_head0.bin", O_i.astype(np.float32))

    # Concatenate all heads: [s, NUM_HEADS * HEAD_DIM]
    attn_concat = np.concatenate(attn_outputs, axis=-1)  # [s, 4096]
    save_fixture("attn_output_full.bin", attn_concat)

    # ---------------------------------------------------------------
    # 7. O projection, first residual, post-attn norm, FFN, second residual
    # ---------------------------------------------------------------
    print("\n--- Output projection + Residuals + FFN ---")
    W_o = read_dump(
        os.path.join(layer_dir, "model_layers_0_self_attn_o_proj_weight.bin"))
    gamma_post = read_dump(
        os.path.join(layer_dir,
                     "model_layers_0_post_attention_layernorm_weight.bin"))
    W_gate = read_dump(
        os.path.join(layer_dir, "model_layers_0_mlp_gate_proj_weight.bin"))
    W_up = read_dump(
        os.path.join(layer_dir, "model_layers_0_mlp_up_proj_weight.bin"))
    W_down = read_dump(
        os.path.join(layer_dir, "model_layers_0_mlp_down_proj_weight.bin"))

    # O projection: attn_out = attn_concat @ W_o^T
    attn_out = attn_concat @ W_o.T  # [s, 4096]
    save_fixture("o_proj_layer0.bin", attn_out)

    # First residual: X = X + attn_out
    X_res1 = X + attn_out
    save_fixture("residual1_layer0.bin", X_res1)

    # Post-attention RMSNorm
    X_norm2 = rmsnorm(X_res1, gamma_post)
    save_fixture("post_attn_rmsnorm_layer0.bin", X_norm2)

    # SwiGLU FFN
    def silu(x):
        return x / (1.0 + np.exp(-x))

    gate = X_norm2 @ W_gate.T  # [s, 14336]
    up = X_norm2 @ W_up.T      # [s, 14336]
    H = silu(gate) * up
    ffn_out = H @ W_down.T     # [s, 4096]
    save_fixture("swiglu_layer0.bin", ffn_out)

    # Second residual: X = X_res1 + ffn_out
    X_res2 = X_res1 + ffn_out
    save_fixture("decoder_block_layer0.bin", X_res2)

    # ---------------------------------------------------------------
    # 8. Full 32-layer forward pass + output layer
    # ---------------------------------------------------------------
    print("\n--- Full 32-layer forward pass ---")
    X_cur = X.copy()  # [s, 4096] - starting from embeddings

    for layer_idx in range(32):
        layer_dir_i = os.path.join(DUMP_DIR, f"layer_{layer_idx:02d}")
        prefix = f"model_layers_{layer_idx}_"

        gamma_in_i = read_dump(
            os.path.join(layer_dir_i, prefix + "input_layernorm_weight.bin"))
        gamma_post_i = read_dump(
            os.path.join(layer_dir_i, prefix + "post_attention_layernorm_weight.bin"))

        Wq_i = read_dump(
            os.path.join(layer_dir_i, prefix + "self_attn_q_proj_weight.bin"))
        Wk_i = read_dump(
            os.path.join(layer_dir_i, prefix + "self_attn_k_proj_weight.bin"))
        Wv_i = read_dump(
            os.path.join(layer_dir_i, prefix + "self_attn_v_proj_weight.bin"))
        Wo_i = read_dump(
            os.path.join(layer_dir_i, prefix + "self_attn_o_proj_weight.bin"))

        Wgate_i = read_dump(
            os.path.join(layer_dir_i, prefix + "mlp_gate_proj_weight.bin"))
        Wup_i = read_dump(
            os.path.join(layer_dir_i, prefix + "mlp_up_proj_weight.bin"))
        Wdown_i = read_dump(
            os.path.join(layer_dir_i, prefix + "mlp_down_proj_weight.bin"))

        # RMSNorm
        Xn = rmsnorm(X_cur, gamma_in_i)

        # Q, K, V projections
        Qi = Xn @ Wq_i.T
        Ki = Xn @ Wk_i.T
        Vi = Xn @ Wv_i.T

        # RoPE
        Qi = apply_rope(Qi, NUM_HEADS)
        Ki = apply_rope(Ki, NUM_KV_HEADS)

        # Attention
        Qh = Qi.reshape(s, NUM_HEADS, HEAD_DIM)
        Kh = Ki.reshape(s, NUM_KV_HEADS, HEAD_DIM)
        Vh = Vi.reshape(s, NUM_KV_HEADS, HEAD_DIM)

        heads_per_group_i = NUM_HEADS // NUM_KV_HEADS
        head_outs = []
        for hi in range(NUM_HEADS):
            kvg = hi // heads_per_group_i
            q_hi = Qh[:, hi, :]
            k_g = Kh[:, kvg, :]
            v_g = Vh[:, kvg, :]
            S_hi = (q_hi @ k_g.T) * (1.0 / np.sqrt(HEAD_DIM))
            mask = np.triu(np.ones((s, s), dtype=np.float32), k=1) * (-1e6)
            S_hi = S_hi + mask
            row_max = S_hi.max(axis=-1, keepdims=True)
            exp_S = np.exp(S_hi - row_max)
            alpha = exp_S / exp_S.sum(axis=-1, keepdims=True)
            O_hi = alpha @ v_g
            head_outs.append(O_hi)

        attn_cat = np.concatenate(head_outs, axis=-1)
        attn_out = attn_cat @ Wo_i.T

        # First residual
        X_cur = X_cur + attn_out

        # Post-attention RMSNorm
        Xn2 = rmsnorm(X_cur, gamma_post_i)

        # SwiGLU FFN
        gate_val = Xn2 @ Wgate_i.T
        up_val = Xn2 @ Wup_i.T
        H_val = silu(gate_val) * up_val
        ffn_out = H_val @ Wdown_i.T

        # Second residual
        X_cur = X_cur + ffn_out

        if (layer_idx + 1) % 8 == 0:
            print(f"  layer {layer_idx} done, "
                  f"X range: [{X_cur.min():.4f}, {X_cur.max():.4f}]")

    # Save pre-final-norm hidden state (for final_rmsnorm_fixture test)
    save_fixture("pre_final_norm_hello.bin", X_cur)

    # Final RMSNorm
    final_gamma = read_dump(
        os.path.join(DUMP_DIR, "global", "model_norm_weight.bin"))
    X_final = rmsnorm(X_cur, final_gamma)
    save_fixture("final_hidden.bin", X_final)

    # Last-token logits: x_last @ emb_table^T (lm_head = embedding table)
    x_last = X_final[-1, :]  # [4096]
    logits = x_last @ emb_table.T  # [128256]
    next_token = int(np.argmax(logits))
    print(f"\n  next token ID (greedy): {next_token}")
    save_fixture("logits_hello.bin", logits)

    # Save the answer for the test
    with open(os.path.join(OUT_DIR, "next_token_hello.txt"), "w") as f:
        f.write(f"{next_token}\n")
    print(f"  wrote {OUT_DIR}/next_token_hello.txt")

    # ---------------------------------------------------------------
    # 9. Medium prompt: "The capital of France is"
    # ---------------------------------------------------------------
    print("\n--- Medium prompt forward pass ---")

    # Token IDs verified against C++ BPETokenizer output:
    # BOS=128000, "The"=791, " capital"=6864, " of"=315, " France"=9822, " is"=374
    medium_ids = [128000, 791, 6864, 315, 9822, 374]
    print(f"  medium prompt tokens: {medium_ids}")

    medium_s = len(medium_ids)
    X_med = emb_table[medium_ids]  # [s_med, 4096]
    print(f"  sequence length: {medium_s}")

    # Recompute RoPE for medium sequence length
    med_positions = np.arange(medium_s, dtype=np.float32).reshape(-1, 1)
    med_angles = med_positions * theta.reshape(1, -1)
    med_cos = np.cos(med_angles)
    med_sin = np.sin(med_angles)

    def apply_rope_med(proj, num_heads):
        out = proj.copy()
        for p in range(medium_s):
            for h in range(num_heads):
                base_idx = h * HEAD_DIM
                for ii in range(half_hd):
                    c = med_cos[p, ii]
                    sn = med_sin[p, ii]
                    q_first = out[p, base_idx + ii]
                    q_second = out[p, base_idx + ii + half_hd]
                    out[p, base_idx + ii] = q_first * c - q_second * sn
                    out[p, base_idx + ii + half_hd] = q_first * sn + q_second * c
        return out

    X_med_cur = X_med.copy()
    for layer_idx in range(32):
        layer_dir_i = os.path.join(DUMP_DIR, f"layer_{layer_idx:02d}")
        prefix = f"model_layers_{layer_idx}_"

        gamma_in_i = read_dump(
            os.path.join(layer_dir_i, prefix + "input_layernorm_weight.bin"))
        gamma_post_i = read_dump(
            os.path.join(layer_dir_i,
                         prefix + "post_attention_layernorm_weight.bin"))

        Wq_i = read_dump(
            os.path.join(layer_dir_i, prefix + "self_attn_q_proj_weight.bin"))
        Wk_i = read_dump(
            os.path.join(layer_dir_i, prefix + "self_attn_k_proj_weight.bin"))
        Wv_i = read_dump(
            os.path.join(layer_dir_i, prefix + "self_attn_v_proj_weight.bin"))
        Wo_i = read_dump(
            os.path.join(layer_dir_i, prefix + "self_attn_o_proj_weight.bin"))

        Wgate_i = read_dump(
            os.path.join(layer_dir_i, prefix + "mlp_gate_proj_weight.bin"))
        Wup_i = read_dump(
            os.path.join(layer_dir_i, prefix + "mlp_up_proj_weight.bin"))
        Wdown_i = read_dump(
            os.path.join(layer_dir_i, prefix + "mlp_down_proj_weight.bin"))

        Xn = rmsnorm(X_med_cur, gamma_in_i)
        Qi = Xn @ Wq_i.T
        Ki = Xn @ Wk_i.T
        Vi = Xn @ Wv_i.T
        Qi = apply_rope_med(Qi, NUM_HEADS)
        Ki = apply_rope_med(Ki, NUM_KV_HEADS)

        Qh = Qi.reshape(medium_s, NUM_HEADS, HEAD_DIM)
        Kh = Ki.reshape(medium_s, NUM_KV_HEADS, HEAD_DIM)
        Vh = Vi.reshape(medium_s, NUM_KV_HEADS, HEAD_DIM)

        head_outs = []
        for hi in range(NUM_HEADS):
            kvg = hi // (NUM_HEADS // NUM_KV_HEADS)
            q_hi = Qh[:, hi, :]
            k_g = Kh[:, kvg, :]
            v_g = Vh[:, kvg, :]
            S_hi = (q_hi @ k_g.T) * (1.0 / np.sqrt(HEAD_DIM))
            mask = np.triu(np.ones((medium_s, medium_s),
                                    dtype=np.float32), k=1) * (-1e6)
            S_hi = S_hi + mask
            row_max = S_hi.max(axis=-1, keepdims=True)
            exp_S = np.exp(S_hi - row_max)
            alpha = exp_S / exp_S.sum(axis=-1, keepdims=True)
            O_hi = alpha @ v_g
            head_outs.append(O_hi)

        attn_cat = np.concatenate(head_outs, axis=-1)
        attn_out = attn_cat @ Wo_i.T
        X_med_cur = X_med_cur + attn_out

        Xn2 = rmsnorm(X_med_cur, gamma_post_i)
        gate_val = Xn2 @ Wgate_i.T
        up_val = Xn2 @ Wup_i.T
        H_val = silu(gate_val) * up_val
        ffn_out = H_val @ Wdown_i.T
        X_med_cur = X_med_cur + ffn_out

        if (layer_idx + 1) % 8 == 0:
            print(f"  layer {layer_idx} done")

    X_med_final = rmsnorm(X_med_cur, final_gamma)
    x_med_last = X_med_final[-1, :]
    med_logits = x_med_last @ emb_table.T
    med_token = int(np.argmax(med_logits))
    print(f"  next token ID (greedy): {med_token}")

    with open(os.path.join(OUT_DIR, "next_token_medium.txt"), "w") as f:
        f.write(f"{med_token}\n")
    print(f"  wrote {OUT_DIR}/next_token_medium.txt")

    print(f"\nAll fixtures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()

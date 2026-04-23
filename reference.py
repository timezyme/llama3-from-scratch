"""
CS265 Project – Part 2: PyTorch Reference Implementations
==========================================================
These functions implement every operator described in Part 2 at the PyTorch
level.  Use them to verify the numerical correctness of your CUDA kernels.
They are NOT part of your submission; your project must be implemented in C++/CUDA.

Weight names used here match the HuggingFace Llama-3 checkpoint exactly, so you
can load the model with `transformers` and extract tensors by name for comparison.

Model: meta-llama/Meta-Llama-3-8B-Instruct
"""

import math
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hyperparameters  (Llama-3-8B)
# ---------------------------------------------------------------------------
D       = 4096       # embedding dimension
H       = 32         # number of query heads
H_K     = 8          # number of key/value heads  (GQA)
H_D     = 128        # head dimension  (D == H * H_D)
D_FF    = 14336      # FFN intermediate size
VOCAB   = 128256     # vocabulary size
N_LAYERS= 32         # number of decoder blocks
EPS     = 1e-5       # RMSNorm epsilon
ROPE_BASE = 500_000  # RoPE base frequency (Llama-3 specific, NOT 10000)

# ---------------------------------------------------------------------------
# Milestone 2 – RMSNorm and Q, K, V projections
# ---------------------------------------------------------------------------

def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    RMSNorm applied row-wise.

    Args:
        x     : (s, d)   – input matrix (one row per token)
        gamma : (d,)     – learned scale  [checkpoint: input_layernorm.weight
                                            or post_attention_layernorm.weight]
        eps   : float    – added inside the square root for numerical stability

    Returns:
        out   : (s, d)   – normalized matrix
    """
    # sum of squares over last dimension, keep dims for broadcasting
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)  # (s, 1)
    return (x / rms) * gamma                                       # (s, d)


def qkv_projections(
    x_norm: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Q, K, V via three independent matmuls.

    Args:
        x_norm : (s, d)            – RMSNorm output
        W_q    : (H*H_D,  d)       – query projection    [q_proj.weight]
        W_k    : (H_K*H_D, d)      – key projection      [k_proj.weight]
        W_v    : (H_K*H_D, d)      – value projection    [v_proj.weight]

    Returns:
        Q : (s, H*H_D)
        K : (s, H_K*H_D)
        V : (s, H_K*H_D)
    """
    Q = x_norm @ W_q.T   # (s, H*H_D)   == (s, 4096)
    K = x_norm @ W_k.T   # (s, H_K*H_D) == (s, 1024)
    V = x_norm @ W_v.T   # (s, H_K*H_D) == (s, 1024)
    return Q, K, V


# ---------------------------------------------------------------------------
# Milestone 3 – RoPE
# ---------------------------------------------------------------------------

def precompute_rope_tables(
    h_d: int,
    max_seq_len: int,
    base: float = ROPE_BASE,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute cosine and sine tables for RoPE.

    The angle for dimension pair i is:
        theta_i = 1 / base^(2i / h_d),  i in {0, ..., h_d/2 - 1}

    cos/sin are returned in shape (max_seq_len, h_d) by repeating the
    (max_seq_len, h_d/2) tables twice along the last dimension, matching
    the HuggingFace implementation.

    Returns:
        cos : (max_seq_len, h_d)
        sin : (max_seq_len, h_d)
    """
    i = torch.arange(0, h_d // 2, dtype=torch.float32, device=device)
    theta = 1.0 / (base ** (2 * i / h_d))                    # (h_d/2,)
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, theta)                      # (s, h_d/2)
    emb = torch.cat([freqs, freqs], dim=-1)                    # (s, h_d)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate by splitting into first half and second half (not interleaved).
        rotate_half([a, b]) = [-b, a]
    where a = x[..., :h_d/2] and b = x[..., h_d/2:]
    """
    x1 = x[..., : x.shape[-1] // 2]    # first half
    x2 = x[..., x.shape[-1] // 2 :]    # second half
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    Q: torch.Tensor,
    K: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to Q and K in-place (returns new tensors).

    Args:
        Q   : (H,   s, H_D)   – query heads after reshape
        K   : (H_K, s, H_D)   – key heads after reshape
        cos : (s, H_D)         – precomputed cosine table
        sin : (s, H_D)         – precomputed sine table

    Returns:
        Q_rot : (H,   s, H_D)
        K_rot : (H_K, s, H_D)
    """
    # broadcast cos/sin over the head dimension: (1, s, H_D)
    cos = cos.unsqueeze(0)  # (1, s, H_D)
    sin = sin.unsqueeze(0)  # (1, s, H_D)
    Q_rot = Q * cos + rotate_half(Q) * sin
    K_rot = K * cos + rotate_half(K) * sin
    return Q_rot, K_rot


# ---------------------------------------------------------------------------
# Milestone 3 – Grouped Query Attention (GQA)
# ---------------------------------------------------------------------------

def grouped_query_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """
    Grouped Query Attention with causal masking and numerically stable softmax.

    H=32 query heads are divided into H_K=8 groups of H/H_K=4 heads.
    Each group shares one K head and one V head.
    Standard MHA would have H K/V heads; GQA uses only H_K, saving 4x memory.

    Args:
        Q : (H,   s, H_D)  – query heads (after RoPE)
        K : (H_K, s, H_D)  – key heads   (after RoPE)
        V : (H_K, s, H_D)  – value heads

    Returns:
        O : (s, H * H_D)   – concatenated head outputs
    """
    h, s, h_d = Q.shape
    h_k = K.shape[0]
    group_size = h // h_k           # = 4
    scale = 1.0 / math.sqrt(h_d)   # = 1/sqrt(128)

    # Causal mask: upper triangle (positions q > p) gets -inf before softmax.
    # Using a large negative value (-1e6) approximates -inf and avoids NaN.
    causal_mask = torch.full((s, s), float("-inf"), device=Q.device, dtype=Q.dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)  # (s, s)

    outputs = []
    for i in range(h):
        g = i // group_size          # KV head index for query head i

        q_i = Q[i]     # (s, H_D)
        k_g = K[g]     # (s, H_D)
        v_g = V[g]     # (s, H_D)

        # Scaled dot-product scores
        scores = (q_i @ k_g.T) * scale   # (s, s)

        # Apply causal mask
        scores = scores + causal_mask     # (s, s)

        # Numerically stable softmax: subtract row max before exponentiating.
        # This is mathematically equivalent to standard softmax.
        row_max = scores.max(dim=-1, keepdim=True).values    # (s, 1)
        scores  = scores - row_max
        exp_s   = scores.exp()
        alpha   = exp_s / exp_s.sum(dim=-1, keepdim=True)    # (s, s)

        out_i   = alpha @ v_g           # (s, H_D)
        outputs.append(out_i)

    # Concatenate all head outputs
    O = torch.cat(outputs, dim=-1)      # (s, H * H_D)
    return O


# ---------------------------------------------------------------------------
# Milestone 3 – Output projection and residual
# ---------------------------------------------------------------------------

def attention_output_proj(O: torch.Tensor, W_o: torch.Tensor) -> torch.Tensor:
    """
    Project concatenated attention output back to embedding dimension d.

    Args:
        O   : (s, H * H_D)     – concatenated head outputs
        W_o : (d, H * H_D)     – output projection   [o_proj.weight]

    Returns:
        attn_out : (s, d)
    """
    return O @ W_o.T    # (s, d)


def residual_add(x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    """
    Elementwise residual addition.  Same kernel is reused for both residuals.

    Args:
        x      : (s, d)
        update : (s, d)

    Returns:
        (s, d)
    """
    return x + update


# ---------------------------------------------------------------------------
# Milestone 3 – SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

def swiglu_ffn(
    x_norm: torch.Tensor,
    W_gate: torch.Tensor,
    W_up:   torch.Tensor,
    W_down: torch.Tensor,
) -> torch.Tensor:
    """
    SwiGLU feed-forward network.

    Three matmuls + one elementwise SiLU-gating kernel:
        gate     = x_norm @ W_gate^T                  (s, d_ff)
        up       = x_norm @ W_up^T                    (s, d_ff)
        hidden   = SiLU(gate) * up                    (s, d_ff)  <-- new kernel
        ffn_out  = hidden @ W_down^T                  (s, d)

    SiLU(z) = z * sigmoid(z) = z / (1 + exp(-z))  applied elementwise.

    Args:
        x_norm : (s, d)
        W_gate : (d_ff, d)  [mlp.gate_proj.weight]
        W_up   : (d_ff, d)  [mlp.up_proj.weight]
        W_down : (d, d_ff)  [mlp.down_proj.weight]

    Returns:
        ffn_out : (s, d)
    """
    gate    = x_norm @ W_gate.T                  # (s, d_ff)
    up      = x_norm @ W_up.T                    # (s, d_ff)
    hidden  = F.silu(gate) * up                  # (s, d_ff)
    ffn_out = hidden @ W_down.T                  # (s, d)
    return ffn_out


# ---------------------------------------------------------------------------
# Milestone 3 – Full decoder block
# ---------------------------------------------------------------------------

def decoder_block(
    x: torch.Tensor,
    weights: dict,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    One complete Llama-3 decoder block.

    Execution order:
        RMSNorm -> QKV projections -> reshape -> RoPE -> GQA ->
        output proj -> residual -> RMSNorm -> SwiGLU FFN -> residual

    Args:
        x       : (s, d)    – input token embeddings
        weights : dict      – weight tensors for this layer (see key names below)
        cos/sin : (s, H_D)  – precomputed RoPE tables

    Weight keys expected (matching HuggingFace checkpoint names):
        input_layernorm.weight
        self_attn.q_proj.weight   : (H*H_D, d)
        self_attn.k_proj.weight   : (H_K*H_D, d)
        self_attn.v_proj.weight   : (H_K*H_D, d)
        self_attn.o_proj.weight   : (d, H*H_D)
        post_attention_layernorm.weight
        mlp.gate_proj.weight      : (d_ff, d)
        mlp.up_proj.weight        : (d_ff, d)
        mlp.down_proj.weight      : (d, d_ff)
    """
    s = x.shape[0]

    # --- Attention sub-block ---
    x_norm = rmsnorm(x, weights["input_layernorm.weight"])

    Q, K, V = qkv_projections(
        x_norm,
        weights["self_attn.q_proj.weight"],
        weights["self_attn.k_proj.weight"],
        weights["self_attn.v_proj.weight"],
    )

    # Reshape to per-head layout: (s, H*H_D) -> (H, s, H_D)
    Q = Q.view(s, H,   H_D).transpose(0, 1)   # (H,   s, H_D)
    K = K.view(s, H_K, H_D).transpose(0, 1)   # (H_K, s, H_D)
    V = V.view(s, H_K, H_D).transpose(0, 1)   # (H_K, s, H_D)

    Q, K = apply_rope(Q, K, cos, sin)

    O = grouped_query_attention(Q, K, V)       # (s, H*H_D)

    attn_out = attention_output_proj(O, weights["self_attn.o_proj.weight"])
    x = residual_add(x, attn_out)

    # --- FFN sub-block ---
    x_norm = rmsnorm(x, weights["post_attention_layernorm.weight"])

    ffn_out = swiglu_ffn(
        x_norm,
        weights["mlp.gate_proj.weight"],
        weights["mlp.up_proj.weight"],
        weights["mlp.down_proj.weight"],
    )
    x = residual_add(x, ffn_out)

    return x


# ---------------------------------------------------------------------------
# Milestone 3 – Output layer and greedy token generation
# ---------------------------------------------------------------------------

def output_layer(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    W_lm: torch.Tensor,
) -> torch.Tensor:
    """
    Final RMSNorm + lm_head projection for the last token position only.

    Args:
        x           : (s, d)   – output of the last decoder block
        norm_weight : (d,)     – final norm weight  [model.norm.weight]
        W_lm        : (V, d)   – lm head            [lm_head.weight]
                                 (shared with embedding table in Llama-3)

    Returns:
        logits : (V,)  – unnormalized scores over the vocabulary
    """
    x_norm  = rmsnorm(x, norm_weight)               # (s, d)
    x_last  = x_norm[-1, :].unsqueeze(0)            # (1, d) – last token only
    logits  = (x_last @ W_lm.T).squeeze(0)          # (V,)
    return logits


def greedy_decode_one_token(logits: torch.Tensor) -> int:
    """
    Greedy decoding: return the vocabulary index with the highest logit.

    Args:
        logits : (V,)

    Returns:
        token_id : int
    """
    return int(logits.argmax().item())


# ---------------------------------------------------------------------------
# End-to-end forward pass (one token generation step)
# ---------------------------------------------------------------------------

def forward_one_step(
    token_ids: list[int],
    all_layer_weights: list[dict],
    final_norm_weight: torch.Tensor,
    W_lm: torch.Tensor,
    embed_table: torch.Tensor,
) -> int:
    """
    Full forward pass for a single generation step (no KV cache).

    Args:
        token_ids          : list of s token IDs (current sequence)
        all_layer_weights  : list of 32 weight dicts, one per decoder block
        final_norm_weight  : (d,)    [model.norm.weight]
        W_lm               : (V, d)  [lm_head.weight]
        embed_table        : (V, d)  [model.embed_tokens.weight]

    Returns:
        new_token_id : int  – the next predicted token
    """
    s = len(token_ids)

    # Embedding lookup: token IDs -> (s, d)
    ids = torch.tensor(token_ids, dtype=torch.long)
    x = embed_table[ids]                              # (s, d)

    # Pre-compute RoPE tables for this sequence length
    cos, sin = precompute_rope_tables(H_D, s, device=x.device)

    # Run all 32 decoder blocks
    for layer_idx in range(N_LAYERS):
        x = decoder_block(x, all_layer_weights[layer_idx], cos, sin)

    # Output layer -> logits for the last position
    logits = output_layer(x, final_norm_weight, W_lm)

    # Greedy decode
    return greedy_decode_one_token(logits)


# ---------------------------------------------------------------------------
# Quick sanity-check shapes (run with: python reference.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    dtype  = torch.float32
    s      = 10    # short sequence for testing

    x      = torch.randn(s, D, dtype=dtype)
    gamma  = torch.ones(D, dtype=dtype)

    # RMSNorm
    x_norm = rmsnorm(x, gamma)
    assert x_norm.shape == (s, D), f"RMSNorm shape mismatch: {x_norm.shape}"
    print(f"RMSNorm        : {x_norm.shape}  OK")

    # QKV
    W_q = torch.randn(H * H_D,    D, dtype=dtype)
    W_k = torch.randn(H_K * H_D,  D, dtype=dtype)
    W_v = torch.randn(H_K * H_D,  D, dtype=dtype)
    Q, K, V = qkv_projections(x_norm, W_q, W_k, W_v)
    assert Q.shape == (s, H * H_D)
    assert K.shape == (s, H_K * H_D)
    assert V.shape == (s, H_K * H_D)
    print(f"Q              : {Q.shape}  OK")
    print(f"K              : {K.shape}  OK")
    print(f"V              : {V.shape}  OK")

    # Reshape
    Q = Q.view(s, H,   H_D).transpose(0, 1)
    K = K.view(s, H_K, H_D).transpose(0, 1)
    V = V.view(s, H_K, H_D).transpose(0, 1)
    print(f"Q reshaped     : {Q.shape}  OK")
    print(f"K reshaped     : {K.shape}  OK")

    # RoPE
    cos, sin = precompute_rope_tables(H_D, s)
    Q, K = apply_rope(Q, K, cos, sin)
    print(f"Q after RoPE   : {Q.shape}  OK")

    # GQA
    O = grouped_query_attention(Q, K, V)
    assert O.shape == (s, H * H_D)
    print(f"Attention out  : {O.shape}  OK")

    # Output proj + residual
    W_o  = torch.randn(D, H * H_D, dtype=dtype)
    attn_out = attention_output_proj(O, W_o)
    x_out    = residual_add(x, attn_out)
    assert x_out.shape == (s, D)
    print(f"After residual : {x_out.shape}  OK")

    # SwiGLU
    W_gate = torch.randn(D_FF, D, dtype=dtype)
    W_up   = torch.randn(D_FF, D, dtype=dtype)
    W_down = torch.randn(D, D_FF, dtype=dtype)
    ffn_out = swiglu_ffn(x_norm, W_gate, W_up, W_down)
    assert ffn_out.shape == (s, D)
    print(f"SwiGLU out     : {ffn_out.shape}  OK")

    # Output layer
    W_lm        = torch.randn(VOCAB, D, dtype=dtype)
    final_norm  = torch.ones(D, dtype=dtype)
    logits      = output_layer(x_out, final_norm, W_lm)
    assert logits.shape == (VOCAB,)
    print(f"Logits         : {logits.shape}  OK")

    token_id = greedy_decode_one_token(logits)
    print(f"Predicted token: {token_id}  OK")
    print("\nAll shape checks passed.")

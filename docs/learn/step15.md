---

## Step 15: Final RMSNorm + Last-Token Extract + lm_head

**File:** `src/inference_layer.cu:444-461` (final norm + extract), `145-156` (lm_head logits)
**Where in the pipeline:** We've exited the 32-layer loop. `X` is now the model's final hidden representation. Three operations convert it into a predicted next token.

### High-level picture

```
X_out = RMSNorm(X, model.norm.weight)       [s, 4096]   ← final norm (line 448)
x_last = X_out[s-1, :]                      [4096]       ← last-token extract (line 455-460)
logits = lm_head @ x_last                   [128256]     ← vocabulary scores (line 145-156)
```

Three steps, each with a purpose:

**1. Final RMSNorm** (line 446-448): Uses `model.norm.weight` — a third gamma vector, separate from any layer's gammas. This is the 65th RMSNorm call in the forward pass.

**2. Last-token extract** (lines 454-460): Only copy row `s-1` back to the CPU. The model's prediction for the next token is encoded in the *last* position's hidden state. Rows 0 through s-2 predicted the tokens that already exist in the prompt — we don't need those predictions. For batched inference, it extracts `X_out[b * q_seq + (q_seq - 1), :]` for each batch slot.

**3. lm_head projection** (lines 145-156): Multiply the 4096-dim hidden vector by the lm_head weight matrix `[128256, 4096]` to get a score (logit) for every possible token in the vocabulary. The token with the highest logit is the model's best guess for what comes next.

### lm_head: the biggest TA-scrutiny point here

Two pitfalls from Part 2 section 4:

**lm_head is NOT the embedding table.** The assignment text says "shared with the embedding table in Llama 3." That's true for vanilla Llama 3, but **not** for the Llama-3-8B-Instruct checkpoint used in this project. The config has `tie_word_embeddings = false`, so lm_head is a separate `[128256, 4096]` weight matrix loaded from `global/lm_head_weight.bin` (see `model_weights.cpp:53-55`). Using the embedding table here produces wrong logits — they differ by up to 0.345. This is documented in `docs/learnings.md`.

**Last-token only.** Projecting all `s` rows through lm_head would produce an `[s, 128256]` matrix — that's `s * 128256 * 4 bytes`. For s=512, that's ~250 MB of useless output. Only the last row matters for greedy decoding.

### lm_head runs on CPU

`compute_lm_head_logits` (line 145) is a CPU-side dot-product loop — not a GPU kernel. It computes 128,256 dot products of 4,096 elements each. This works because it's only one row (the last token), making it effectively a matrix-vector product. A dedicated GPU GEMV kernel would be faster but is an optional optimization per the assignment.

### Where this sits in the full pipeline

```
Embeddings → [32x Decoder Block] → Final RMSNorm → extract x_last → lm_head → logits
                                                                                  ↓
                                                                              argmax (step 16)
```

---

**TA-style question:**

The assignment text says lm_head shares weights with the embedding table. Your implementation loads them separately. If a TA challenges you on this — "why aren't you using the shared embedding table like the spec says?" — what evidence would you point to in the model's config and what numerical test proves the two matrices are different?

**answer**

Two pieces of evidence:

**1. The config file.** `assets/llama3/config.json` has `"tie_word_embeddings": false`. This is the authoritative flag — HuggingFace's `from_pretrained` checks it to decide whether lm_head gets its own tensor or aliases the embedding table. When it's `false`, the checkpoint contains a distinct `lm_head.weight` tensor.

**2. Numerical comparison.** The two matrices differ by up to 0.345 in absolute value (documented in `docs/learnings.md`). If they were the same tensor, the max diff would be exactly 0.0. You can verify this with `tools/verify_reference.py`, which runs the PyTorch reference forward pass using the real checkpoint weights and confirms the separate lm_head produces the correct argmax token.

The assignment text's statement "shared with the embedding table in Llama 3" is correct for the *base* Llama 3 model but not for the *Instruct* fine-tuned variant used in this project. The instruct fine-tune unfroze lm_head and trained it separately, so the two tensors diverged. Using the embedding table as lm_head would produce wrong next-token predictions.

---

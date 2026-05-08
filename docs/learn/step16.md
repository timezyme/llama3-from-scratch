---

## Step 16: Argmax + Decode

**Files:** `src/inference_loop.cu:179-180` (argmax), `src/tokenizer_bpe.cpp:40-49` (decode)
**Where in the pipeline:** The very end. lm_head produced 128,256 logits — one score per vocabulary token. We pick the winner and convert it back to text.

### High-level picture

Two operations:

```
token_id = argmax(logits)              ← pick the highest-scoring token (line 179-180)
text = tokenizer.decode([token_id])    ← convert ID back to string (line 40-49)
```

That's it. The entire 32-layer forward pass bottlenecks down to one `std::max_element` call over 128,256 floats and one table lookup.

### Argmax (greedy decoding)

Line 179-180:
```cpp
next_ids[b] = static_cast<int>(
    std::max_element(logits.begin(), logits.end()) - logits.begin());
```

This is **greedy decoding** — always pick the token with the highest logit. No randomness, no temperature, no top-k/top-p sampling. The assignment requires greedy decoding; sampling strategies are out of scope.

The result is a single integer: the token ID of the predicted next word (or word-piece).

### Decode (token ID -> text)

`BPETokenizer::decode()` at `tokenizer_bpe.cpp:40-49` reverses the encoding:
- Look up the token ID in `id2tok` to get the raw byte string (line 45-46)
- Skip any special tokens (BOS, EOT, headers) — they're control markers, not user-visible text (line 43-44)

For example, token ID `12366` maps to the byte string `" Paris"` (with the leading space, since BPE tokens often include whitespace).

### The autoregressive loop

Step 16 completes one token generation. For multi-token generation (paths 2-4 from step 1), the newly generated token ID feeds back into the pipeline:

```
token_id → embed → forward_step(q_seq=1) → lm_head → argmax → next token_id → ...
```

Each decode step only processes the *one new token* (q_seq=1), because the KV cache already holds the K/V values from all previous positions. The loop repeats until either `max_new_tokens` is reached or the model emits `EOT_ID` (128009) — its way of saying "I'm done answering."

This is visible in `inference_loop.cu:187-209`: a `for` loop that embeds one token, calls `forward_step` with `q_seq=1`, runs lm_head, does argmax, and checks for EOT.

### The full pipeline, end to end

You can now trace one token through the entire system:

```
"What is 2+2?" → chat template → BPE encode → embed [s, 4096]
→ 32x { RMSNorm → QKV → RoPE → GQA Attention → O proj → residual
        → RMSNorm → SwiGLU FFN → residual }
→ final RMSNorm → extract last token → lm_head [128256] → argmax → token 578
→ decode → " The"
→ feed back, repeat...
```

---

**TA-style question:**

Greedy decoding always picks the single highest-probability token. Why might this not always produce the best overall response, and what alternative decoding strategies exist? (This is a conceptual question — no code in this project implements alternatives.)

**answer**

Greedy decoding is locally optimal but not globally optimal. Picking the best token at each step can paint the model into a corner — the highest-probability first word might lead to a worse overall sentence than a slightly less likely first word that opens up better continuations. It's like chess: the best immediate move isn't always the best strategy.

Alternatives:

- **Beam search**: Track the top-k partial sequences in parallel, scoring each full path. More compute, but finds higher-probability sequences overall.
- **Temperature sampling**: Divide logits by a temperature T before softmax. T<1 sharpens the distribution (more deterministic), T>1 flattens it (more creative/random).
- **Top-k sampling**: Zero out all logits except the top k candidates, then sample from the remainder. Prevents the model from picking extremely unlikely tokens.
- **Top-p (nucleus) sampling**: Keep the smallest set of tokens whose cumulative probability exceeds p (e.g., 0.95), then sample. Adapts the candidate set size dynamically — narrow when the model is confident, wide when it's uncertain.

For this project, greedy is correct and sufficient — the assignment only asks for deterministic one-token-at-a-time generation.

---

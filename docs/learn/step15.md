## Step 15: Final RMSNorm, Last-Token Extract, and lm_head

This step is needed because the decoder blocks produce hidden states, not a
token ID. The final RMSNorm puts the last hidden row on the scale `lm_head`
expects, `lm_head` turns it into vocabulary scores, and greedy decoding picks
the biggest score.

### Main idea

After 32 layers, `X` has one 4,096-number row per token. For next-token
generation, only the last row answers the current question: what token comes
next?

Earlier rows describe earlier positions in the prompt. Projecting all of them
would do extra work for tokens we are not choosing now.

The final RMSNorm runs at `src/inference_layer.cu:448`. It uses
`model.norm.weight`, not either RMSNorm weight from inside a decoder layer.

Then `forward_step` copies only the last normalized row back to the CPU at
`src/inference_layer.cu:458`.

`lm_head` is the output matrix:

```text
lm_head: [128256, 4096]
last row: [4096]
logits:  [128256]
```

Logits are raw scores, one per vocabulary entry. The CPU helper
`compute_lm_head_logits` starts at `src/inference_layer.cu:145` and computes one  
dot product per vocabulary row. The caller chooses the largest logit as the next  
token ID.

### Where it fits

This is the last model step before tokenizer decode.

One project detail matters here. The assignment text says `lm_head` is shared
with the embedding table. That is true for some Llama checkpoints, but not this
one. This checkpoint sets `tie_word_embeddings = false`, so the loader reads a
separate `global/lm_head_weight.bin` tensor at `src/model_weights.cpp:53`.

Using the embedding table here would still have the right shape, but it would
produce the wrong logits.

### Review question

Why does this step project only the last hidden row, and which weight matrix
should it use?

**answer**

Only the last row predicts the next token after the whole current sequence.
Earlier rows correspond to earlier prompt positions, so projecting them would be
wasted work for greedy decoding.

It should use the separate `lm_head.weight`, not the embedding table. This
checkpoint has `tie_word_embeddings = false`, so `lm_head` is its own learned
output matrix.
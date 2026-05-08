## Step 16: Argmax and Decode

This step is needed because logits are only scores, not text. Argmax chooses one
token ID from those scores, and decode turns that ID into the text the user
actually sees.

### Main idea

After `lm_head`, the model has one logit per vocabulary token:

```text
logits: [128256]
```

Greedy decoding means: pick the largest logit. The first generated token uses
`std::max_element` at `src/inference_loop.cu:180`.

There is no softmax here. Softmax would turn logits into probabilities, but it
would not change which logit is largest:

```text
argmax(logits) == argmax(softmax(logits))
```

So greedy decoding can skip softmax and get the same token ID.

For multi-token generation, that chosen ID becomes the next input token. The
decode loop chooses later tokens the same way at `src/inference_loop.cu:220`,
then marks the sequence done if the new ID is `EOT_ID`.

### Where it fits

The token ID is still just an integer. To print text, the program calls
`decode_token` at `src/inference.cu:88`.

That function uses the BPE tokenizer. BPE means byte-pair encoding: common byte
patterns get stored as reusable token pieces. A decoded token may already include
a leading space, so the program should not add its own extra space.

`BPETokenizer::decode` starts at `src/tokenizer_bpe.cpp:40`. It skips special
IDs like beginning-of-text and end-of-turn, then appends the normal token piece
from `id2tok`.

Special IDs are control markers for the chat template. They help the model
understand roles and turns, but they are not user-visible answer text.

### Review question

Why does greedy decoding skip softmax, and why does decode skip special tokens?

**answer**

Greedy decoding only needs the index of the largest score. Softmax changes the
scale of the scores, but not their order, so the largest logit stays the winner.

Decode skips special tokens because they are control markers, not answer text.
Printing them would leak chat-template scaffolding into the output.

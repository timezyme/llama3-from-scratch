---

## Step 2: Chat Template

**File:** `src/inference_chat.cu:38-55`
**Where in the pipeline:** Right after the CLI picks a path, before tokenization. Every prompt passes through here on the way to the model.

### High-level picture

Llama 3 8B Instruct isn't a raw text-completion model — it was fine-tuned on conversations formatted with a specific template. If you feed it a bare prompt like `"What is 2+2?"`, it produces garbage. The chat template wraps your prompt so it looks like what the model saw during training.

For the prompt `"What is 2+2?"`, the function builds this token sequence:

```
<|begin_of_text|>                    ← 128000  (BOS: "Beginning Of Sequence")
<|start_header_id|>user<|end_header_id|>\n\n   ← 128006, 882, 128007, 271
What is 2+2?                         ← [your prompt's BPE tokens]
<|eot_id|>                           ← 128009  (EOT: "End Of Turn")
<|start_header_id|>assistant<|end_header_id|>\n\n  ← 128006, 78191, 128007, 271
```

The user turn is **closed** with `<|eot_id|>` — "I'm done talking." The assistant turn is **open** — header present, but no body and no EOT. This is the trick: the model sees an incomplete assistant turn and its job is to continue it. The first token it generates is the start of the answer.

### New concepts

- **BOS (Beginning Of Sequence)**: Token ID 128000. Signals "this is the start of a new conversation." Always the very first token.
- **EOT (End Of Turn)**: Token ID 128009. Signals "this speaker is done." During generation, when the model itself emits EOT, that's its way of saying "I'm finished answering."
- **Special tokens**: These IDs (128000+) live above the normal vocabulary. The BPE tokenizer never produces them from text — they're inserted programmatically by this function.

### Where in the code

The whole function is 18 lines (`inference_chat.cu:38-55`). It calls `tok.encode(prompt)` to get the BPE tokens for the user's text (line 40), then sandwiches them between the special-token bookends (lines 43-53). Pure CPU code — no GPU work here.

### TA-scrutiny items

None scored here, but understand that **M1 grading bypasses this function entirely**. The M1 tests call `encode()` directly on bare text. The chat template only matters for the end-to-end inference path (Milestones 2-3).

---

**TA-style question:**

The model was trained on conversations where every completed turn ends with `<|eot_id|>`. During generation, the model will eventually emit `<|eot_id|>` itself. What would happen if you accidentally closed the assistant turn too — added an EOT after the `\n\n` on line 53 — before feeding this to the model?

**answer**

If you closed the assistant turn with EOT, the model would see a complete conversation with an empty assistant reply — every turn opened and closed, nobody left talking. From the model's perspective, there's no incomplete turn to continue. It would likely start a *new* turn (maybe another user header, or random text), not answer your question. You'd get incoherent output instead of a reply.

The open assistant turn is what makes the model "fill in the blank." Close it and there's no blank to fill.

---
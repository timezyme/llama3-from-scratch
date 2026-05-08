---

## Step 3: BPE Tokenizer (Encode)

**File:** `src/tokenizer_bpe.cpp`
**Where in the pipeline:** Step 2 (chat template) produced a mix of special token IDs and raw prompt text. The raw text portion needs to be converted to token IDs. That's this step.

### High-level picture

BPE stands for **Byte Pair Encoding**. The idea: start by treating every byte of the input as its own token, then repeatedly merge the "best" adjacent pair into a single token. "Best" means the pair whose combined string has the lowest **rank** — the rank comes from training, where the most frequent pairs got merged first (rank 0 = most common merge, rank 1 = second most common, etc.).

Encoding happens in two phases:

**Phase 1 — Special token peeling** (`tokenizer_bpe.cpp:185-231`). Walk the text left-to-right. At each position, check if a special token like `<|begin_of_text|>` starts here (longest match first). If yes, emit its ID and skip past it. If no, scan forward to the next special token (or end of string) and hand that plain-text chunk to phase 2.

**Phase 2 — BPE merge loop** (`tokenizer_bpe.cpp:251-307`). For one plain-text chunk:
1. Split into individual bytes: `"Hi"` becomes `["H", "i"]` (line 256-257)
2. Scan all adjacent pairs, find the one with the lowest rank (lines 273-279)
3. Merge that pair in-place (line 286-287)
4. Repeat until no mergeable pair remains (line 281-283)
5. Convert the final merged strings to integer IDs (lines 296-305)

Example: `"Hello"` starts as `["H","e","l","l","o"]`. If `"ll"` has the lowest rank among all adjacent pairs, it merges first to give `["H","e","ll","o"]`. Then maybe `"He"` merges, then `"Hell"`, then `"Hello"` — until you reach a single token ID (or a few tokens if some pairs aren't in the vocabulary).

### Important detail: `encode()` does NOT prepend BOS

Line 26: `encode()` returns bare token IDs. The BOS token (128000) is added by callers — `apply_chat_template` for inference, or the `TestAPI` wrapper for M1 grading. This is a clean separation: the tokenizer converts text to IDs, and the caller decides what framing to add.

### New concept

- **Rank**: Each merge the BPE algorithm learned during training got a number. Rank 0 was the first pair it ever merged (the most common byte pair in the training data). At encode time, always merging the lowest-rank pair first replays the exact same merge order the model saw during training. If you merged in a different order, you'd get different token IDs, and the model would see input it was never trained on.

### TA-scrutiny items

- **Special tokens live above the regular vocab** (IDs 128000+). The BPE merge loop never produces them — they're inserted programmatically in phase 1 or by the chat template. If a TA asks "where does BOS come from," the answer is *not* the tokenizer's `encode()`.

---

**TA-style question:**

The BPE merge loop (lines 269-288) is O(n^2) — each merge scans all pairs, and there can be up to n-1 merges. Why is this acceptable for this project, and under what conditions would it become a problem?

**answer**

Tokenization runs once per prompt on the CPU, and prompts are capped at 1,000 tokens. Even at n=1,000, an O(n^2) scan is trivially fast — microseconds. The 32-layer GPU forward pass that follows takes seconds. Tokenization is never the bottleneck.

It would become a problem if you were tokenizing very long documents (hundreds of thousands of bytes) or tokenizing in a high-throughput server handling many requests per second. Production tokenizers (like HuggingFace's Rust-based `tokenizers` library) use a priority queue to track the best pair, bringing it down to O(n log n). But for a single-prompt inference engine with a 1,000-token ceiling, the quadratic loop is fine.

---
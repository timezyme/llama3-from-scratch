# Test 1 — Tokenize a short string

**Source:** `tests/test.cpp:119-140`
**Run:** `./bin/tests 1`

## What it does

Feeds the string `"Hello world"` through your tokenizer and checks the output
matches the exact list `[128000, 9906, 1917]`.

## Why it matters

An LLM doesn't read letters. It reads numbered pieces called **tokens**.
Before the model can do anything, you have to turn text into the right
sequence of token IDs. If this step is wrong, nothing downstream can work.

## What the numbers mean

- `128000` — the "beginning of sentence" (BOS) marker. Every input must
  start with this.
- `9906` — the token for `Hello`.
- `1917` — the token for ` world` (with a leading space).

## What a failure usually means

- You forgot to prepend the BOS token.
- You loaded the wrong tokenizer file.
- Your byte-pair-encoding merge loop is off.
- You're splitting the input into pieces incorrectly (e.g., not handling
  the leading space on ` world`).

## Why it's a good first test

The expected answer is **hardcoded** in the test — no fixture file, no
floating-point tolerance. If this fails, you know it's your tokenizer, not
the test infrastructure.

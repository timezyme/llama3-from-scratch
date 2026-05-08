# Test 2 — Tokenize a long sentence

**Source:** `tests/test.cpp:142-163`
**Fixture:** `tests/data/test2_tokenize.bin`
**Run:** `./bin/tests 2`

## What it does

Tokenizes a full sentence (~25 words of natural prose) and compares the
output to a pre-computed list of token IDs stored in a fixture file.

The sentence:

> first try give yourself a break don't be so hard on yourself first tries
> often fail and it is your first time living

## Why it matters

Test 1 is a sanity check. Test 2 is the real one. Natural text has all the
awkward cases:

- Apostrophes (`don't`)
- Repeated words that may tokenize differently in different positions
- Common words vs. rare words (which merge at different rates)
- Lots of spaces

If your tokenizer handles `"Hello world"` but fails here, you have a subtle
bug in the merge loop or pre-tokenization.

## How the fixture is stored

The file starts with a 4-byte integer `n` (the token count), then `n`
4-byte integers (the token IDs). The test reads both and compares them to
your output one-by-one.

## What a failure usually means

- Merge priorities (the "rank" table) are loaded in the wrong order.
- The regex that splits text before BPE is missing a case.
- Byte-level handling is wrong for a particular character.
- Off-by-one in BOS placement.

// Tokenizer interface and BPE implementation for Llama 3.

#pragma once

#include "prelude.h"

// Abstract tokenizer interface — encode text to token IDs and decode back.
class LLMTokenizer {
  public:
    virtual ~LLMTokenizer() = default;

    // Encode text to a sequence of token IDs.  e.g. "Hello world" -> [9906, 1917]
    virtual vector<int> encode(const string &text) const = 0;

    // Decode token IDs back to text.  e.g. [9906, 1917] -> "Hello world"
    virtual string decode(const vector<int> &ids) const = 0;

    virtual int bos_id() const = 0; // beginning-of-sequence token ID
    virtual int eos_id() const = 0; // end-of-sequence token ID
};

// Byte Pair Encoding tokenizer loaded from a vocab file (base64 + rank format).
// Supports special tokens (BOS, EOS, reserved) and greedy BPE merging.
class BPETokenizer final : public LLMTokenizer {
  public:
    unordered_map<string, int> rank;    // token string -> rank (also used as ID)
    vector<string> id2tok;              // rank/ID -> token string

    unordered_map<string, int> special2id;  // special token string -> ID
    unordered_map<int, string> id2special;  // ID -> special token string
    vector<string> specials_sorted;         // specials sorted longest-first for matching

    int bos_id_ = -1;
    int eos_id_ = -1;

    BPETokenizer();                        // disabled — throws (must provide path)
    BPETokenizer(const string &path);      // load vocab from file
    vector<int> encode(const string &text) const override;
    string decode(const vector<int> &ids) const override;
    int bos_id() const override;
    int eos_id() const override;

    // Encode without BPE merging — each byte becomes its own token.
    vector<int> encode_no_merge(const string &text) const;

  private:
    static string b64decode(const string &s); // base64 decoder for vocab file

    // Core encode: split on special tokens, then BPE-encode each chunk.
    vector<int> encode_impl(const string &text, bool enable_merge) const;

    // BPE-encode a single chunk (no special tokens) with greedy merging.
    vector<int> encode_chunk(const string &s, bool enable_merge) const;
};

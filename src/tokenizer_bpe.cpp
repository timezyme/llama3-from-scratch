// BPE (Byte Pair Encoding) tokenizer for Llama 3.
// Loads a vocabulary file of base64-encoded tokens with ranks,
// builds lookup tables, and supports encode/decode with special tokens.

#include "prelude.h"
#include "tokenizer.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstring>
#include <fstream>
#include <stdexcept>

// Encode text to token IDs with BPE merging enabled.
vector<int> BPETokenizer::encode(const string &text) const {
    return encode_impl(text, true);
}

// Encode text to token IDs without BPE merging (byte-level only).
vector<int> BPETokenizer::encode_no_merge(const string &text) const {
    return encode_impl(text, false);
}

// Decode token IDs back to a string, skipping special tokens.
string BPETokenizer::decode(const vector<int> &ids) const {
    string out;
    for (int id : ids) {
        if (id2special.count(id))
            continue; // skip special tokens like <|begin_of_text|>
        if (id >= 0 && id < static_cast<int>(id2tok.size()))
            out += id2tok[id];
    }
    return out;
}

int BPETokenizer::bos_id() const { return bos_id_; }
int BPETokenizer::eos_id() const { return eos_id_; }

// Default constructor is disallowed — must provide a vocab file path.
BPETokenizer::BPETokenizer() {
    throw std::runtime_error("BPETokenizer must be initialized with a path");
}

// Load vocabulary from file.
// File format: one token per line as "<base64-encoded token> <rank>"
// Rank doubles as the token ID in the vocabulary.
BPETokenizer::BPETokenizer(const string &path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot open " + path);

    // Parse each line: decode the base64 token string and read its rank.
    string line;
    vector<std::pair<string, int>> entries;
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        size_t sp = line.find_last_of(' ');
        if (sp == string::npos)
            continue;
        string tok = b64decode(line.substr(0, sp));
        int r = std::stoi(line.substr(sp + 1));
        entries.emplace_back(std::move(tok), r);
    }

    // Find the highest rank to size the id-to-token lookup table.
    size_t maxid = 0;
    for (auto &kv : entries)
        maxid = std::max(maxid, static_cast<size_t>(kv.second));

    // Build bidirectional mappings: token string <-> rank (ID).
    id2tok.assign(maxid + 1, string());
    for (auto &kv : entries) {
        rank[kv.first] = kv.second;
        id2tok[kv.second] = kv.first;
    }

    // Define the Llama 3 special tokens (BOS, EOS, reserved slots, etc.).
    vector<string> specials = {"<|begin_of_text|>",
                               "<|end_of_text|>",
                               "<|reserved_special_token_0|>",
                               "<|reserved_special_token_1|>",
                               "<|finetune_right_pad_id|>",
                               "<|step_id|>",
                               "<|start_header_id|>",
                               "<|end_header_id|>",
                               "<|eom_id|>",
                               "<|eot_id|>",
                               "<|python_tag|>"};

    // Fill remaining reserved slots up to 256 total special tokens.
    int reserved_total = 256;
    for (int i = 2; i < reserved_total - static_cast<int>(specials.size()) + 2;
         i++) {
        specials.push_back("<|reserved_special_token_" + std::to_string(i) +
                           "|>");
    }

    // Assign IDs to special tokens starting after the regular vocabulary.
    int next = static_cast<int>(id2tok.size());
    for (const auto &s : specials) {
        special2id[s] = next;
        id2special[next] = s;
        ++next;
    }

    bos_id_ = special2id.at("<|begin_of_text|>");
    eos_id_ = special2id.at("<|end_of_text|>");

    // Sort special tokens longest-first for greedy longest-match during encoding.
    for (const auto &kv : special2id)
        specials_sorted.push_back(kv.first);
    std::sort(
        specials_sorted.begin(), specials_sorted.end(),
        [](const string &a, const string &b) { return a.size() > b.size(); });
}

// Decode a base64-encoded string to raw bytes.
// Uses a static lookup table (initialized once) mapping ASCII chars to 6-bit values.
string BPETokenizer::b64decode(const string &s) {
    static int T[256];
    static bool init = false;
    if (!init) {
        std::fill(std::begin(T), std::end(T), -1);
        string a =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++)
            T[static_cast<unsigned char>(a[i])] = i;
        init = true;
    }

    string out;
    out.reserve(s.size() * 3 / 4);

    // Accumulate 6-bit chunks into bytes: shift in each char's value,
    // emit a byte whenever we have 8+ bits buffered.
    int val = 0, valb = -8;
    for (unsigned char c : s) {
        if (std::isspace(c))
            continue;
        if (c == '=')
            break; // padding signals end of data
        int d = T[c];
        if (d < 0)
            continue; // skip invalid chars
        val = (val << 6) + d;
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// Split text at special-token boundaries, then BPE-encode each normal chunk.
// Special tokens are matched greedily (longest first) and emitted directly.
vector<int> BPETokenizer::encode_impl(const string &text,
                                      bool enable_merge) const {
    vector<int> ids;
    size_t i = 0, n = text.size();

    while (i < n) {
        bool matched = false;

        // Try to match a special token at the current position.
        for (const auto &s : specials_sorted) {
            if (i + s.size() <= n &&
                std::memcmp(text.data() + i, s.data(), s.size()) == 0) {
                ids.push_back(special2id.at(s));
                i += s.size();
                matched = true;
                break;
            }
        }
        if (matched)
            continue;

        // No special token here — scan forward to find the next special token
        // (or end of string) to delimit a normal text chunk.
        size_t j = i + 1;
        while (j < n) {
            bool hit = false;
            for (const auto &sp : specials_sorted) {
                if (j + sp.size() <= n &&
                    std::memcmp(text.data() + j, sp.data(), sp.size()) == 0) {
                    hit = true;
                    break;
                }
            }
            if (hit)
                break;
            ++j;
        }

        // BPE-encode the normal text chunk between special tokens.
        string chunk = text.substr(i, j - i);
        auto v = encode_chunk(chunk, enable_merge);
        ids.insert(ids.end(), v.begin(), v.end());
        i = j;
    }

    return ids;
}

// Greedy BPE merge loop for a single text chunk (no special tokens).
// Starts with one token per byte, then repeatedly merges the adjacent pair
// with the lowest rank until no more merges are possible.
vector<int> BPETokenizer::encode_chunk(const string &s,
                                       bool enable_merge) const {
    // Initialize: each byte becomes its own token.
    vector<string> toks;
    toks.reserve(s.size());
    for (unsigned char c : s)
        toks.emplace_back(1, char(c));

    // Look up the rank of a candidate merge (concatenation of a and b).
    // Returns INT_MAX if the pair isn't in the vocabulary.
    auto pair_rank = [&](const string &a, const string &b) -> int {
        auto it = rank.find(a + b);
        return it == rank.end() ? INT_MAX : it->second;
    };

    if (enable_merge) {
        // Each iteration: find the lowest-ranked adjacent pair and merge it.
        // Stop when no mergeable pair remains (all pairs return INT_MAX).
        while (toks.size() > 1) {
            int best_rank = INT_MAX;
            size_t best_idx = 0;

            for (size_t i = 0; i + 1 < toks.size(); ++i) {
                int r = pair_rank(toks[i], toks[i + 1]);
                if (r < best_rank) {
                    best_rank = r;
                    best_idx = i;
                }
            }

            if (best_rank == INT_MAX) {
                break; // no more valid merges
            }

            // Merge: concatenate the pair in-place and remove the second element.
            toks[best_idx] += toks[best_idx + 1];
            toks.erase(toks.begin() + (best_idx + 1));
        }
    }

    // Convert merged token strings to their integer IDs.
    // If a multi-byte token isn't in the vocab (shouldn't happen after BPE),
    // fall back to encoding each byte individually.
    vector<int> ids;
    ids.reserve(toks.size());
    for (const auto &tkn : toks) {
        auto it = rank.find(tkn);
        if (it != rank.end()) {
            ids.push_back(it->second);
        } else {
            for (unsigned char c : tkn) {
                ids.push_back(rank.at(string(1, char(c))));
            }
        }
    }
    return ids;
}

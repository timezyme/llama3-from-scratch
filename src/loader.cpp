// Binary weight loader for llm_part1 §3.1.1 Step 3.
// tools/dumper.py emits one .bin per tensor: a 280-byte header followed
// by FP32, FP16, or BF16 payload bytes. The streaming path widens values
// to FP32; the resident-weight path can keep raw BF16 bits.

#include "loader.h"

#include "milifloat.h" // bf16_to_float(), half_to_float()

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {

// Dump-file header layout (280 bytes, all little-endian). This format
// is what tools/dumper.py writes; both producer and consumer must agree
// on it exactly. Strict validation here is a deliberate guardrail —
// silent header drift is the kind of bug that produces "model runs but
// outputs garbage", which is hard to debug after the fact.
//   [0..255]   tensor name (ASCII, null-padded)
//   [256..259] dtype code (uint32): 0=FP32, 1=FP16, 2=BF16
//   [260..263] ndims (uint32): 1 or 2
//   [264..271] shape[0] (uint64)
//   [272..279] shape[1] (uint64; 0 when ndims==1)
constexpr size_t kTensorNameBytes = 256;
constexpr size_t kHeaderSize = kTensorNameBytes + 4 + 4 + 8 + 8;

constexpr uint32_t kDtypeFP32 = 0;
constexpr uint32_t kDtypeFP16 = 1;
constexpr uint32_t kDtypeBF16 = 2;

// Parsed header fields from a dump file.
struct TensorHeader {
    string tensor_name;
    uint32_t dtype_code = 0;
    uint32_t ndims = 0;
    uint64_t shape0 = 0;
    uint64_t shape1 = 0;
};

// Detect host byte order. Dump files are always little-endian (every
// supported platform is LE), but the check stays explicit so big-endian
// hosts get a clear error rather than wrong numbers.
bool is_little_endian_host() {
    uint16_t x = 1;
    return *reinterpret_cast<uint8_t *>(&x) == 1;
}

// Return the byte width of a single element for the given dtype.
uint32_t bytes_per_element(uint32_t dtype_code) {
    switch (dtype_code) {
    case kDtypeFP32:
        return 4;
    case kDtypeFP16:
    case kDtypeBF16:
        return 2;
    default:
        throw runtime_error("unsupported dtype_code in dump header");
    }
}

// Read a little-endian uint32 from a byte pointer. memcpy is the
// alias-safe equivalent of *(const uint32_t*)p (which is UB on
// platforms where p is not 4-byte aligned). On big-endian hosts the
// bytes are reassembled manually so dump files load correctly anyway.
uint32_t read_u32_le(const uint8_t *p) {
    if (is_little_endian_host()) {
        uint32_t v = 0;
        std::memcpy(&v, p, sizeof(v));
        return v;
    }
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

// Read a little-endian uint64 from a byte pointer.
uint64_t read_u64_le(const uint8_t *p) {
    if (is_little_endian_host()) {
        uint64_t v = 0;
        std::memcpy(&v, p, sizeof(v));
        return v;
    }
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= (static_cast<uint64_t>(p[i]) << (8 * i));
    }
    return v;
}

// Multiply two size_t values, throwing on overflow. The lm_head tensor
// has 128256 * 4096 elements which fits in 32 bits comfortably, but the
// allocator checks all use the result in size_t arithmetic and a
// silent overflow on a malformed dump file would be hard to catch.
size_t checked_mul(size_t a, size_t b, const char *context) {
    if (a == 0 || b == 0) {
        return 0;
    }
    if (a > (std::numeric_limits<size_t>::max() / b)) {
        throw runtime_error(string("size overflow while computing ") + context);
    }
    return a * b;
}

size_t tensor_element_count(size_t dim0, size_t dim1, bool is_2d) {
    return is_2d ? checked_mul(dim0, dim1, "tensor element_count") : dim0;
}

void validate_shape(const TensorHeader &h, const string &dump_file,
                    size_t dim0, size_t dim1, bool is_2d) {
    if (is_2d) {
        if (h.ndims != 2 || static_cast<size_t>(h.shape0) != dim0 ||
            static_cast<size_t>(h.shape1) != dim1) {
            throw runtime_error("2D tensor shape mismatch in " + dump_file);
        }
    } else if (h.ndims != 1 || static_cast<size_t>(h.shape0) != dim0) {
        throw runtime_error("1D tensor shape mismatch in " + dump_file);
    }
}

void validate_file_size(const std::vector<uint8_t> &blob,
                        const TensorHeader &h, size_t count,
                        const string &dump_file) {
    const uint32_t elem_bytes = bytes_per_element(h.dtype_code);
    const size_t payload_bytes = checked_mul(
        count, static_cast<size_t>(elem_bytes), "payload bytes");
    if (blob.size() != kHeaderSize + payload_bytes) {
        throw runtime_error("file size does not match header metadata in " +
                            dump_file);
    }
}

// Parse the 280-byte header at the start of a dump-file blob into
// a TensorHeader struct. The tensor name occupies the first 256 bytes
// as a null-padded ASCII string; everything after is fixed-width LE
// integers (see kDtypeFP32/etc. above for the dtype codes).
TensorHeader parse_header(const std::vector<uint8_t> &blob) {
    if (blob.size() < kHeaderSize) {
        throw runtime_error("dump file too small to contain header");
    }
    TensorHeader h;

    // Read tensor name: scan for null terminator within the 256-byte field.
    const char *name_ptr = reinterpret_cast<const char *>(blob.data());
    size_t name_len = 0;
    while (name_len < kTensorNameBytes && name_ptr[name_len] != '\0') {
        ++name_len;
    }
    h.tensor_name.assign(name_ptr, name_len);

    // Read metadata fields immediately after the name.
    const uint8_t *meta = blob.data() + kTensorNameBytes;
    h.dtype_code = read_u32_le(meta);
    h.ndims = read_u32_le(meta + 4);
    h.shape0 = read_u64_le(meta + 8);
    h.shape1 = read_u64_le(meta + 16);
    return h;
}

// Read an entire file into a byte vector (seek to end for size, then read).
std::vector<uint8_t> read_file_binary(const string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw runtime_error("failed to open dump file: " + path);
    }
    in.seekg(0, std::ios::end);
    std::streamoff end = in.tellg();
    if (end < 0) {
        throw runtime_error("failed to read file size: " + path);
    }
    std::vector<uint8_t> data(static_cast<size_t>(end));
    in.seekg(0, std::ios::beg);
    if (!data.empty()) {
        in.read(reinterpret_cast<char *>(data.data()),
                static_cast<std::streamsize>(data.size()));
        if (!in) {
            throw runtime_error("failed to read dump file bytes: " + path);
        }
    }
    return data;
}

// Decode one element from the payload to FP32 based on the dtype code.
// FP32 is a straight memcpy. FP16 and BF16 widen via the converters in
// milifloat.h. memcpy (rather than reinterpret_cast) keeps this
// alias-safe regardless of payload alignment.
float decode_value(const uint8_t *ptr, uint32_t dtype_code) {
    switch (dtype_code) {
    case kDtypeFP32: {
        float v = 0.0f;
        std::memcpy(&v, ptr, sizeof(v));
        return v;
    }
    case kDtypeFP16: {
        uint16_t bits = 0;
        std::memcpy(&bits, ptr, sizeof(bits));
        return half_to_float(bits); // IEEE 754 half -> float
    }
    case kDtypeBF16: {
        uint16_t bits = 0;
        std::memcpy(&bits, ptr, sizeof(bits));
        return bf16_to_float(bits); // bfloat16 -> float
    }
    default:
        throw runtime_error("unsupported dtype_code while decoding tensor");
    }
}

// Load a 1D or 2D tensor with full shape and size validation, then
// decode every element to FP32. Returns a heap-allocated FP32 array
// (caller owns it via delete[]).
float_t *load_dense_tensor_checked(const string &dump_file, size_t dim0,
                                   size_t dim1, bool is_2d) {
    std::vector<uint8_t> blob = read_file_binary(dump_file);
    TensorHeader h = parse_header(blob);

    validate_shape(h, dump_file, dim0, dim1, is_2d);
    const size_t count = tensor_element_count(dim0, dim1, is_2d);
    validate_file_size(blob, h, count, dump_file);

    // Decode every element from its native dtype to FP32.
    const uint8_t *payload = blob.data() + kHeaderSize;
    const uint32_t elem_bytes = bytes_per_element(h.dtype_code);
    std::unique_ptr<float_t[]> out(new float_t[count]);
    for (size_t i = 0; i < count; ++i) {
        out[i] = decode_value(payload + i * elem_bytes, h.dtype_code);
    }
    return out.release();
}

// Load a BF16-only tensor without widening to FP32. Used by the
// resident-VRAM (video RAM) path so weights stay half precision in GPU memory.
// This halves weight HBM (high-bandwidth memory) bytes during matmul.
// Falls back to a clear error if the dump dtype is not BF16.
std::vector<uint16_t> load_bf16_raw_tensor_checked(const string &dump_file,
                                                   size_t dim0, size_t dim1,
                                                   bool is_2d) {
    if (!is_little_endian_host()) {
        throw runtime_error(
            "unsupported big-endian host for raw BF16 dump payload");
    }

    std::vector<uint8_t> blob = read_file_binary(dump_file);
    TensorHeader h = parse_header(blob);
    validate_shape(h, dump_file, dim0, dim1, is_2d);
    if (h.dtype_code != kDtypeBF16) {
        throw runtime_error("raw BF16 loader expected BF16 dtype in " +
                            dump_file);
    }

    const size_t count = tensor_element_count(dim0, dim1, is_2d);
    validate_file_size(blob, h, count, dump_file);

    std::vector<uint16_t> out(count);
    const uint8_t *payload = blob.data() + kHeaderSize;
    std::memcpy(out.data(), payload, count * sizeof(uint16_t));
    return out;
}
} // namespace

// --- Public LlamaDumpLoader methods ---

LlamaDumpLoader::LlamaDumpLoader(DumpFloatType float_type)
    : float_type(float_type) {}

LlamaDumpLoader::~LlamaDumpLoader() = default;

// Read just the embedding dump's vocab size (= shape[0]) by loading
// and caching the file. Subsequent calls and later get_embeddings()
// reuse the cached blob instead of re-reading the file.
size_t LlamaDumpLoader::vocab_size(const std::string &dump_path,
                                   int embedding_dim) {
    if (embedding_dim <= 0) {
        throw runtime_error("embedding_dim must be > 0");
    }

    // Return the cached value if this file has already been loaded.
    if (embeddings_source_file_ == dump_path &&
        embeddings_dim_ == embedding_dim && !embeddings_blob_.empty()) {
        return embeddings_vocab_size_;
    }

    if (!load_embeddings(dump_path, embedding_dim)) {
        throw runtime_error("failed to load embeddings while reading vocab size");
    }
    return embeddings_vocab_size_;
}

// Load the entire embedding dump into memory and cache the blob.
//
// We do not eagerly decode all 128256 rows to FP32. The raw payload stays
// in memory and get_embeddings() decodes only requested token rows, which
// matches the embedding-lookup pattern from llm_part1 §3.1.1 Step 4.
// Returns false on header/shape mismatch so callers can surface a clean error.
bool LlamaDumpLoader::load_embeddings(const std::string &dump_path,
                                      int embedding_dim) {
    if (embedding_dim <= 0) {
        return false;
    }
    if (!is_little_endian_host()) {
        throw runtime_error(
            "unsupported big-endian host for little-endian dump format");
    }

    std::vector<uint8_t> blob = read_file_binary(dump_path);
    TensorHeader h = parse_header(blob);
    if (h.ndims != 2) {
        return false;
    }

    if (static_cast<int>(h.shape1) != embedding_dim) {
        return false;
    }

    // Validate that file size matches header-declared shape.
    const size_t vocab = static_cast<size_t>(h.shape0);
    const size_t dim = static_cast<size_t>(h.shape1);
    const uint32_t elem_bytes = bytes_per_element(h.dtype_code);
    const size_t payload_bytes = checked_mul(
        checked_mul(vocab, dim, "embedding element_count"),
        static_cast<size_t>(elem_bytes), "embedding payload bytes");

    if (blob.size() != kHeaderSize + payload_bytes) {
        return false;
    }

    // Cache the blob and metadata for later row lookups.
    embeddings_blob_ = std::move(blob);
    embeddings_payload_offset_ = kHeaderSize;
    embeddings_vocab_size_ = vocab;
    embeddings_dim_ = embedding_dim;
    embeddings_dtype_code_ = h.dtype_code;
    embeddings_source_file_ = dump_path;
    return true;
}

// Embedding lookup (Milestone 1 Step 4): given a sequence of token IDs,
// produce the [s, d] FP32 embedding matrix that feeds the first
// decoder block. Each row is gathered from the cached payload at
// offset `token_id * row_bytes` and decoded to FP32.
//
// Result is heap-allocated; caller owns it and must delete[].
float_t *LlamaDumpLoader::get_embeddings(const std::vector<int> &token_ids) {
    if (embeddings_blob_.empty()) {
        throw runtime_error(
            "embeddings are not loaded; call load_embeddings() first");
    }

    const size_t dim = static_cast<size_t>(embeddings_dim_);
    const size_t total = checked_mul(token_ids.size(), dim,
                                     "token_ids.size() * embedding_dim");
    std::unique_ptr<float_t[]> out(new float_t[total]);

    const uint8_t *payload = embeddings_blob_.data() + embeddings_payload_offset_;
    const size_t row_bytes =
        checked_mul(dim, static_cast<size_t>(bytes_per_element(embeddings_dtype_code_)),
                    "embedding row bytes");

    // For each token ID, seek to its row in the payload and decode every element.
    for (size_t row = 0; row < token_ids.size(); ++row) {
        int token_id = token_ids[row];
        if (token_id < 0 ||
            static_cast<size_t>(token_id) >= embeddings_vocab_size_) {
            throw runtime_error("token id out of embedding vocab range");
        }

        const uint8_t *src =
            payload + static_cast<size_t>(token_id) * row_bytes;
        for (size_t col = 0; col < dim; ++col) {
            out[row * dim + col] = decode_value(
                src + col * bytes_per_element(embeddings_dtype_code_),
                embeddings_dtype_code_);
        }
    }
    return out.release();
}

// Convenience: load a 1D tensor (e.g., bias, norm weights) with shape validation.
float_t *LlamaDumpLoader::load_1d(const std::string &dump_file, size_t dim0) {
    return load_dense_tensor_checked(dump_file, dim0, 0, false);
}

// Convenience: load a 2D tensor (e.g., weight matrix) with shape validation.
float_t *LlamaDumpLoader::load_2d(const std::string &dump_file, size_t dim0,
                                  size_t dim1) {
    return load_dense_tensor_checked(dump_file, dim0, dim1, true);
}

std::vector<uint16_t>
LlamaDumpLoader::load_1d_bf16_raw(const std::string &dump_file, size_t dim0) {
    return load_bf16_raw_tensor_checked(dump_file, dim0, 0, false);
}

std::vector<uint16_t>
LlamaDumpLoader::load_2d_bf16_raw(const std::string &dump_file, size_t dim0,
                                  size_t dim1) {
    return load_bf16_raw_tensor_checked(dump_file, dim0, dim1, true);
}

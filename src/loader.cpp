// Binary weight loader for Llama 3 model dumps.
// Reads tensor files produced by tools/dumper.py (safetensors -> binary format).
// Each dump file has a fixed 280-byte header followed by a raw payload in
// FP32, FP16, or BF16. This loader parses the header, validates shapes,
// and converts all values to FP32 for use in the inference pipeline.

#include "loader.h"

#include "milifloat.h" // bf16_to_float(), half_to_float()

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {

// Dump file header layout (280 bytes total):
//   [0..255]   tensor name (null-padded ASCII)
//   [256..259] dtype code (uint32 LE): 0=FP32, 1=FP16, 2=BF16
//   [260..263] ndims (uint32 LE): 1 or 2
//   [264..271] shape[0] (uint64 LE)
//   [272..279] shape[1] (uint64 LE, 0 for 1D tensors)
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

// Check host byte order — dump files are always little-endian.
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

// Read a little-endian uint32 from a byte pointer.
// Fast path: memcpy on LE hosts. Manual byte assembly on BE hosts.
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

// Multiply two sizes with overflow detection. Throws on overflow.
size_t checked_mul(size_t a, size_t b, const char *context) {
    if (a == 0 || b == 0) {
        return 0;
    }
    if (a > (std::numeric_limits<size_t>::max() / b)) {
        throw runtime_error(string("size overflow while computing ") + context);
    }
    return a * b;
}

// Parse the 280-byte header from the front of a dump file blob.
// Extracts the tensor name (null-terminated within 256 bytes), dtype,
// number of dimensions, and shape.
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

// Convert a single raw element (FP32/FP16/BF16) to a float.
// Uses memcpy to avoid strict-aliasing issues with reinterpret_cast.
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

// Load a 1D or 2D tensor from a dump file, validating that the header's
// shape matches the expected dimensions. Returns a heap-allocated FP32 array
// (caller takes ownership).
float_t *load_dense_tensor_checked(const string &dump_file, size_t dim0,
                                   size_t dim1, bool is_2d) {
    std::vector<uint8_t> blob = read_file_binary(dump_file);
    TensorHeader h = parse_header(blob);

    // Validate shape matches what the caller expects.
    if (is_2d) {
        if (h.ndims != 2 || static_cast<size_t>(h.shape0) != dim0 ||
            static_cast<size_t>(h.shape1) != dim1) {
            throw runtime_error("2D tensor shape mismatch in " + dump_file);
        }
    } else {
        if (h.ndims != 1 || static_cast<size_t>(h.shape0) != dim0) {
            throw runtime_error("1D tensor shape mismatch in " + dump_file);
        }
    }

    // Verify file size = header + expected payload bytes.
    const size_t count =
        is_2d ? checked_mul(dim0, dim1, "tensor element_count") : dim0;
    const uint32_t elem_bytes = bytes_per_element(h.dtype_code);
    const size_t payload_bytes = checked_mul(count, static_cast<size_t>(elem_bytes),
                                             "payload bytes");
    if (blob.size() != kHeaderSize + payload_bytes) {
        throw runtime_error("file size does not match header metadata in " +
                            dump_file);
    }

    // Decode every element from its native dtype to FP32.
    const uint8_t *payload = blob.data() + kHeaderSize;
    std::unique_ptr<float_t[]> out(new float_t[count]);
    for (size_t i = 0; i < count; ++i) {
        out[i] = decode_value(payload + i * elem_bytes, h.dtype_code);
    }
    return out.release();
}
} // namespace

// --- Public LlamaDumpLoader methods ---

LlamaDumpLoader::LlamaDumpLoader(DumpFloatType float_type)
    : float_type(float_type) {}

LlamaDumpLoader::~LlamaDumpLoader() = default;

// Return the vocab size (number of rows) from the embedding dump file.
// Caches the loaded blob so repeated calls with the same path are free.
size_t LlamaDumpLoader::vocab_size(const std::string &dump_path,
                                   int embedding_dim) {
    if (embedding_dim <= 0) {
        throw runtime_error("embedding_dim must be > 0");
    }

    // Return cached value if we already loaded this file.
    if (embeddings_source_file_ == dump_path &&
        embeddings_dim_ == embedding_dim && !embeddings_blob_.empty()) {
        return embeddings_vocab_size_;
    }

    if (!load_embeddings(dump_path, embedding_dim)) {
        throw runtime_error("failed to load embeddings while reading vocab size");
    }
    return embeddings_vocab_size_;
}

// Load the embedding table dump into memory, caching the raw blob
// so get_embeddings() can decode rows on demand without re-reading the file.
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

// Look up embedding vectors for a list of token IDs.
// Returns a heap-allocated FP32 array of shape [token_ids.size(), embedding_dim].
// Decodes each row from the cached blob's native dtype (BF16/FP16/FP32) to FP32.
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

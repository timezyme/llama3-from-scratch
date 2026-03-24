

// DO NOT CHANGE THIS FILE, THIS IS FOR OUR TESTING PURPOSES ONLY
// But We include a sample test here for you to see how the we do the final
// testing We use functions in API to implement our tests
#include "config.h"
#include "test_api.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>

const float EPSILON = 1e-2;

const char *DATA_DIR = "tests/data";
const char *SENTENCE =
    "first try give yourself a break don't be so hard on yourself first tries "
    "often fail and it is your first time living";

// -----------------------------------------------------------------------------
// Binary loader helpers
// -----------------------------------------------------------------------------

static std::vector<int> read_int_array(std::ifstream &f, int n) {
    std::vector<int> v(n);
    f.read(reinterpret_cast<char *>(v.data()), n * sizeof(int));
    return v;
}

static std::vector<float> read_float_array(std::ifstream &f, int n) {
    std::vector<float> v(n);
    f.read(reinterpret_cast<char *>(v.data()), n * sizeof(float));
    return v;
}

struct TokenizeFixture {
    std::vector<int> token_ids;
};

struct EmbeddingFixture {
    std::vector<int> token_ids;
    std::vector<float> embeddings; // [num_tokens * EMBEDDING_DIM]
};

struct MatmulFixture {
    int M, K, N;
    std::vector<float> A, B, C;
};

static TokenizeFixture load_tokenize(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error(std::string("Cannot open ") + path);
    int n;
    f.read(reinterpret_cast<char *>(&n), sizeof(int));
    return {read_int_array(f, n)};
}

static EmbeddingFixture load_embedding(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error(std::string("Cannot open ") + path);
    int n;
    f.read(reinterpret_cast<char *>(&n), sizeof(int));
    auto ids = read_int_array(f, n);
    auto emb = read_float_array(f, n * EMBEDDING_DIM);
    return {ids, emb};
}

static MatmulFixture load_matmul(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error(std::string("Cannot open ") + path);
    MatmulFixture fix;
    f.read(reinterpret_cast<char *>(&fix.M), sizeof(int));
    f.read(reinterpret_cast<char *>(&fix.K), sizeof(int));
    f.read(reinterpret_cast<char *>(&fix.N), sizeof(int));
    fix.A = read_float_array(f, fix.M * fix.K);
    fix.B = read_float_array(f, fix.K * fix.N);
    fix.C = read_float_array(f, fix.M * fix.N);
    return fix;
}

// -----------------------------------------------------------------------------
// Comparison helper
// -----------------------------------------------------------------------------

static bool check_max_abs(const std::vector<float> &got,
                          const std::vector<float> &expected, float epsilon) {
    if (got.size() != expected.size()) {
        std::cout << "  size mismatch: got " << got.size() << " expected "
                  << expected.size() << "\n";
        return false;
    }
    float max_err = 0.0f;
    int worst = 0;
    for (int i = 0; i < (int)got.size(); i++) {
        float err = std::fabs(got[i] - expected[i]);
        if (err > max_err) {
            max_err = err;
            worst = i;
        }
    }
    if (max_err > epsilon) {
        std::cout << "  max |err|=" << max_err << " at index " << worst
                  << " (got=" << got[worst] << " expected=" << expected[worst]
                  << ") epsilon=" << epsilon << "\n";
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

bool test_1() {
    TestAPI api;

    string input = "Hello world";
    vector<int> token_ids = api.tokenize(input);
    vector<int> expected = {128000, 9906, 1917};

    if (token_ids.size() != expected.size()) {
        std::cout << "Test failed: size mismatch. Expected " << expected.size()
                  << " but got " << token_ids.size() << "\n";
        return false;
    }

    for (int i = 0; i < (int)token_ids.size(); i++) {
        if (token_ids[i] != expected[i]) {
            std::cout << "Test failed: element mismatch. Expected "
                      << expected[i] << " but got " << token_ids[i] << "\n";
            return false;
        }
    }
    return true;
}

bool test_2() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test2_tokenize.bin", DATA_DIR);
    auto fix = load_tokenize(path);

    TestAPI api;
    auto got = api.tokenize(SENTENCE);

    if (got.size() != fix.token_ids.size()) {
        std::cout << "  size mismatch: got " << got.size() << " expected "
                  << fix.token_ids.size() << "\n";
        return false;
    }
    for (int i = 0; i < (int)got.size(); i++) {
        if (got[i] != fix.token_ids[i]) {
            std::cout << "  mismatch at position " << i << ": got " << got[i]
                      << " expected " << fix.token_ids[i] << "\n";
            return false;
        }
    }
    return true;
}

bool test_3() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test3_embedding.bin", DATA_DIR);
    auto fix = load_embedding(path);

    TestAPI api;
    auto got = api.get_embeddings(fix.token_ids);

    return check_max_abs(got, fix.embeddings, EPSILON);
}

bool test_4() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test4_embedding.bin", DATA_DIR);
    auto fix = load_embedding(path);

    TestAPI api;
    auto got = api.get_embeddings(fix.token_ids);

    return check_max_abs(got, fix.embeddings, EPSILON);
}

static bool run_matmul_test(const char *bin_path, const char *label) {
    auto fix = load_matmul(bin_path);

    TestAPI api;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto got = api.matmul(fix.A, fix.B, fix.M, fix.K, fix.N);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  " << label << " M=" << fix.M << " K=" << fix.K
              << " N=" << fix.N << "  time=" << ms << "ms\n";

    return check_max_abs(got, fix.C, EPSILON);
}

bool test_5() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test5_matmul.bin", DATA_DIR);
    return run_matmul_test(path, "seq_len=1  [first]");
}

bool test_6() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test6_matmul.bin", DATA_DIR);
    return run_matmul_test(path, "seq_len=10 ");
}

bool test_7() {
    char path[256];
    std::snprintf(path, sizeof(path), "%s/test7_matmul.bin", DATA_DIR);
    return run_matmul_test(path, "seq_len=100 [last] ");
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    const char *GREEN = "\033[32m";
    const char *RED = "\033[31m";
    const char *RESET = "\033[0m";

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <test_id>\n";
        return 1;
    }

    int test_id = 0;
    try {
        test_id = std::stoi(argv[1]);
    } catch (...) {
        std::cout << RED << "Invalid test id: " << argv[1] << RESET << "\n";
        return 2;
    }

    bool ok = false;
    try {
        switch (test_id) {
        case 1:
            ok = test_1();
            break;
        case 2:
            ok = test_2();
            break;
        case 3:
            ok = test_3();
            break;
        case 4:
            ok = test_4();
            break;
        case 5:
            ok = test_5();
            break;
        case 6:
            ok = test_6();
            break;
        case 7:
            ok = test_7();
            break;
        default:
            std::cout << RED << "Unknown test id: " << test_id << RESET << "\n";
            return 2;
        }
    } catch (const std::exception &e) {
        std::cout << RED << "Test threw: " << e.what() << RESET << "\n";
        ok = false;
    }

    std::cout << (ok ? GREEN : RED) << (ok ? "PASSED" : "FAILED") << RESET
              << "\n";
    return ok ? 0 : 3;
}

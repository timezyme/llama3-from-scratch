// Test driver for the M2-3 internal test binary.
//
// argv[1] = test name (or `--list`). Run exactly one named test and
// return its exit code. The test driver script iterates through these
// one at a time so a failure in one test doesn't suppress the others.
//
// Exit codes:
//   0 = test passed
//   2 = invalid usage or unknown test name
//   3 = test ran but failed
//
// Usage:
//   ./bin/tests_m2m3 <test_name>     # run one test
//   ./bin/tests_m2m3 --list          # print every test name and exit
//
// The split puts each milestone phase in its own TU, with that TU's
// register_phaseN() registering its own tests. Phases roughly mirror the
// assignment milestones:
//   Phase 0: matmul/loader smoke and parity (foundation laid in M1)
//   Phase 1: RMSNorm + Q/K/V projections (M2)
//   Phase 2: RoPE, GQA (Grouped Query Attention), causal mask, softmax (M3)
//   Phase 3: residual add, SwiGLU, full decoder block + Phase 4 forward (M3)
//   Phase 5: final norm, untied lm_head, layer streaming, KV cache,
//            B>1 batched parity (M3 plus optional extensions)

#include "tests/test_m2m3_helpers.h"

static Registry build_registry() {
    Registry r;
    register_phase0(r);
    register_phase1(r);
    register_phase2(r);
    register_phase3(r);  // includes Phase 3 kernel smoke + Phase 4
    register_phase5(r);
    return r;
}

int main(int argc, char *argv[]) {
    auto registry = build_registry();

    if (argc == 2 && std::string(argv[1]) == "--list") {
        for (const auto &entry : registry) {
            std::printf("  %s\n", entry.first.c_str());
        }
        return PASS;
    }

    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <test_name>\n", argv[0]);
        std::fprintf(stderr, "       %s --list\n", argv[0]);
        std::fprintf(stderr, "\nAvailable tests:\n");
        for (const auto &entry : registry) {
            std::fprintf(stderr, "  %s\n", entry.first.c_str());
        }
        return USAGE_ERROR;
    }

    std::string name = argv[1];
    auto it = registry.find(name);
    if (it == registry.end()) {
        std::fprintf(stderr, "Unknown test: %s\n", name.c_str());
        std::fprintf(stderr, "\nAvailable tests:\n");
        for (const auto &entry : registry) {
            std::fprintf(stderr, "  %s\n", entry.first.c_str());
        }
        return USAGE_ERROR;
    }

    return it->second();
}

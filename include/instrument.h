// Lightweight instrumentation helpers for diagnosing where time and
// memory go inside the forward pass. Header-only, no .cu unit.
//
// Usage:
//   {
//       Stopwatch sw("layer.load");
//       weights.load_layer(layer);
//   }   // prints elapsed ms when scope ends, accumulates into named bucket
//   ...
//   Stopwatch::print_summary();  // dump per-name aggregates
//
// VRAM probe:
//   probe_vram("startup");       // logs free/total VRAM with a label

#pragma once

#include <chrono>
#include <cstdio>
#include <map>
#include <string>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

class Stopwatch {
public:
    explicit Stopwatch(const char *name) : name_(name) {
        start_ = std::chrono::steady_clock::now();
    }

    ~Stopwatch() {
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        auto &agg = aggregates()[name_];
        agg.count += 1;
        agg.total_ms += ms;
        if (ms < agg.min_ms || agg.count == 1) agg.min_ms = ms;
        if (ms > agg.max_ms) agg.max_ms = ms;
    }

    static void reset() { aggregates().clear(); }

    static void print_summary() {
        const auto &m = aggregates();
        if (m.empty()) {
            std::printf("[telemetry] no timers recorded\n");
            return;
        }
        std::printf("\n[telemetry] timer summary (ms):\n");
        std::printf("  %-32s %6s %10s %10s %10s %10s\n",
                    "name", "count", "total", "avg", "min", "max");
        for (const auto &kv : m) {
            const auto &a = kv.second;
            double avg = a.count ? a.total_ms / a.count : 0.0;
            std::printf("  %-32s %6lld %10.2f %10.3f %10.3f %10.3f\n",
                        kv.first.c_str(), (long long)a.count, a.total_ms,
                        avg, a.min_ms, a.max_ms);
        }
    }

private:
    struct Aggregate {
        long long count = 0;
        double total_ms = 0.0;
        double min_ms = 0.0;
        double max_ms = 0.0;
    };

    static std::map<std::string, Aggregate> &aggregates() {
        static std::map<std::string, Aggregate> m;
        return m;
    }

    const char *name_;
    std::chrono::steady_clock::time_point start_;
};

#ifdef CUDA_ENABLED
inline void probe_vram(const char *label) {
    size_t free_b = 0, total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        std::printf("[telemetry][vram] %s: cudaMemGetInfo failed: %s\n",
                    label, cudaGetErrorString(err));
        return;
    }
    const double GiB = 1073741824.0;
    std::printf("[telemetry][vram] %s: free=%.2f GiB / total=%.2f GiB\n",
                label, free_b / GiB, total_b / GiB);
}
#else
inline void probe_vram(const char *) {}
#endif

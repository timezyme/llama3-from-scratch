// ============================================================================
// matmul.cu — Tiled GEMM (general matrix multiply): C = A * B.
// ============================================================================
//
// What it does: multiplies two large matrices on the GPU, fast. Workhorse
// of every per-layer projection in the forward pass (Q, K, V, O, gate, up,
// down all reduce to a call into one of the entry points below). The
// final lm_head projection runs on the CPU instead (see
// compute_lm_head_logits in src/inference_layer.cu) — it only needs the
// last token's hidden vector, so the GEMM tiling here is overkill for
// that one [V, d] * [d, 1] product. Two kernels live here: FP32 weights
// (matmul_kernel) and BF16 weights widened to FP32 inline
// (matmul_bf16_weight_kernel). Activations are always FP32, accumulation
// is always FP32; only the weight storage dtype differs.
//
// Why it's fast: each weight byte is loaded from slow HBM (high-bandwidth
// memory) once and reused many times from fast on-chip shared memory.
// Without this trick, every thread would re-read the same row of A from
// HBM thousands of times — we'd be memory-bandwidth-bound, not compute-
// bound, and the kernel would run an order of magnitude slower.
//
// Read the file top-to-bottom — the layout matches execution flow:
//   Section 1: Tile-size constants (BM, BN, BK, TM, TN) — the design choices
//   Section 2: Small helpers (CUDA error wrap, BF16 -> FP32 widening)
//   Section 3: matmul_kernel              — the FP32 workhorse, 4 phases A-D
//   Section 4: matmul_bf16_weight_kernel  — same kernel, BF16 weights inline
//   Section 5: gpu_matmul                 — host entry for the M1 grader
//   Section 6: gpu_matmul_device          — device-pointer entry (forward pass)
//   Section 7: gpu_matmul_device_bf16_weight — BF16 forward-pass entry
//
// Credit: the block-tiled / shared-memory-staged pattern follows the
// canonical example in NVIDIA's CUDA Programming Guide (Chapter 3,
// "Shared Memory") and the cuda-samples matrixMul reference. The
// optimizations layered on top — double-buffered shared tiles, 8x8
// per-thread register accumulation, float4/BF16 vectorized loads, and
// +1 shared-memory padding to avoid bank conflicts — are this project's
// choices and are explained inline at their first use below.
//
// Tensor cores are NOT used: that would require WMMA or mma.sync intrinsics
// and stricter tile shapes than the ones we picked.
//
// Glossary (used throughout the comments below):
//   HBM  — GPU global DRAM. Big (24 GB on L4), slow (~300 ns).
//   Shared memory — per-block on-chip scratchpad. Tiny, fast (~10 ns).
//   FMA  — fused multiply-add (one-cycle a*b+c on every CUDA core).
//   SM   — streaming multiprocessor. The GPU's "core"; runs one block.
// ============================================================================

#include "kernel/kernels.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace {

// ----------------------------------------------------------------------------
// Section 1 — Tile-size constants. The numbers below are deliberate: each
// is justified by a reuse-math comment that explains how many FMAs every
// loaded element feeds. Read the comments before the numbers.
// ----------------------------------------------------------------------------

// Block tile dimensions — each thread block owns a BM x BN region of C
// and walks K in BK-wide tiles. 128x128 was chosen because during one
// rank-1 outer-product step each smA element (one row, one K-depth) is
// reused by every output column in the block (BN = 128 reuses), and each
// smB element is reused by every output row (BM = 128 reuses). BK = 16
// keeps two double-buffered shared tiles (smA + smB, including the +1
// padding) within typical per-block shared-memory budgets.
constexpr int BM = 128;   // rows of C per block
constexpr int BN = 128;   // cols of C per block
constexpr int BK = 16;    // K-depth of each loaded tile

// Thread tile dimensions — each thread accumulates TM x TN = 64 output
// elements in registers. 8x8 keeps register pressure tolerable while
// raising arithmetic intensity (one smA/smB load services 8*8=64 FMAs).
constexpr int TM = 8;
constexpr int TN = 8;

// Thread block shape: 16x16 = 256 threads per block.
constexpr int BLOCK_X = BN / TN;              // 16
constexpr int BLOCK_Y = BM / TM;             // 16
constexpr int NUM_THREADS = BLOCK_X * BLOCK_Y; // 256

// Cooperative load counts — all 256 threads participate in loading tiles.
// smA: BM*BK = 2048 floats / 256 threads = 8 scalar loads per thread.
constexpr int SMA_LOADS_PER_THREAD = (BM * BK) / NUM_THREADS; // 8
// smB: BK*BN = 2048 floats / 256 threads = 8 floats = 2 float4 loads per thread.
constexpr int SMB_F4_PER_THREAD = (BK * BN) / (NUM_THREADS * 4); // 2
constexpr int SMB_F4_PER_ROW = BN / 4; // 32 float4s per row of smB

// ----------------------------------------------------------------------------
// Section 2 — Small helpers. A CUDA error wrapper that turns a status code
// into a thrown exception, plus a BF16 -> FP32 widener that runs on the
// device. Nothing exotic; both are used by the kernels below.
// ----------------------------------------------------------------------------

// Format a CUDA error with source location and throw as runtime_error.
void throw_cuda_error(cudaError_t err, const char *expr, const char *file,
                      int line) {
    if (err == cudaSuccess) {
        return;
    }
    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << " for " << expr << ": "
        << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
}

// BF16 (bfloat16) shares FP32's 8-bit exponent, so widening to FP32 is
// a 16-bit left shift into the high half of a float. __uint_as_float
// reinterprets those bits on the device.
__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
    return __uint_as_float(static_cast<unsigned int>(bits) << 16);
}

// Load 4 BF16 values from src and widen each to FP32 in dst.
// One uint2 (8 bytes) is loaded per call, so the compiler can use an
// aligned 64-bit load instead of four separate 16-bit loads.
__device__ __forceinline__ void load_bf16_quad(float *dst,
                                               const uint16_t *src) {
    uint2 packed = *reinterpret_cast<const uint2 *>(src);
    dst[0] = bf16_bits_to_float(static_cast<uint16_t>(packed.x & 0xffffu));
    dst[1] = bf16_bits_to_float(static_cast<uint16_t>(packed.x >> 16));
    dst[2] = bf16_bits_to_float(static_cast<uint16_t>(packed.y & 0xffffu));
    dst[3] = bf16_bits_to_float(static_cast<uint16_t>(packed.y >> 16));
}

} // namespace

#define CUDA_CHECK(expr) throw_cuda_error((expr), #expr, __FILE__, __LINE__)

// ============================================================================
// Section 3 — matmul_kernel: the FP32 workhorse.
// ============================================================================
//
// One thread block computes one BM x BN (= 128 x 128) output tile of C and
// walks across the K dimension in BK-wide (= 16) slabs. Inside the block,
// 256 threads cooperate. Each thread owns a TM x TN (= 8 x 8 = 64) sub-tile
// of C and holds its 64 running sums in registers.
//
// Phase outline — the labels are repeated inline below so a top-to-bottom
// read of the function lets the comments narrate each step of the code:
//   PHASE A — Allocate shared memory and zero the register accumulator.
//   PHASE B — Prefetch K-tile 0 so the main loop has data on iteration 0.
//   PHASE C — Main loop over K-tiles (double-buffered):
//               (1) prefetch the NEXT tile into the alternate buffer,
//               (2) compute on the CURRENT tile (rank-1 outer products
//                   into the register accumulator),
//               (3) __syncthreads(), then swap buffers.
//             The SM overlaps the load in (1) with the FMAs in (2)
//             because they read/write different shared-memory buffers.
//   PHASE D — Write each thread's 64 accumulators back to C.
// ============================================================================
__global__ void matmul_kernel(const float *A, const float *B, float *C, int M,
                              int K, int N) {
    // ---- PHASE A: shared-memory buffers + register accumulator ----------
    // Two buffers per matrix so we can ping-pong (double-buffering). The
    // leading [2] is the buffer index. The trailing +1 is column padding
    // to dodge bank conflicts: shared memory has 32 banks, so without the
    // +1 every thread reading the same logical column would hit one bank
    // and serialize 32-way. Padding pushes them onto different banks.
    __shared__ float smA[2][BM][BK + 1];
    __shared__ float smB[2][BK][BN + 1];

    const int tid = threadIdx.y * BLOCK_X + threadIdx.x; // flat thread id (0..255)
    const int num_tiles = (K + BK - 1) / BK;             // K-tiles to walk

    // Per-thread register accumulator: 64 floats, one per output element
    // this thread is responsible for. Lives in registers, not memory —
    // that's why the inner FMA loop in PHASE C is so fast.
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    // ---- PHASE B: prefetch K-tile 0 into buffer 0 -----------------------
    // We load the very first tile BEFORE entering the main loop so that
    // iteration 0 already has data in `cur` to compute on while it
    // prefetches tile 1 into `nxt`. This is what kicks off the
    // load-and-compute overlap throughout the main loop.

    // Cooperatively load smA: each thread loads 8 scalar elements from A.
    // smA is BM*BK = 2048 floats; 256 threads * 8 floats = 2048 — every
    // thread participates so the load is fully parallel and coalesced.
    // The flat element index `tid + i*NUM_THREADS` strides each thread's
    // 8 elements across the tile so consecutive lanes touch consecutive
    // global addresses.
    #pragma unroll
    for (int i = 0; i < SMA_LOADS_PER_THREAD; ++i) {
        int elem = tid + i * NUM_THREADS;
        int sr = elem / BK;   // shared memory row
        int sc = elem % BK;   // shared memory col
        int gr = blockIdx.y * BM + sr; // global row in A
        int gc = sc;                   // global col in A (tile 0 starts at col 0)
        smA[0][sr][sc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }

    // Cooperatively load smB: each thread loads 2 float4s (8 floats) from B.
    // Use 128-bit vector reads when aligned and in bounds; edge tiles fall
    // back to scalar loads.
    #pragma unroll
    for (int i = 0; i < SMB_F4_PER_THREAD; ++i) {
        int f4 = tid + i * NUM_THREADS;
        int sr = f4 / SMB_F4_PER_ROW;      // shared memory row
        int sc = (f4 % SMB_F4_PER_ROW) * 4; // shared memory col (x4 for float4)
        int gr = sr;                         // global row in B (tile 0)
        int gc = blockIdx.x * BN + sc;      // global col in B
        int idx = gr * N + gc;
        if (gr < K && gc + 3 < N && (idx % 4 == 0)) {
            float4 val = *reinterpret_cast<const float4 *>(&B[idx]);
            smB[0][sr][sc]     = val.x;
            smB[0][sr][sc + 1] = val.y;
            smB[0][sr][sc + 2] = val.z;
            smB[0][sr][sc + 3] = val.w;
        } else {
            for (int j = 0; j < 4; ++j) {
                int cj = gc + j;
                smB[0][sr][sc + j] =
                    (gr < K && cj < N) ? B[gr * N + cj] : 0.0f;
            }
        }
    }

    __syncthreads(); // make sure tile 0 is fully written before anyone reads it

    // ---- PHASE C: main loop over K-tiles, double-buffered ---------------
    // Each iteration does THREE things, in this order:
    //   (1) Prefetch the NEXT tile into buffer `nxt`. The SM issues these
    //       HBM loads concurrently with the FMAs in step (2) — that
    //       overlap is the entire point of double-buffering.
    //   (2) Compute on the CURRENT tile in buffer `cur`: pull values into
    //       registers and run a TM x TN outer product, accumulating into
    //       acc[][]. This is where the FLOPs actually happen.
    //   (3) Barrier, then swap. Threads must finish reading `cur` before
    //       any future iteration overwrites it. After the swap, what we
    //       just loaded into `nxt` becomes the next iteration's `cur`.

    int cur = 0; // index of the buffer we'll compute on this iteration
    for (int tile = 0; tile < num_tiles; ++tile) {
        int nxt = 1 - cur; // the OTHER buffer — that's where the prefetch goes

        // ---- (1) Prefetch next tile into `nxt` (skip on the last iter) -----
        if (tile + 1 < num_tiles) {
            int next_tile = tile + 1;

            // Load smA for next tile (same pattern as prefetch above).
            #pragma unroll
            for (int i = 0; i < SMA_LOADS_PER_THREAD; ++i) {
                int elem = tid + i * NUM_THREADS;
                int sr = elem / BK;
                int sc = elem % BK;
                int gr = blockIdx.y * BM + sr;
                int gc = next_tile * BK + sc;
                smA[nxt][sr][sc] =
                    (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            }

            // Load smB for next tile (float4 vectorized).
            #pragma unroll
            for (int i = 0; i < SMB_F4_PER_THREAD; ++i) {
                int f4 = tid + i * NUM_THREADS;
                int sr = f4 / SMB_F4_PER_ROW;
                int sc = (f4 % SMB_F4_PER_ROW) * 4;
                int gr = next_tile * BK + sr;
                int gc = blockIdx.x * BN + sc;
                int idx = gr * N + gc;
                if (gr < K && gc + 3 < N && (idx % 4 == 0)) {
                    float4 val =
                        *reinterpret_cast<const float4 *>(&B[idx]);
                    smB[nxt][sr][sc]     = val.x;
                    smB[nxt][sr][sc + 1] = val.y;
                    smB[nxt][sr][sc + 2] = val.z;
                    smB[nxt][sr][sc + 3] = val.w;
                } else {
                    for (int j = 0; j < 4; ++j) {
                        int cj = gc + j;
                        smB[nxt][sr][sc + j] =
                            (gr < K && cj < N) ? B[gr * N + cj] : 0.0f;
                    }
                }
            }
        }

        // ---- (2) Compute on the CURRENT tile: rank-1 outer products -------
        // For each depth k in [0, BK): pull this thread's TM values from
        // smA[cur] (one column of its rows) into a_reg, and its TN values
        // from smB[cur] (one row of its columns) into b_reg. Then do a
        // TM x TN grid of FMAs, all from registers — no shared/global
        // loads inside the inner loop. Each register load feeds 8 FMAs
        // (column reuse), and each row load feeds 8 FMAs (row reuse).
        // 16 depths * 64 FMAs = 1024 FMAs per thread per tile.
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // This thread's TM values from its rows of smA (column k).
            float a_reg[TM];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm)
                a_reg[tm] = smA[cur][threadIdx.y * TM + tm][k];

            // This thread's TN values from smB (row k, its columns).
            float b_reg[TN];
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn)
                b_reg[tn] = smB[cur][k][threadIdx.x * TN + tn];

            // Outer product: each a_reg[tm] * b_reg[tn] is one FMA.
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm)
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn)
                    acc[tm][tn] += a_reg[tm] * b_reg[tn];
        }

        // ---- (3) Barrier and swap buffers ---------------------------------
        __syncthreads(); // every thread must finish reading `cur` before reuse
        cur = nxt;       // what we just prefetched is the next iteration's input
    }

    // ---- PHASE D: write each thread's 64 accumulators back to C ---------
    // Bounds-check each element — the block's tile may extend past the
    // real matrix edges, so the last block in each direction can have
    // some out-of-range outputs to skip.
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        int gr = blockIdx.y * BM + threadIdx.y * TM + tm;
        if (gr < M) {
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                int gc = blockIdx.x * BN + threadIdx.x * TN + tn;
                if (gc < N) {
                    C[gr * N + gc] = acc[tm][tn];
                }
            }
        }
    }
}

// ============================================================================
// Section 4 — matmul_bf16_weight_kernel: same shape, BF16 weights.
// ============================================================================
//
// Identical phase structure (A, B, C, D) to Section 3 above. The only
// difference is that B is laid out as raw BF16 bits (uint16_t), which
// halves the HBM bytes per weight load. Each load widens 4 BF16 values to
// 4 FP32 values inline (load_bf16_quad) before they go into smB. FP32
// accumulation in `acc[][]` is unchanged — only the storage dtype of
// the weight matrix shrinks.
//
// Why this kernel exists: it's how the resident-weights inference path
// fits 8B params in 24 GB VRAM. 8B params * 2 bytes = 16 GB resident;
// FP32 would be 32 GB and not fit on an L4.
// ============================================================================
__global__ void matmul_bf16_weight_kernel(const float *A,
                                          const uint16_t *B_bf16,
                                          float *C, int M, int K, int N) {
    // ---- PHASE A: shared-memory buffers + register accumulator ----------
    // Same layout as Section 3 — smB still stores FP32, the BF16 -> FP32
    // widening happens at load time before each value enters smB.
    __shared__ float smA[2][BM][BK + 1];
    __shared__ float smB[2][BK][BN + 1];

    const int tid = threadIdx.y * BLOCK_X + threadIdx.x;
    const int num_tiles = (K + BK - 1) / BK;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    // ---- PHASE B: prefetch K-tile 0 into buffer 0 -----------------------
    // smA load is identical to the FP32 kernel — A is FP32 in both paths.
    #pragma unroll
    for (int i = 0; i < SMA_LOADS_PER_THREAD; ++i) {
        int elem = tid + i * NUM_THREADS;
        int sr = elem / BK;
        int sc = elem % BK;
        int gr = blockIdx.y * BM + sr;
        int gc = sc;
        smA[0][sr][sc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }

    // smB load reads BF16 bits and widens to FP32 inline. Each thread
    // reads two aligned uint2 chunks (8 BF16 values = 4 + 4) when possible;
    // scalar fallback handles edge tiles.
    #pragma unroll
    for (int i = 0; i < SMB_F4_PER_THREAD; ++i) {
        int f4 = tid + i * NUM_THREADS;
        int sr = f4 / SMB_F4_PER_ROW;
        int sc = (f4 % SMB_F4_PER_ROW) * 4;
        int gr = sr;
        int gc = blockIdx.x * BN + sc;
        int idx = gr * N + gc;
        if (gr < K && gc + 3 < N && (idx % 4 == 0)) {
            load_bf16_quad(&smB[0][sr][sc], &B_bf16[idx]);
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int cj = gc + j;
                smB[0][sr][sc + j] =
                    (gr < K && cj < N)
                        ? bf16_bits_to_float(B_bf16[gr * N + cj])
                        : 0.0f;
            }
        }
    }

    __syncthreads();

    // ---- PHASE C: main loop over K-tiles, double-buffered ---------------
    // Same three-step body as Section 3: prefetch, compute, barrier+swap.
    // The only difference is in step (1)'s smB load, which uses the BF16
    // widening helpers instead of a plain float4 cast.
    int cur = 0;
    for (int tile = 0; tile < num_tiles; ++tile) {
        int nxt = 1 - cur;

        // ---- (1) Prefetch next tile into `nxt` (skip on the last iter) -----
        if (tile + 1 < num_tiles) {
            int next_tile = tile + 1;

            #pragma unroll
            for (int i = 0; i < SMA_LOADS_PER_THREAD; ++i) {
                int elem = tid + i * NUM_THREADS;
                int sr = elem / BK;
                int sc = elem % BK;
                int gr = blockIdx.y * BM + sr;
                int gc = next_tile * BK + sc;
                smA[nxt][sr][sc] =
                    (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            }

            #pragma unroll
            for (int i = 0; i < SMB_F4_PER_THREAD; ++i) {
                int f4 = tid + i * NUM_THREADS;
                int sr = f4 / SMB_F4_PER_ROW;
                int sc = (f4 % SMB_F4_PER_ROW) * 4;
                int gr = next_tile * BK + sr;
                int gc = blockIdx.x * BN + sc;
                int idx = gr * N + gc;
                if (gr < K && gc + 3 < N && (idx % 4 == 0)) {
                    load_bf16_quad(&smB[nxt][sr][sc], &B_bf16[idx]);
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int cj = gc + j;
                        smB[nxt][sr][sc + j] =
                            (gr < K && cj < N)
                                ? bf16_bits_to_float(B_bf16[gr * N + cj])
                                : 0.0f;
                    }
                }
            }
        }

        // ---- (2) Compute on the CURRENT tile: rank-1 outer products -------
        // Identical to Section 3 — by the time values enter smB they are
        // already FP32, so the inner FMA loop has no idea the weights came
        // from BF16 storage.
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_reg[TM];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm)
                a_reg[tm] = smA[cur][threadIdx.y * TM + tm][k];

            float b_reg[TN];
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn)
                b_reg[tn] = smB[cur][k][threadIdx.x * TN + tn];

            #pragma unroll
            for (int tm = 0; tm < TM; ++tm)
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn)
                    acc[tm][tn] += a_reg[tm] * b_reg[tn];
        }

        // ---- (3) Barrier and swap buffers ---------------------------------
        __syncthreads();
        cur = nxt;
    }

    // ---- PHASE D: write each thread's 64 accumulators back to C ---------
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        int gr = blockIdx.y * BM + threadIdx.y * TM + tm;
        if (gr < M) {
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                int gc = blockIdx.x * BN + threadIdx.x * TN + tn;
                if (gc < N) {
                    C[gr * N + gc] = acc[tm][tn];
                }
            }
        }
    }
}

// ============================================================================
// Section 5 — gpu_matmul: host entry used by the M1 grading tests.
// ============================================================================
//
// This is what TestAPI::matmul calls: pass in plain host pointers and get
// host pointers back. The function owns the full round trip:
//   (1) cudaMalloc d_A, d_B, d_C
//   (2) cudaMemcpy A and B host -> device
//   (3) launch matmul_kernel
//   (4) cudaMemcpy C device -> host
//   (5) cudaFree everything
//
// Exception-safe: if any step throws, the catch block frees whatever was
// allocated so a mid-call error does not leak GPU memory.
//
// We do NOT use this path inside the forward pass — every layer's inputs
// are already on the device, so paying for cudaMemcpy each call would
// waste PCIe bandwidth. See Section 6 for the device-pointer entry.
// ============================================================================
void gpu_matmul(const float *A, const float *B, float *C, int M, int K,
                int N) {
    if (M < 0 || K < 0 || N < 0) {
        throw std::runtime_error("gpu_matmul expects non-negative dimensions");
    }

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    // Helper to free all device allocations, returning the first error seen.
    auto free_all = [&]() -> cudaError_t {
        cudaError_t first_err = cudaSuccess;
        if (d_A != nullptr) {
            cudaError_t err = cudaFree(d_A);
            if (first_err == cudaSuccess && err != cudaSuccess) {
                first_err = err;
            }
            d_A = nullptr;
        }
        if (d_B != nullptr) {
            cudaError_t err = cudaFree(d_B);
            if (first_err == cudaSuccess && err != cudaSuccess) {
                first_err = err;
            }
            d_B = nullptr;
        }
        if (d_C != nullptr) {
            cudaError_t err = cudaFree(d_C);
            if (first_err == cudaSuccess && err != cudaSuccess) {
                first_err = err;
            }
            d_C = nullptr;
        }
        return first_err;
    };

    // Compute byte sizes (use size_t to avoid overflow on large matrices).
    const size_t bytes_A = static_cast<size_t>(M) * static_cast<size_t>(K) *
                           sizeof(float);
    const size_t bytes_B = static_cast<size_t>(K) * static_cast<size_t>(N) *
                           sizeof(float);
    const size_t bytes_C = static_cast<size_t>(M) * static_cast<size_t>(N) *
                           sizeof(float);

    try {
        // Allocate device memory for A, B, and C.
        CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
        CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
        CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

        // Copy input matrices from host to device.
        CUDA_CHECK(cudaMemcpy(d_A, A, bytes_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, B, bytes_B, cudaMemcpyHostToDevice));

        // Launch kernel: one block per BM x BN tile of the output matrix.
        const dim3 block(BLOCK_X, BLOCK_Y);
        const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

        matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaGetLastError());       // check launch errors
        CUDA_CHECK(cudaDeviceSynchronize());  // wait for kernel completion

        // Copy result matrix back to host.
        CUDA_CHECK(cudaMemcpy(C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    } catch (...) {
        (void)free_all(); // clean up device memory on error
        throw;
    }

    CUDA_CHECK(free_all());
}

// ============================================================================
// Section 6 — gpu_matmul_device: device-pointer entry for the forward pass.
// ============================================================================
//
// Same kernel as Section 5, but the caller has already put its tensors in
// VRAM (Q, K, V, X, layer norms, etc. all live on the GPU during the
// forward pass). So we skip the H2D and D2H copies — those would be wasted
// PCIe (Peripheral Component Interconnect Express) traffic. Just launch
// the kernel and return. Caller owns the d_* buffers.
//
// This is the entry point used by every projection inside the forward pass
// when running with FP32 weights.
// ============================================================================
void gpu_matmul_device(const float *d_A, const float *d_B, float *d_C,
                       int M, int K, int N) {
    if (M < 0 || K < 0 || N < 0) {
        throw std::runtime_error(
            "gpu_matmul_device expects non-negative dimensions");
    }
    if (M == 0 || K == 0 || N == 0) {
        return; // nothing to compute
    }

    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Section 7 — gpu_matmul_device_bf16_weight: BF16-weight forward entry.
// ============================================================================
//
// Same shape as Section 6, but dispatches to the BF16-weight kernel from
// Section 4. This is the hot path for the resident-weights forward pass:
// every Q/K/V/O/gate/up/down projection in every layer goes through here
// so weight bytes stay BF16 in VRAM and HBM bandwidth halves.
// ============================================================================
void gpu_matmul_device_bf16_weight(const float *d_A,
                                   const uint16_t *d_B_bf16,
                                   float *d_C, int M, int K, int N) {
    if (M < 0 || K < 0 || N < 0) {
        throw std::runtime_error(
            "gpu_matmul_device_bf16_weight expects non-negative dimensions");
    }
    if (M == 0 || K == 0 || N == 0) {
        return;
    }

    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    matmul_bf16_weight_kernel<<<grid, block>>>(d_A, d_B_bf16, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

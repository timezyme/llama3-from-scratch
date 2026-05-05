// Tiled GEMM (general matrix multiply): C = A * B, row-major FP32.
// Satisfies llm_part1 §3.1.1 Step 5 with block tiling, shared-memory
// reuse, and coalesced global loads. HBM means high-bandwidth memory.
//
// Extra choices: double-buffered shared tiles, 8x8 per-thread register
// accumulation, float4/BF16 vectorized loads, and +1 shared-memory
// padding to avoid transpose-style bank conflicts. Tensor cores are not
// used; that would require WMMA or mma.sync intrinsics and stricter tiles.

#include "kernel/kernels.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace {

// Block tile dimensions — each thread block owns a BM x BN region of C
// and walks K in BK-wide tiles. 128x128 was chosen because, during one
// rank-1 outer-product step, each smA element (at one row, one K-depth)
// is reused by every output column in the block (BN = 128 reuses), and
// each smB element is reused by every output row (BM = 128 reuses).
// BK = 16 keeps two double-buffered shared tiles (smA + smB, including
// the +1 padding) well under the per-block shared-memory budget on
// Turing/Ada (~64 KiB default).
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
// One uint2 (8 bytes) is loaded per call, so the compiler can issue an
// LDG.E.64 — a single coalesced 64-bit transaction per thread — instead
// of four separate 16-bit loads.
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

// FP32 GEMM kernel: C[M,N] = A[M,K] * B[K,N], all row-major.
// Algorithm flow per block: prefetch tile 0, then for each K-tile —
//   (a) prefetch the next tile into the alternate buffer,
//   (b) run rank-1 outer-product accumulation on the current tile,
//   (c) sync and swap buffers so the next iteration computes on what
//       step (a) just loaded.
// Each thread holds a private TM*TN register accumulator that sums every
// K-tile's contribution to its output sub-tile.
__global__ void matmul_kernel(const float *A, const float *B, float *C, int M,
                              int K, int N) {
    // Double-buffered tiles in shared memory. The leading dimension `2`
    // is the buffer index that ping-pongs between tiles to overlap loads
    // with compute. The trailing `+1` pads the inner dimension so threads
    // accessing the same logical column hit different banks during the
    // outer-product loop (avoids 32-way bank-conflict serialization).
    __shared__ float smA[2][BM][BK + 1];
    __shared__ float smB[2][BK][BN + 1];

    const int tid = threadIdx.y * BLOCK_X + threadIdx.x; // flat thread ID in block
    const int num_tiles = (K + BK - 1) / BK;             // number of K-tiles

    // Per-thread accumulators — TM x TN = 64 floats held in registers.
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    // === Prefetch first tile (tile 0) into buffer 0 ===
    // We do this once before the main loop so iteration 0 already has data
    // in `cur` to compute on while it prefetches tile 1 into `nxt`.

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
        // Use vectorized float4 load when aligned and in-bounds; scalar fallback otherwise.
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

    __syncthreads();

    // === Main loop: iterate over K-tiles with double buffering ===
    // Each iteration:
    //   1. Prefetch the NEXT tile into the alternate buffer `nxt`. This
    //      overlaps HBM loads with FMA (fused multiply-add) work on the
    //      current tile on the SM (streaming multiprocessor).
    //   2. Compute on the CURRENT tile in `cur`. This is where the work
    //      actually happens — register-level outer products that produce
    //      TM*TN partial sums per thread.
    //   3. __syncthreads() to make sure every thread has finished reading
    //      `cur` before we reuse that buffer in a future iteration. Then
    //      swap so next iteration's compute sees what step 1 just loaded.

    int cur = 0; // current buffer index (0 or 1)
    for (int tile = 0; tile < num_tiles; ++tile) {
        int nxt = 1 - cur; // alternate buffer for prefetching

        // --- Prefetch next tile into buffer `nxt` (if not the last tile) ---
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

        // --- Compute on current tile: rank-1 outer product accumulation ---
        // For each k in [0, BK): pull this thread's TM rows of smA at depth
        // k into registers (a_reg) and its TN cols of smB at depth k into
        // registers (b_reg), then run a TM*TN FMA grid. This is the core
        // arithmetic-intensity boost: BK=16 sweeps produce 16*64 = 1024
        // FMAs per thread, all sourced from registers.
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

        __syncthreads(); // ensure all threads are done reading `cur` before swapping
        cur = nxt;       // swap buffers: next iteration reads from what we just loaded
    }

    // === Write back TM x TN accumulated results to global memory C ===
    // Bounds-check each element since the tile may extend past M or N.
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

// FP32 activations * BF16 weights: C[M,N] = A[M,K] * B_bf16[K,N].
// Same tiling as matmul_kernel, but smB is loaded by widening BF16 bits
// to FP32 inline. Keeping resident weights in BF16 halves weight bytes
// read from HBM; accumulation remains FP32.
__global__ void matmul_bf16_weight_kernel(const float *A,
                                          const uint16_t *B_bf16,
                                          float *C, int M, int K, int N) {
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

    #pragma unroll
    for (int i = 0; i < SMA_LOADS_PER_THREAD; ++i) {
        int elem = tid + i * NUM_THREADS;
        int sr = elem / BK;
        int sc = elem % BK;
        int gr = blockIdx.y * BM + sr;
        int gc = sc;
        smA[0][sr][sc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }

    // Cooperatively load smB. Each thread reads two aligned uint2 chunks
    // (8 BF16 values total) when possible; scalar fallback handles edge tiles.
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

    int cur = 0;
    for (int tile = 0; tile < num_tiles; ++tile) {
        int nxt = 1 - cur;

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

        __syncthreads();
        cur = nxt;
    }

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

// Host entry used by the M1 grading tests (TestAPI::matmul).
// Owns the full host->device->host trip: cudaMalloc, copy A and B in,
// launch the kernel, copy C out, free everything. Exception-safe:
// device buffers are freed on every path so a CUDA error mid-call does
// not leak GPU memory.
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

// Device-pointer entry used inside the forward pass. Skips the H2D/D2H
// copies because the inference pipeline already has Q, K, V, X, etc. in
// VRAM (video RAM) — every redundant copy would be wasted PCIe (Peripheral
// Component Interconnect Express) bandwidth. Caller owns the d_* buffers.
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

// Same device-pointer entry as gpu_matmul_device but for the BF16-weight
// kernel. Used for every per-layer projection (Q/K/V/O/gate/up/down) in
// the resident-weights inference path so weight bytes stay BF16 in VRAM.
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

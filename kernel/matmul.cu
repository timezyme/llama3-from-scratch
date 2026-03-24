// Tiled GEMM kernel for C = A * B (row-major, FP32).
//
// Strategy: double-buffered shared memory tiles with thread coarsening.
// Each thread block computes a BM x BN tile of C. Each thread within
// the block computes a TM x TN sub-tile using register-level accumulation.
// Shared memory uses +1 padding to avoid bank conflicts.
// Matrix B is loaded via float4 vectorized reads for coalesced global access.

#include "kernel/kernels.cuh"

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace {

// Block tile dimensions — each thread block handles a BM x BN region of C,
// stepping through K in chunks of BK.
constexpr int BM = 128;   // rows of C per block
constexpr int BN = 128;   // cols of C per block
constexpr int BK = 16;    // depth of each tile along K

// Thread tile dimensions — each thread computes TM x TN = 64 output elements.
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

} // namespace

#define CUDA_CHECK(expr) throw_cuda_error((expr), #expr, __FILE__, __LINE__)

// GEMM kernel: C[M,N] = A[M,K] * B[K,N].
// Each thread computes a TM x TN sub-tile of C using register accumulation.
// Double buffering: while computing on buffer `cur`, the next tile loads into `nxt`.
__global__ void matmul_kernel(const float *A, const float *B, float *C, int M,
                              int K, int N) {
    // Double-buffered shared memory with +1 padding to avoid bank conflicts.
    // Buffer index alternates between 0 and 1 each iteration.
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

    // Cooperatively load smA: each thread loads 8 scalar elements from A.
    // Maps flat element index -> (row, col) in the BM x BK shared tile.
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
    // float4 loads give coalesced 128-bit global memory access.
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
    // On each iteration we:
    //   1. Prefetch the NEXT tile into the alternate shared memory buffer (overlap with compute)
    //   2. Compute the outer-product accumulation on the CURRENT tile
    //   3. Sync and swap buffers

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
        // For each k in [0, BK): load a column of smA and a row of smB into
        // registers, then accumulate TM x TN FMAs into acc[][].
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

// Host-side entry point: allocates device memory, copies A and B to GPU,
// launches the kernel, copies C back, and frees device memory.
// Exception-safe: device memory is freed on both success and error paths.
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

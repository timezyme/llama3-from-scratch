// RMSNorm (root-mean-square layer normalization) for Llama 3.
// Formula per row: y_i = x_i / sqrt(mean(x^2) + epsilon) * gamma_i.
// llm_part2 §4 requires epsilon inside sqrt and the gamma multiply.
// One block handles one row, matching the reduction pattern in §2.2.

#include "kernel/kernels.cuh"

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

namespace {

void throw_cuda_error(cudaError_t err, const char *expr, const char *file,
                      int line) {
    if (err == cudaSuccess) return;
    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << " for " << expr << ": "
        << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
}

} // namespace

#define CUDA_CHECK(expr) throw_cuda_error((expr), #expr, __FILE__, __LINE__)

// rmsnorm_kernel — one block per row of `input`. Each thread strides
// across the row in steps of blockDim.x so threads access consecutive
// columns within each step (coalesced HBM reads/writes).
__global__ void rmsnorm_kernel(const float *__restrict__ input,
                               const float *__restrict__ gamma,
                               float *__restrict__ output,
                               int rows, int cols, float epsilon) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float *x = input + row * cols;
    float *y = output + row * cols;

    extern __shared__ float sdata[];

    // Pass 1: each thread accumulates a partial sum of squares for the
    // columns it owns. blockDim.x partial sums end up in sdata[].
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = x[i];
        thread_sum += val * val;
    }
    sdata[threadIdx.x] = thread_sum;
    __syncthreads();

    // Tree reduction (PMPP-style): each step halves the active threads
    // and adds the upper half into the lower half until sdata[0] holds
    // the total. log2(blockDim.x) sync barriers, no warp divergence.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Compute RMS for this row. Epsilon is inside the sqrt so a row
    // with sum_sq=0 still produces a finite (small) divisor instead of
    // dividing by zero; this is the numerical-stability invariant from
    // llm_part2 §2.1.
    //
    // llm_part2 §2.2 suggests "a single thread computes RMS(x),
    // broadcasts it through shared memory, and each thread writes its
    // scaled output." We let every thread compute the same scalar
    // independently — the inputs (sdata[0], cols, epsilon) are
    // already visible to all threads after the reduction's syncthreads,
    // so each thread does one extra div+sqrt and skips the dedicated
    // broadcast step. Equivalent results; simpler code.
    float rms = sqrtf(sdata[0] / static_cast<float>(cols) + epsilon);

    // Pass 2: write y[i] = (x[i] / rms) * gamma[i]. The gamma multiply
    // must be applied here — skipping it is one of the most common
    // RMSNorm bugs flagged in the assignment.
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        y[i] = x[i] / rms * gamma[i];
    }
}

// Host launcher. All pointers must already live in device memory.
void gpu_rmsnorm(const float *d_input, const float *d_gamma,
                 float *d_output, int rows, int cols, float epsilon) {
    if (rows <= 0 || cols <= 0) return;

    // 256 threads per block: power-of-2 (clean tree reduction) and a
    // good occupancy point on Turing/Ada. cols=4096 means each thread
    // handles 4096/256 = 16 elements per pass, plenty of work per
    // thread to amortize the reduction overhead.
    const int threads = 256;
    const int shared_bytes = threads * sizeof(float);

    rmsnorm_kernel<<<rows, threads, shared_bytes>>>(
        d_input, d_gamma, d_output, rows, cols, epsilon);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

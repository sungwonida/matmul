#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace david {
namespace algorithms {
namespace math {

// #define MATRIX_ROWS 1024
// #define MATRIX_K 1024
// #define MATRIX_COLS 1024
// #define MATRIX_A_SIZE (MATRIX_ROWS * MATRIX_K)
// #define MATRIX_B_SIZE (MATRIX_K * MATRIX_COLS)
// #define MATRIX_C_SIZE (MATRIX_ROWS * MATRIX_COLS)

#define MAX_THREADS_PER_BLOCK 1024
#define BLOCK_SIZE_PER_DIM 32
#define BLOCK_SIZE (BLOCK_SIZE_PER_DIM * BLOCK_SIZE_PER_DIM)
#if BLOCK_SIZE > MAX_THREADS_PER_BLOCK
#error "Block size is too big"
#endif

#define FLOAT_BITS 32
#define SHARED_MEM_PER_BLOCK 49152

// #define BLOCK_MATRIX_A_SIZE (BLOCK_SIZE_PER_DIM * MATRIX_K)
// #define BLOCK_MATRIX_B_SIZE (MATRIX_K * BLOCK_SIZE_PER_DIM)


__global__ void matmulRegister(float* A, float* B, float* C, int mat_rows, int mat_k, int mat_cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(r < mat_rows && c < mat_cols)) return;
    int ind = r * mat_cols + c;

    float r_c = 0;
    for (int k = 0; k < mat_k; k++)
        r_c += A[r * mat_k + k] * B[k * mat_cols + c];
    C[ind] = r_c;
}

// #if ((FLOAT_BITS/8) * BLOCK_MATRIX_A_SIZE + (FLOAT_BITS/8) * BLOCK_MATRIX_B_SIZE) <= SHARED_MEM_PER_BLOCK
// #warning "Algorithm A"
__global__ void matmulSharedMemA(float* A, float* B, float* C, int mat_rows, int mat_k, int mat_cols) {
    /* Use shared memory as a cache
     */
    int br = threadIdx.y;
    int bc = threadIdx.x;
    int r = blockIdx.y * blockDim.y + br;
    int c = blockIdx.x * blockDim.x + bc;
    if (!(r < mat_rows && c < mat_cols)) return;
    int ind = r * mat_cols + c;

    extern __shared__ float s_A[];
    extern __shared__ float s_B[];
    // __shared__ float s_A[BLOCK_SIZE_PER_DIM][mat_k];
    // __shared__ float s_B[mat_k][BLOCK_SIZE_PER_DIM];

    // Load data from the global memory to the shared memory
    for (int k = 0; k < mat_k; k++) {
        // s_A[br][k] = A[r * mat_k + k];
        // s_B[k][bc] = B[k * mat_cols + c];
        s_A[br * mat_k + k] = A[r * mat_k + k];
        s_B[k * BLOCK_SIZE_PER_DIM + bc] = B[k * mat_cols + c];
    }

    // Wait for other threads to reach at this point before going further
    __syncthreads();

    // Calculate the matrix multiplication
    C[ind] = 0;
    for (int k = 0; k < mat_k; k++)
        // C[ind] += s_A[br][k] * s_B[k][bc];
        C[ind] += s_A[br * mat_k + k] * s_B[k * BLOCK_SIZE_PER_DIM + bc];
}
// #elif ((FLOAT_BITS/8) * MATRIX_C_SIZE) <= SHARED_MEM_PER_BLOCK
// #else
// #warning "Algorithm B"
__global__ void matmulSharedMemB(float* A, float* B, float* C, int mat_rows, int mat_k, int mat_cols) {
    int br = threadIdx.y;
    int bc = threadIdx.x;
    int r = blockIdx.y * blockDim.y + br;
    int c = blockIdx.x * blockDim.x + bc;
    if (!(r < mat_rows && c < mat_cols)) return;
    int bind = br * blockDim.x + bc;
    int ind = r * mat_cols + c;

    __shared__ float s_C[BLOCK_SIZE];

    s_C[bind] = 0;
    for (int k = 0; k < mat_k; k++)
        s_C[bind] += A[r * mat_k + k] * B[k * mat_cols + c];
    C[ind] = s_C[bind];
}
// #endif

__global__ void matmulGlobalMem(float* A, float* B, float* C, int mat_rows, int mat_k, int mat_cols) {
    /* The simplest implementation using the global memory only
     */
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(r < mat_rows && c < mat_cols)) return;
    int ind = r * mat_cols + c;

    C[ind] = 0;
    for (int k = 0; k < mat_k; k++)
        C[ind] += A[r * mat_k + k] * B[k * mat_cols + c];
}

void matmult_cuda(int m, int n, int k, const float* h_A, const float* h_B, float* h_C, int impl = 0) {
    if (impl < 0 || impl > 2) return;

    int mat_a_size = sizeof(float) * m * k;
    int mat_b_size = sizeof(float) * k * n;
    int mat_c_size = sizeof(float) * m * n;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, mat_a_size); 
    cudaMalloc((void**)&d_B, mat_b_size); 
    cudaMalloc((void**)&d_C, mat_c_size);    

    // Send to Device
    cudaMemcpy(d_A, h_A, mat_a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mat_b_size, cudaMemcpyHostToDevice);

    // Thread layout
    dim3 block(BLOCK_SIZE_PER_DIM, BLOCK_SIZE_PER_DIM);
    dim3 grid((n + block.y - 1) / block.y, (m + block.x - 1) / block.x);

    // Compute
    if (impl == 0)
        matmulGlobalMem <<<grid, block>>> (d_A, d_B, d_C, m, k, n);
    else if (impl == 1)
        matmulSharedMemB <<<grid, block>>> (d_A, d_B, d_C, m, k, n);
    else // (impl == 2)
        matmulRegister <<<grid, block>>> (d_A, d_B, d_C, m, k, n);

    // Wait to synchronize because kernel-launch is asynchronous
    cudaThreadSynchronize(); 

    // Get back to Host
    cudaMemcpy(h_C, d_C, mat_c_size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

} // math
} // algorithms
} // david
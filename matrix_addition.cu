#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    int work_id = threadIdx.x + blockDim.x * blockIdx.x;

    if(work_id < N*N){
        C[work_id] = A[work_id] + B[work_id];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    printf("blocks per grid %d %d\n", (N * N + threadsPerBlock - 1), threadsPerBlock);

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

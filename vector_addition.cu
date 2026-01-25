#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    printf("hello from GPU %d %d %d\n", threadIdx.x, blockDim.x, blockIdx.x);
    int work_index = threadIdx.x + blockDim.x * blockIdx.x;
    //int work_index = threadIdx.x;
    if(work_index < N){
        C[work_index] = A[work_index] + B[work_index];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}

#include <cuda_runtime.h>
#include <stdio.h> 

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int work_id = threadIdx.x + blockDim.x * blockIdx.x;
    if(work_id<N*N){
        //printf("%f\n", A[work_id]);
        B[work_id]=A[work_id];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
}

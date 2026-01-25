#include <cuda_runtime.h>
#include <stdio.h>

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int work_id = threadIdx.x + blockDim.x * blockIdx.x;
    if(work_id<N){
        if(input[work_id]==K){
            atomicAdd(&output[0], 1);
            printf("%d\n", input[work_id]);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    cudaDeviceSynchronize();
}

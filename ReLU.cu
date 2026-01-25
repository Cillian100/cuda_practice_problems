#include <cuda_runtime.h>
#include <stdio.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int work_index = threadIdx.x + blockDim.x * blockIdx.x;
    if(work_index < N){
        if(input[work_index]>0){
            output[work_index]=input[work_index];
        }else{
            output[work_index]=0;
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
    printf("hello\n");
}

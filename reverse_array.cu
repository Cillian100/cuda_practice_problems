#include <cuda_runtime.h>
#include <stdio.h> 

__global__ void reverse_array(float* input, int N) {
    int work_index = threadIdx.x + blockDim.x * blockIdx.x;

    if(work_index < N/2){
        //printf("%d %d\n", work_index, N-work_index-1);
        float temp=input[work_index];
        input[work_index]=input[N-work_index-1];
        input[N-work_index-1]=temp;
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}

#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_DIM 256

__global__ void reduce(const float* input, float* output, int N){
    int global_id = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ float input_s[BLOCK_DIM];
    int t = threadIdx.x;
    if(t>=N){
        return;
    }
    if(global_id < N){
        input_s[t] = input[global_id];
    } else {
        input_s[t] = 0.0f;
    }
    __syncthreads();
    for(int stride=blockDim.x/2; stride>=1; stride/=2){
        if(t+stride<N && threadIdx.x<stride){
            input_s[t]+=input_s[t + stride];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        atomicAdd(output, input_s[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
  int threadsPerBlock = BLOCK_DIM;
    int blocksPerGrid = (N + threadsPerBlock + 1) / threadsPerBlock;
    reduce<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

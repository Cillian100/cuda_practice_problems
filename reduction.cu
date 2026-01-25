#include <cuda_runtime.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define STRIDE_FACTOR 8 
#define BLOCK_SIZE STRIDE_FACTOR*THREADS_PER_BLOCK

__device__ void warp_reduce(volatile float *shared_data, int tid){
    shared_data[tid] += shared_data[tid + 32];
    shared_data[tid] += shared_data[tid + 16];
    shared_data[tid] += shared_data[tid + 8];
    shared_data[tid] += shared_data[tid + 4];
    shared_data[tid] += shared_data[tid + 2];
    shared_data[tid] += shared_data[tid + 1];
}

__global__ void reduce(const float* input, float* output, int N){
    __shared__ float shared_data[THREADS_PER_BLOCK];
    int tid = threadIdx.x;

    float sum = 0.0f;
    int block_start = blockIdx.x * BLOCK_SIZE;
    for(int i=0; i < STRIDE_FACTOR; i++){
        int gid = block_start + i * THREADS_PER_BLOCK + tid;
        if(gid < N){
            sum += input[gid];
        }
    }

    shared_data[tid] = sum;
    __syncthreads();

    for(int i=blockDim.x/2; i > WARP_SIZE; i >>= 1){
        if(tid < i){
            shared_data[tid] += shared_data[i+tid];
        }
        __syncthreads();
    }

    if(tid < WARP_SIZE){
        warp_reduce(shared_data, tid);
    }

    if(tid == 0){
        atomicAdd(output, shared_data[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    dim3 threads(THREADS_PER_BLOCK);
    //dim3 blocks(cdiv(N, THREADS_PER_BLOCK * STRIDE_FACTOR));
    int blocksPerGrid = (N + THREADS_PER_BLOCK * STRIDE_FACTOR - 1) / THREADS_PER_BLOCK;
    reduce<<<blocksPerGrid, threads>>>(input, output, N);
}

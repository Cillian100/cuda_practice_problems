#include <cuda_runtime.h>
#include <stdio.h>

// M is x
// N is y

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    int work_id_x = threadIdx.x + blockDim.x * blockIdx.x;
    int work_id_y = threadIdx.y + blockDim.y * blockIdx.y;

    if(work_id_x<M && work_id_y<N){
        if(input[work_id_x + work_id_y*M] == K){
            atomicAdd(&output[0], 1);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
    cudaDeviceSynchronize();
}

#include <cuda_runtime.h>
#include <math.h>

__global__ void silu_kernel(const float* input, float* output, int N) {
    int work_id = threadIdx.x + blockDim.x * blockIdx.x;
    if(work_id < N){
        output[work_id]=input[work_id]*1/(1+exp(-input[work_id]));
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

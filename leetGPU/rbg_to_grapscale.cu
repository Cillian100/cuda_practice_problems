#include <cuda_runtime.h>
#include <stdio.h>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height){
    int work_id = threadIdx.x + blockDim.x * blockIdx.x;

    if(work_id<width*height){
        output[work_id]=input[work_id*3+0]*0.299+input[work_id*3+1]*0.587+input[work_id*3+2]*0.114;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}

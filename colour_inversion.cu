#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int work_index = (threadIdx.x + blockIdx.x * blockDim.x);
    
    if(work_index < (width * height * 4)){
        if(work_index % 4 != 3){
            image[work_index] = 255-image[work_index];
        }
    }
}


// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height * 4 + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}

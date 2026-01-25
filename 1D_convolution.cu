#include <cuda_runtime.h>
#include <stdio.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
int input_size, int kernel_size) {
    int input_variable = threadIdx.x + blockDim.x * blockIdx.x;
    int limit_variable = input_size - kernel_size + 1;
    int kernel_variable=0;

    if(input_variable < limit_variable){
        output[input_variable]=0;
        for(kernel_variable=0; kernel_variable<kernel_size; kernel_variable++){
            output[input_variable]+=input[input_variable+kernel_variable] * kernel[kernel_variable];
            //printf("%d - %f %f\n",input_variable, input[input_variable+kernel_variable], kernel[kernel_variable]);
        }
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int kernel_col = threadIdx.x + blockDim.x * blockIdx.x;
    int kernel_row = threadIdx.y + blockDim.y * blockIdx.y;

    if(kernel_col < cols && kernel_row < rows){
        int index=kernel_col + kernel_row * cols;

        int row_value = index % rows;
        int col_value = index / rows;

        int value = (row_value*cols) + col_value;
        output[index]=input[value];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);


    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

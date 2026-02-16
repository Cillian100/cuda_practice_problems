#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

template <typename T>
__global__ void matrix_transpose(int m, int n, int d, const float* Q, const float* K, float* C){
    int C_col = blockIdx.x * blockDim.x + threadIdx.x;
    int C_row = blockIdx.y * blockDim.y + threadIdx.y;

    if(C_row < m && C_col < n){
        float sum = 0;
        for(size_t idx{}; idx<d; idx++){
            sum+=Q[C_row*d + idx] * K[idx + C_col * d];
        }
        C[C_col + C_row * d] = sum / sqrt(d);
    }
}

__global__ void softmax_kernel(float* C, int id){
    int work_id = threadIdx.x + blockDim.x * blockIdx.x;
    int i = 2*threadIdx.x;
    
    for(int stride=1; stride<=blockDim.x || stride<=id;stride*=2){
        if(threadIdx.x % stride == 0){
            C[i]+=C[i+stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        printf("poop %f\n", C[i]);
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,int d){
    dim3 const block_dim{32U, 32U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(N) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(d) + block_dim.y - 1U) / block_dim.y, 1U};
    int threadsPerBlock=256;
    int blocksPerDim=(threadsPerBlock + M*d - 1)/threadsPerBlock;

    float* C = nullptr;
    cudaMalloc(&C, M*d*sizeof(float));
    

    matrix_transpose<float><<<grid_dim, block_dim>>>(M, N, d, Q, K, C);
    cudaDeviceSynchronize();
    softmax_kernel<<<blocksPerDim, threadsPerBlock>>>(C, M*N);
    cudaDeviceSynchronize();
    
    float* C_h = nullptr;
    cudaMallocHost(&C_h, M*d*sizeof(float));
    cudaMemcpy(C_h, C, M*d*sizeof(float), cudaMemcpyDefault);
    for(int a=0;a<M*N;a++){
        printf("%f\n", C_h[a]);
    }
}

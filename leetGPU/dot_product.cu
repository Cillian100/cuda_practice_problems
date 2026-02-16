#include <cuda_runtime.h>
#include <stdio.h>

#define thread_size 256

__global__ void simple_dot_product(const float* A, const float*B, float* result, int N){
    int work_id = threadIdx.x + blockDim.x * blockIdx.x;
    if(work_id<N){
        atomicAdd(&result[0], A[work_id]*B[work_id]);
    }
}

__global__ void improved_dot_product(const float* A, const float* B, float* result, int N){
    __shared__ float shared_values[thread_size];
    int thread_id = threadIdx.x;
    int working_id = threadIdx.x + blockDim.x * blockIdx.x;
    if(working_id < N){
        shared_values[thread_id]=A[working_id]*B[working_id];
    }else{
        shared_values[thread_id]=0.0f;
    }
    __syncthreads();
    for(int stride=blockDim.x/2; stride>=1; stride/=2){
        if(thread_id+stride<blockDim.x && thread_id < stride){
            shared_values[thread_id]+=shared_values[thread_id+stride];
        }
        __syncthreads();
    }

    if(thread_id==0){
        result[blockIdx.x]=shared_values[0];
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threadsPerBlock = thread_size;
    int blocksPerGrid = (N + threadsPerBlock -1 ) / threadsPerBlock;

    float* result_2;
    cudaMalloc(&result_2, blocksPerGrid*sizeof(float));
    improved_dot_product<<<blocksPerGrid, threadsPerBlock>>>(A, B, result_2, N);
    cudaDeviceSynchronize();
    float* result_3 = new float[blocksPerGrid];
    cudaMemcpy(result_3, result_2, blocksPerGrid*sizeof(float), cudaMemcpyDefault);
    for(int a=1;a<blocksPerGrid;a++){
        result_3[0]=result_3[a]+result_3[0];
    }
    cudaMemcpy(result, result_3, sizeof(float), cudaMemcpyDefault);
}

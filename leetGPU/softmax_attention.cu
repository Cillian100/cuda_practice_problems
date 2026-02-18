#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_DIM 1024

__global__ void matrix_transpose(int m, int n, int d, float* Q, float* K, float* C){
  int C_col = threadIdx.x + blockDim.x * blockIdx.x;
  int C_row = threadIdx.y + blockDim.y * blockIdx.y;

  if(C_col < n && C_row < m){
    float sum = 0;
    for(size_t idx{0}; idx<d; idx++){
      sum+=Q[C_row * d + idx]*K[idx + C_col * d];
    }
    C[C_col + C_row * n] = (float)(sum / sqrtf(d));
  }
}

__global__ void matrix_multiplication(int m, int n, int d, float* C, float* V, float* output){
  int C_col = threadIdx.x + blockDim.x * blockIdx.x;
  int C_row = threadIdx.y + blockDim.y * blockIdx.y;

  if(C_col < n && C_row < m){
    float sum = 0;
    for(size_t idx{0}; idx<d; idx++){
      
    }
  }
}

__global__ void softmax(float* C, float* output, int value){
  int work_id = threadIdx.x + blockDim.x * blockIdx.x;
  if(work_id>=value){
    return;
  }
  printf("poop - %f\n", output[0]);
}

__global__ void softmax_2(float* C, float* output, int value){
  int work_id=threadIdx.x + blockDim.x * blockIdx.x;
  if(work_id<value){
    C[work_id]=C[work_id]/output[0];
  }
}

int main(){
  int M = 2;
  int N = 3;
  int d = 4;

  float *output = nullptr;
  cudaMalloc(&output, M*N*sizeof(float));
  float Q_host[]={1,0,0,0,0,1,0,0};
  float* Q = nullptr;
  cudaMalloc(&Q, M*d*sizeof(float));
  cudaMemcpy(Q, Q_host, M*d*sizeof(float), cudaMemcpyDefault);

  float K_host[]={1,0,0,0,
                  0,1,0,0,
                  0,0,1,0};
  float* K = nullptr;
  cudaMalloc(&K, N*d*sizeof(float));
  cudaMemcpy(K, K_host, N*d*sizeof(float), cudaMemcpyDefault);
  
  float* V = nullptr;
  cudaMalloc(&V, N*d*sizeof(float));
  float* C = nullptr;
  cudaMalloc(&C, M*N*sizeof(float));
  
  dim3 const block_dim{32U, 32U};
  dim3 const grid_dim{
		      (static_cast<unsigned int>(N) + block_dim.x - 1U) / block_dim.x,
		      (static_cast<unsigned int>(d) + block_dim.y - 1U) / block_dim.y, 1U};

  int block_dimension=BLOCK_DIM;
  int grid_dimension=(M*N + block_dimension - 1)/block_dimension;
  
  matrix_transpose<<<grid_dim, block_dim>>>(M, N, d, Q, K, C);
  cudaDeviceSynchronize();

  float *output_2 = nullptr;
  cudaMalloc(&output_2, M*sizeof(float));
  
  softmax<<<grid_dimension, block_dimension>>>(C, output_2, M*N);
  cudaDeviceSynchronize();
  
  softmax_2<<<grid_dimension, block_dimension>>>(C, output_2, M*N);
  cudaDeviceSynchronize();

  float* C_host = nullptr;
  cudaMallocHost(&C_host, M*N*sizeof(float));
  cudaMemcpy(C_host, C, M*N*sizeof(float), cudaMemcpyDefault);
  for(int a=0;a<M*N;a++){
    printf("%f\n", C_host[a]);
  }
  
  

  matrix_multiplication<<<grid_dim, block_dim>>>(M, N, d, C, V, output);
  cudaDeviceSynchronize();
}

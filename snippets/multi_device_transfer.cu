#include <cuda_runtime.h>
#include <stdio.h>

__global__ void myKernel(float* p){
  int work_id = threadIdx.x + blockDim.x * blockIdx.x;

  p[work_id]+=1;
}

int main(){
  cudaSetDevice(0);
  float* p0;
  size_t size = 1024 * sizeof(foat);
  cudaMalloc(&p0, size);

  cudaSetDevice(1);
  float* p1;
  size_t size = 1024 * sizeof(float);

  cudaSetDevice(0);
  myKernel<<<256, 4>>>(p0);

  cudaSetDevice(1);
  cudaMemcpy(p1, 1, p0, 0, size);
  myKernel<<<256, 4>>>(p1);
}

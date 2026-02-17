#include <cuda_runtime.h>
#include <stdio.h>

__global__ void myKernel(float* p0){
  int work_id = threadIdx.x + blockDim.x * blockIDx.x;
}

int main(){
  cudaSetDevice(0);
  float* p0;
  size_t size = 1024 * sizeof(float);
  cudaMalloc(&p0, size);
  myKernel<<<1000, 128>>>(p0);

  cudaSetDevice(1);
  cudaDeviceEnablePeerAccess(0, 0);

  myKernel<<<1000, 128>>>(p0);
}

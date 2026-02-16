#include <cuda_runtime.h>
#include <stdio.h>

int main(){
  int deviceCount;
  cudaDeviceCount(&deviceCount);
  int device;
  for(device = 0; device < deviceCount; device++){
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capabilities %d %d\n", device, deviceProp.major, deviceProp.minor);
  }


  size_t size 1024 * sizeof(float);
  cudaSetDevice(0);
  float *p0;
  cudaMalloc(&p0, size);
  MyKernel<<<1000, 128>>>(p0);

  cudaSetDevice(1);
  float *p1;
  cudaMalloc(&p1, size);
  MyKernel<<<1000, 128>>>(p1);
}

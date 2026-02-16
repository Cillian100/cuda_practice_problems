#include <cuda_runtime.h>
#include <stdio.h>

__global__ void MyKernel(int GPU){
        int work_id = threadIdx.x + blockDim.x * blockIdx.x;

        if(work_id < 5){
                printf("hello from GPU %d\n", GPU);
        }
}

int main(){
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        int device;
        for(device = 0; device < deviceCount; device++){
                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, device);
                printf("Device %d has compute capabilites %d %d\n", device, deviceProp.major, deviceProp.minor);
        }

        size_t size = 1024 * sizeof(float);
        cudaSetDevice(0);
        float *p0;
        cudaMalloc(&p0, size);
        MyKernel<<<1000, 128>>>(1);

        cudaDeviceSynchronize();

        cudaSetDevice(1);
        float *p1;
        cudaMalloc(&p1, size);
        MyKernel<<<1000, 128>>>(2);

        cudaDeviceSynchronize();
        printf("poop");
}

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024
// 0 = flase, 1 = true

__device__ void swap(int* array, int work_id_1, int work_id_2){
  int temp=array[work_id_2];
  array[work_id_2]=array[work_id_1];
  array[work_id_1]=temp;
}

__device__ void print_array_kernel(int* array, int N){
  for(int a=0;a<N;a++){
    printf("%d ", array[a]);
  }
  printf("\n");
}

__host__ void print_array_host(int* array, int N){
  for(int a=0;a<N;a++){
    printf("%d ", array[a]);
  }
  printf("\n");
}

__global__ void odd_even_kernel(int* array, int N){
  int work_id = threadIdx.x;
  if(work_id>=N){
    return;
  }
  
  __syncthreads();
  __shared__ int shared_array[BLOCK_SIZE];
  __shared__ bool isSorted[1];

  if(work_id==0){
    isSorted[0]=false;
  }
  shared_array[work_id]=array[work_id];
  __syncthreads();
  
  while(isSorted[0]==false){
    isSorted[0]=true;

    if(work_id%2==1 && work_id+1!=N){
      if(shared_array[work_id]>shared_array[work_id+1]){
	swap(shared_array, work_id, work_id+1);
	isSorted[0]=false;
      }
    }
    __syncthreads();
    
    if(work_id%2==0 && work_id+1!=N){
      if(shared_array[work_id]>shared_array[work_id+1]){
	swap(shared_array, work_id, work_id+1);
	isSorted[0]=false;
      }
    }
    __syncthreads();
  }
  array[work_id]=shared_array[work_id];
}


int main(){
  int N=BLOCK_SIZE*5;
  int* array_host = nullptr;
  int* array_device = nullptr;
  int threadsPerBlock=BLOCK_SIZE;
  int blocksPerGrid=(threadsPerBlock + N - 1)/threadsPerBlock;

  cudaMallocHost(&array_host, N*sizeof(int));
 
  for(int a=0;a<N;a++){
    array_host[a]=rand()%1000;
  }

  for(int a=0;a<blocksPerGrid;a++){
    cudaMalloc(&array_device, BLOCK_SIZE*sizeof(int));
    cudaMemcpy(array_device, array_host+(BLOCK_SIZE*a), BLOCK_SIZE*sizeof(int), cudaMemcpyDefault);
    odd_even_kernel<<<1, threadsPerBlock>>>(array_device, BLOCK_SIZE);
    cudaDeviceSynchronize();
    cudaMemcpy(array_host+(BLOCK_SIZE*a), array_device, BLOCK_SIZE*sizeof(int), cudaMemcpyDefault);
  }

  for(int a=0;a<blocksPerGrid;a++){
    for(int b=0;b<BLOCK_SIZE;b++){
      printf("%d ", array_host[b+BLOCK_SIZE*a]);
    }
    printf("\n");
  }

  cudaDeviceSynchronize();
}

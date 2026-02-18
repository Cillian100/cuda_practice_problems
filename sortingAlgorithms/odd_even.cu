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

void create_array_host(int* array, int N){
  for(int a=0;a<N;a++){
    array[a]=rand()%(100);
  }
}

void create_array_device(int* array, int N, int blocksPerGrid){
  printf("%d\n", blocksPerGrid);
  //  int (*array_device)[1024] = calloc(blocksPerGrid, sizeof*array_device);
  //  int (*array_device)[10] = calloc(10, sizeof*array);
  int** array_device = nullptr;
}

int main(){
  int N=100;
  int *array=(int*)malloc(N*sizeof(int));
  int* array_device = nullptr;
  int threadsPerBlock=BLOCK_SIZE;
  int blocksPerGrid=(threadsPerBlock + N - 1)/threadsPerBlock;
  create_array_host(array, N);
  create_array_device(array, N, blocksPerGrid);
  

  //  cudaMalloc(&array_device, N*sizeof(int));
  //  cudaMemcpy(array_device, array, N*sizeof(int), cudaMemcpyDefault);

  //  odd_even_kernel<<<1, BLOCK_SIZE>>>(array_device, N);

  //  cudaDeviceSynchronize();
  //  printf("poop\n");
  //  cudaMemcpy(array, array_device, N*sizeof(int), cudaMemcpyDefault);

  //  print_array_host(array, N);
}

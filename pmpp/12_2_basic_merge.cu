#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

__device__ int ceiling_kernel(double number){
  if(number==(int)number){
    return number;
  }else{
    return (int)number+1;
  }
}

__device__ int minimum_kernel(int value1, int value2){
  if(value1<value2){
    return value1;
  }else{
    return value2;
  }
}

__device__ int co_rank(int k, int* A, int m, int* B, int n){
  int i = k < m ? k : m;
  int j = k - i;
  int i_low = 0 > (k-n) ? 0 : k-n;
  int j_low = 0 > (k-m) ? 0 : k-m;
  int delta;
  bool active=true;

  while(active){
    if( (i > 0) && (j < n) && (A[i-1] > B[j]) ){
      delta = ((i - i_low + 1) >> 1);
      j_low = j;
      j = j + delta;
      i = i - delta;
    }else if((j > 0) && (i < m) && (B[j-1]>=A[i])){
      delta = ((j - j_low + 1) >> 1);
      i_low = i;
      i = i + delta;
      j = j - delta;
    }else{
      active=false;
    }
  }
  return i;
}

__device__ void merge_sequential_device(int* A, int m, int* B, int n, int* C){
  int i=0;
  int j=0;
  int k=0;

  while((i<m)&&(j<n)){
    if(A[i]<=B[j]){
      C[k++] = A[i++];
    }else{
      C[k++] = B[j++];
    }
  }

  if(i == m){
    while(j < n){
      C[k++] = B[j++];
    }
  }else{
    while(i < m){
      C[k++] = A[i++];
    }
  }
}

__global__ void basic_merge_kernel(int* A, int m, int* B, int n, int* C){
  int work_id = threadIdx.x + blockDim.x * blockIdx.x;
  if(work_id>=m+n){
    return;
  }
  //  if(work_id==0){
  
    int elementsPerThread = ceiling_kernel((double)(m+n)/(blockDim.x*gridDim.x));
    int k_curr = work_id*elementsPerThread;
    int k_next = minimum_kernel((work_id+1)*elementsPerThread, m+n);
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    merge_sequential_device(&A[i_curr], i_next-i_curr, &B[j_curr], j_next-j_curr, &C[k_curr]);

    //  }
}

int main(){
  int A[]={1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21};
  int B[]={2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};
  int C[22];
  int a_num=11, b_num=11;

  basic_merge_kernel<<<1,20>>>(A, a_num, B, b_num, C);
  cudaDeviceSynchronize();

  
  for(int a=0;a<a_num+b_num;a++){
    printf("%d ", C[a]);
  }
  printf("\n");

}

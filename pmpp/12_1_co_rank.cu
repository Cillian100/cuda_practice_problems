#include <cuda_runtime.h>
#include <stdio.h>

int co_rank(int k, int* A, int m, int* B, int n){
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

int main(){
  int k = 8, m = 5, n = 4;
  int A[]={1, 7, 8, 9, 10};
  int B[]={7, 10, 10, 12};

  printf("%d\n", co_rank(k, A, m, B, n));
}

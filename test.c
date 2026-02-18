#include <stdio.h>
#include <stdlib.h>

int main(){
  int (*array)[10] = calloc(10, sizeof*array);

  for(int a=0;a<10;a++){
    for(int b=0;b<10;b++){
      printf("%d\n", array[a][b]);
    }
  }
}

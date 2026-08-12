#include <stdio.h>

int sum(int *, int);

int sum(int *a, int len) {
  if (len == 0) return 0;
  return *a + sum(a+1,len-1);
}

int main(void) {
  int a[] = {1,2,3,14,5,6};
  printf("sum of [1,2,3,14,5,6] = %d\n",sum(a,6));
}

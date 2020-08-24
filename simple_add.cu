#include "stdio.h"
#include "common/book.h"

// global will let the compiler know that this should run on device
// instead of host
__global__ void add(int a, int b, int *c) {
  *c = a + b;
}

int main() {
  int c;
  int *dev_c;
  int a = 2, b = 7;
  HANDLE_ERROR(cudaMalloc((void**) &dev_c, sizeof(int))); // allocated on device
  add<<<1, 1>>>(a, b, dev_c);
  // copy result from device to host
  HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
  printf("%d + %d = %d\n", a, b, c);
  cudaFree(dev_c);
  return 0;
}

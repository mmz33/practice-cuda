#include "stdio.h"
#include "book.h"

// check devices properties

int main() {
  cudaDeviceProp prop;
  int count;
  HANDLE_ERROR(cudaGetDeviceCount(&count));
  for (int i = 0; i < count; ++i) {
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
    printf("Name: %s\n", prop.name);
    printf("Total global memory: %ld\n", prop.totalGlobalMem);
    printf("Total constant memory: %ld\n", prop.totalConstMem);
    // a lot more
  }
  return 0;
}

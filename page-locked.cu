#include <stdio.h>
#include "../common/book.h"

/*
 * Tested on GPU GTX 980
 * Interesting results below:
  
  During copy from host to device
  Time using pageable memory: 1307.5 ms
  MB/s: 3059.3

  During copy from device to host
  Time using pageable memory: 1317.7 ms
  MB/s: 3035.6

  During copy from host to device
  Time using page-locked memory: 706.6 ms
  MB/s: 5661.0

  During copy from device to host
  Time using page-locked memory: 630.1 ms
  MB/s: 6348.0
*/


#define SIZE (10 * 1024 * 1024) // 10 MB

float cuda_host_alloc_test(int size, bool pageable, bool host_to_device) {
  cudaEvent_t start, stop;
  int *a, *dev_a;
  float elapsed_time;

  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));

  if (pageable) {
    a = (int*)malloc(size * sizeof(int));
  } else {
    HANDLE_ERROR(cudaHostAlloc((void**)&a, size * sizeof(int), cudaHostAllocDefault));
  }

  HANDLE_NULL(a);
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(int)));

  HANDLE_ERROR(cudaEventRecord(start, 0));
  // 100 copies
  for (int i = 0; i < 100; ++i) {
    if (host_to_device) {
      HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
    } else {
      HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost));
    }
  }
  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
  
  if (pageable)
    free(a);
  else
    HANDLE_ERROR(cudaFreeHost(a));
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  return elapsed_time;
}

void print_time(float MB, float elapsed_time, bool pageable, bool host_to_device) {
  const char* direction;
  if (host_to_device)
    direction = "host to device";
  else
    direction = "device to host";

  const char* mem_type;
  if (pageable)
    mem_type = "pageable";
  else
    mem_type = "page-locked";
  
  printf("During copy from %s\n", direction);
  printf("Time using %s memory: %3.1f ms\n", mem_type, elapsed_time);
  printf("MB/s: %3.1f\n\n", MB/(elapsed_time/1000));
}
  
int main(void) {
  float elapsed_time;
  float MB = (float)100 * SIZE * sizeof(int) / 1024 / 1024;
  bool mem_type[2] = {true, false};
  bool copy_dir[2] = {true, false};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      elapsed_time = cuda_host_alloc_test(SIZE, mem_type[i], copy_dir[j]);
      print_time(MB, elapsed_time, mem_type[i], copy_dir[j]);
    }
  }
  return 0;
}

#include "stdio.h"
#include "common/book.h"
#include "time.h"
#include "stdlib.h"

#define N (33 * 1024)
#define NUM_THREADS 128
#define NUM_BLOCKS 128

__global__ void add(int *a, int *b, int *c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x; // same as linear access in 2D grid
  while (tid < N) {
    c[tid] = a[tid] + b[tid];
    tid += blockDim.x * gridDim.x; // jump over all threads ids
  }
}

int random(int min, int max) {
  return min + rand() % (max - min + 1);
}

int main() {
  int a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;

  // allocate vectors on device
  HANDLE_ERROR(cudaMalloc((void**) &dev_a, N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**) &dev_b, N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**) &dev_c, N * sizeof(int)));

  // fill host vectors randomly so that later we copy them to device
  srand(time(0)); // use current time as seed
  for (int i = 0; i < N; ++i) {
    a[i] = random(1, 100);
    b[i] = random(1, 100);
  }

  // copy host vectors to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

  // create NUM_BLOCKS block each spawning NUM_THREADS threads
  // this can be imagined as 2D grid dimension where each block
  // is 1D. 
  // setting this depends on the hardware used, also your data. 
  // note that there is some hardware limit here and that's why we can 
  // benefit from blocks and threads combination
  add<<<NUM_BLOCKS, NUM_THREADS>>>(dev_a, dev_b, dev_c);

  // copy result back to c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

  bool success = true;
  for (int i = 0; i < N && success; ++i) {
    if (a[i] + b[i] != c[i])
      success = false;
  }

  if (!success)
    printf("something went wrong!\n");
  else
    printf("worked!\n");

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}

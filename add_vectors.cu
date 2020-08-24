#include "stdio.h"
#include "common/book.h"
#include "time.h"
#include "stdlib.h"

#define N 10

__global__ void add(int *a, int *b, int *c) {
  int tid = blockIdx.x; // thread id
  // do this check just in case by "mistake" the number of blocks were
  // greater than N
  if (tid < N)
    c[tid] = a[tid] + b[tid];
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
    a[i] = random(1, 10);
    b[i] = random(1, 10);
  }

  // copy host vectors to device
  HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

  // spawn N blocks each having 1 thread (each block will run in parallel)
  // number of threads = N blocks * 1 thread/block = N threads
  add<<<N, 1>>>(dev_a, dev_b, dev_c);

  // copy result back to c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i) {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}

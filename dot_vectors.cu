#include "stdio.h"
#include "common/book.h"
#include "time.h"
#include "stdlib.h"

#define imin(a, b) (a < b ? a : b)

const int N = 33 * 1024;
const int threads_per_block = 256;
// this should not so big in order to do the sum on CPU
const int blocks_per_grid = imin(32, (N + threads_per_block - 1) / threads_per_block);

__global__ void dot(float *a, float *b, float *c) {
  // shared memory between threads in this block
  // each thread will store its running sum at its index
  __shared__ float cache[threads_per_block];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cache_idx = threadIdx.x;
  float temp = 0;
  while (tid < N) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }
  cache[cache_idx] = temp;
  __syncthreads(); // make sure all threads stored their running sum

  // do sum reduction. a naive way to do this is simple let one thread
  // do the sum at the end. however, we can do it in a more efficient
  // way that only takes log_2(num_threads). the idea is to only allow
  // threads less than half the array each time to reduce by adding
  // their sum to the sum of the thread in the other half. this step
  // is repeated until we have one total sum at the end. note that here
  // we also need to sync between threads after each split step.
  // NOTE: number of threads should be power of 2
  int step = blockDim.x / 2;
  while (step) {
    if (cache_idx < step) {
      cache[cache_idx] += cache[cache_idx + step];
    }
    __syncthreads();
    step /= 2;
  }
  // any thread could do that here but we can choose thread 0 for simplicity
  // we will do the sum on the host and not here! it turns out GPUs might not 
  // be well utilized when the input is small
  if (cache_idx == 0)
    c[blockIdx.x] = cache[0];
}

int main() {
  float *a, *b, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;

  a = (float*)malloc(N * sizeof(float));
  b = (float*)malloc(N * sizeof(float));
  partial_c = (float*)malloc(blocks_per_grid * sizeof(float));

  HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_partial_c, blocks_per_grid * sizeof(float)));

  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = i * 2;
  }
  
  HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
    
  dot<<<blocks_per_grid, threads_per_block>>>(dev_a, dev_b, dev_partial_c);

  HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocks_per_grid * sizeof(float),
    cudaMemcpyDeviceToHost));  

  float res = 0.f;
  for (int i = 0; i < blocks_per_grid; ++i) {
    res += partial_c[i];
  }
 
  printf("dot product: %.2f\n", res);
  
  // I observed that sometimes this does not match the cuda reduction res
  // this might be due to some undeterministic behavior on the GPU so it
  // a hardware issue. this was tested on GPU GTX 980
  float expected_res = 0.f;
  for (int i = 0; i < N; ++i) {
    expected_res += a[i] * b[i];
  }
  printf("expected: %.2f\n", expected_res);

  // free memory on GPU
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_partial_c);
  
  // free memory on CPU
  delete[] a;
  delete[] b;
  delete[] partial_c;
}

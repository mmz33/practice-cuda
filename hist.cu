#include "common/book.h"

// *** using compute_hist kernel ***
// Some speed benchmarking with different number of blocks.
// all with 256 threads
// Hardware: GPU GTX-980

// number of blocks: 64
// Elapsed Time: 1049.9 ms

// using 2 * number of gpu multiprocessors
// number of blocks: 32
// Elapsed Time: 887.7 ms

// number of blocks: 16
// Elapsed Time: 953.7 ms

// number of blocks: 8
// Elapsed Time: 1053.6 ms

// each thread process one operation (single element)
// number of blocks: 4096000
// Elapsed Time: 1045.3 ms

// each thread process 4096000 operations
// number of blocks: 1
// Elapsed Time: 2172.0 ms

//----------------------------------------------------------//

// when using better_compute_kernel
// here we allocate a shared memory between threads in
// each block and thus reduce the atomic ops requests on the 
// global memory which make it faster

// number of blocks: 32
// Elapsed Time: 468.6 ms


#define SIZE 1000 * 1024 * 1024 // 100 MB

__global__ void compute_hist(unsigned char *dev_buffer, int size, unsigned int *dev_hist) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  while (idx < size) {
    // this will generate an atomic sequence of operations between threads.
    // here we have read-modify-write ops.
    // it will read data from the given address, modify, and write it back
    // it is guaranteed that there is no race condition
    atomicAdd(&(dev_hist[dev_buffer[idx]]), 1);
    idx += stride;
  }
}

__global__ void better_compute_hist(unsigned char *dev_buffer, int size,
  unsigned int *dev_hist) {
  
  __shared__ unsigned int t[256];
  t[threadIdx.x] = 0;
  __syncthreads();
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  while (idx < size) {
    atomicAdd(&(t[dev_buffer[idx]]), 1);
    idx += stride;
  }
  __syncthreads();
  atomicAdd(&(dev_hist[threadIdx.x]), t[threadIdx.x]);
}


int main() {
  // create a 100 MB random block of unsigned chars (values from 0 to 255)
  // see big_random_block function in common/book.h for more details
  unsigned char *buffer = (unsigned char*)big_random_block(SIZE);
  
  // create cuda events
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));

  // copy allocated buffer to GPU
  printf("copying buffer to GPU...\n");
  unsigned char *dev_buffer;
  HANDLE_ERROR(cudaMalloc((void**)&dev_buffer, SIZE));
  HANDLE_ERROR(cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice));

  // create histogram output on GPU
  printf("creating histogram device storage...\n");
  unsigned int *dev_hist;
  HANDLE_ERROR(cudaMalloc((void**)&dev_hist, 256 * sizeof(int)));
  HANDLE_ERROR(cudaMemset(dev_hist, 0, 256 * sizeof(int)));

  // call kernel
  // 100MB ~ 104M bytes / 256 ~ 409K
  // question: How many blocks should we launch to have the best performance?
  // see the speed benchmarks above
  cudaDeviceProp prop;
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
  int blocks = 2 * prop.multiProcessorCount; // use twice this
  printf("number of blocks: %d\n", blocks);
  //compute_hist<<<blocks, 256>>>(dev_buffer, SIZE, dev_hist);
  better_compute_hist<<<blocks, 256>>>(dev_buffer, SIZE, dev_hist);

  // copy result to here
  unsigned int hist[256];
  HANDLE_ERROR(cudaMemcpy(hist, dev_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));

  float elapsed_time;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf("Elapsed Time: %3.1f ms\n", elapsed_time);
  
  // verify results

  for (int i = 0; i < SIZE; ++i) {
    hist[buffer[i]]--;
  }

  for (int i = 0; i < 256; ++i) {
    if (hist[i] != 0) {
      printf("Incorrect entry at %d!\n", i);
    }
  }

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  cudaFree(dev_hist);
  cudaFree(dev_buffer);
  free(buffer);
  return 0; 
}

#include "common/book.h"
#include "common/cpu_anim.h"

#define DIM 16

struct DataBlock {
  unsigned char *dev_bitmap;
  CPUAnimBitmap *bitmap;
};

void cleanup(DataBlock *d) {
  cudaFree(d->dev_bitmap);
}

__global__ void kernel(unsigned char *bitmap, int ticks) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  int offset = x + y * blockDim.x * gridDim.x;

  // calculate color value at position to simulate ripple effect
  // formula used as it is
  float fx = x - DIM / 2;
  float fy = y - DIM / 2;
  float d = sqrtf(fx * fx + fy * fy);
  unsigned char grey = (unsigned char) (128.0f + 127.0f * cos(d/10.0f - ticks/7.0f) / (d/10.0f + 1.0f));
  bitmap[offset * 4 + 0] = grey;
  bitmap[offset * 4 + 1] = grey;
  bitmap[offset * 4 + 2] = grey;
  bitmap[offset * 4 + 3] = 255;
}

void generate_frame(DataBlock *d, int ticks) {
  // divide the 2D image into blocks and each block
  // has 16x16 threads
  dim3 blocks(DIM / 16, DIM / 16);
  dim3 threads(16, 16);
  kernel<<<blocks, threads>>>(d->dev_bitmap, ticks);
  HANDLE_ERROR(cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(), 
    cudaMemcpyDeviceToHost));
}

int main() {
  DataBlock data;
  CPUAnimBitmap bitmap(DIM, DIM, &data); // same result image here. see cpu_anim.h for more info
  data.bitmap = &bitmap;
  HANDLE_ERROR(cudaMalloc((void**) &data.dev_bitmap, bitmap.image_size()));
  // pass pointers to functions
  bitmap.anim_and_exit((void (*)(void*, int)) generate_frame, (void (*)(void*)) cleanup);
  return 0;
}

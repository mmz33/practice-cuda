#include "common/book.h"

#define DIM 16

struct DataBlock {
  unsigned char *dev_bitmap;
  CPUAnimBitmap *bitmap;
};

void cleanup(DataBlock *d) {
  cudaFree(d->dev_bitmap);
}

int main() {
  Datablock data;
  CPUAnimBitmap bitmap(DIM, DIM, &data);
  

  return 0;
}

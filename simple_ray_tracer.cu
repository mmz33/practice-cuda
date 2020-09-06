#include "stdio.h"
#include "common/book.h"
#include "common/cpu_bitmap.h"

// Here we will build a very simple ray tracer. Briefly, ray tracing is a technique
// that produce a 2D image containing 3D objects. For simpliciy here, the camera
// is restricted to z-axis facing the origin. Moreover, lightining will not be supported.
// The only objects are spheres. Also, only spheres closest to the camera can be seen 
// (in case ray hits multiple spheres in depth)

#define INF 2e10f
#define rnd(x) (x * rand() / RAND_MAX)
#define NUM_SPHERES 20
#define DIM 1024

struct Sphere {
  float r, g, b;
  float x, y, z;
  float radius;
 
  // given a ray shot from the pixel at position (ox, oy),
  // this function determines whether this ray hits this 
  // sphere or not. in case it hits, we return the distance
  // to the camera since we only care about the closest one
  __device__ float hit(float ox, float oy, float *n) {
    // distance to sphere
    int dx = ox - x;
    int dy = oy - y;
    // sphere equation: x^2 + y^2 + z^2 = r^2
    if (dx * dx + dy * dy < radius * radius) {
      float dz = sqrtf(radius * radius - dx * dx - dy * dy);
      *n = dz / sqrtf(radius * radius);
      return dz + z; // distance from camera
    }
    return -INF;
  }
};

__global__ void ray_trace(Sphere *s, unsigned char *ptr) {
  // each thread generate one output pixel
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;
  float ox = x - DIM / 2;
  float oy = y - DIM / 2;
  // check for hit
  float r = 0, g = 0, b = 0;
  float maxz = -INF;
  for (int i = 0; i < NUM_SPHERES; ++i) {
    float n;
    float t = s[i].hit(ox, oy, &n);
    if (t > maxz) {
      float fscale = n;
      r = s[i].r * fscale;
      g = s[i].g * fscale;
      b = s[i].b * fscale;
      maxz = t;
    }
  }
  ptr[offset * 4 + 0] = (int)(r * 255);
  ptr[offset * 4 + 1] = (int)(g * 255);
  ptr[offset * 4 + 2] = (int)(b * 255);
  ptr[offset * 4 + 3] = 255;
}

struct DataBlock {
  unsigned char *dev_bitmap;
  Sphere *s;
};

int main() {
  // capture start time
  cudaEvent_t start, stop;
  HANDLE_ERROR(cudaEventCreate(&start));
  HANDLE_ERROR(cudaEventCreate(&stop));
  HANDLE_ERROR(cudaEventRecord(start, 0));
 
  DataBlock data;
  CPUBitmap bitmap(DIM, DIM, &data);
  unsigned char *dev_bitmap;
  Sphere *s;

  // bitmap image to be filled later with output pixels as we ray trace
  HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size())); 
  // array of sphere objects
  HANDLE_ERROR(cudaMalloc((void**)&s, sizeof(Sphere) * NUM_SPHERES));

  // randomly generate sphere objects
  Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * NUM_SPHERES);
  for (int i = 0; i < NUM_SPHERES; ++i) {
    temp_s[i].r = rnd(1.0f);
    temp_s[i].g = rnd(1.0f);
    temp_s[i].b = rnd(1.0f);
    temp_s[i].x = rnd(1000.0f) - 500;
    temp_s[i].y = rnd(1000.0f) - 500;
    temp_s[i].z = rnd(1000.0f) - 500;
    temp_s[i].radius = rnd(100.0f) + 20;
  }

  HANDLE_ERROR(cudaMemcpy(s, temp_s, sizeof(Sphere) * NUM_SPHERES,
    cudaMemcpyHostToDevice));

  free(temp_s);

  dim3 grids(DIM / 16, DIM / 16);
  dim3 threads(16, 16);
  ray_trace<<<grids, threads>>>(s, dev_bitmap);
  
  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), 
    cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaEventRecord(stop, 0));
  HANDLE_ERROR(cudaEventSynchronize(stop));
  float elapsed_time;
  HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
  printf("Time to generate image: %3.1f ms\n", elapsed_time);

  HANDLE_ERROR(cudaEventDestroy(start));
  HANDLE_ERROR(cudaEventDestroy(stop));
  
  cudaFree(dev_bitmap);
  cudaFree(s);
  
  FILE *out = fopen("images/image1.ppm", "w");
  bitmap.output_ppm(out, 4);
  fclose(out);

  return 0;
}





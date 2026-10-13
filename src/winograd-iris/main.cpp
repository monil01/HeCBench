#include <hip/hip_runtime.h>
#include "utils.h"
#include "dmem.hpp"
int main(int argc, char* argv[]) {
  port::Session session(&argc, &argv);

  auto start = std::chrono::steady_clock::now();

  DATA_TYPE *A = (DATA_TYPE*)malloc(MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE));
  DATA_TYPE *B = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *B_outputFromGpu = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *C = (DATA_TYPE*)malloc(4 * 4 * sizeof(DATA_TYPE));

  for (int i = 0; i < MAP_SIZE; ++i)
    for (int j = 0; j < MAP_SIZE; ++j)
      A[i * MAP_SIZE + j] = rand() / (float)RAND_MAX;

  // transformed filter
  WinogradConv2D_2x2_filter_transformation(C);

  DATA_TYPE *d_A;
  port::allocate((void**)&d_A, MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE));
  port::copy(d_A, A, MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE), port::h2d);

  DATA_TYPE *d_B;
  port::allocate((void**)&d_B, (MAP_SIZE-2) * (MAP_SIZE-2) * sizeof(DATA_TYPE));

  DATA_TYPE *d_C;
  port::allocate((void**)&d_C, 16 * sizeof(DATA_TYPE));
  port::copy(d_C, C, 16 * sizeof(DATA_TYPE), port::h2d);

  const int tile_n = (MAP_SIZE - 2 + 1) / 2;

  // initial problem size
  size_t globalWorkSize[2] = {
    (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X,
    (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y };

  size_t localWorkSize[2] = {DIM_LOCAL_WORK_GROUP_X, DIM_LOCAL_WORK_GROUP_Y};

  // adjust problem size for co-run
  size_t cpu_global_size[2];
  size_t gpu_global_size[2];
  int global_offset[2];

  bool pass = true;

  // sweep over cpu_offset (accumulated in nanoseconds)
  long co_time = 0;

  for (int cpu_offset = 0; cpu_offset <= 100; cpu_offset++) {

    cpu_global_size[0] = cpu_offset * (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_X))
      / 100 * DIM_LOCAL_WORK_GROUP_X;
    cpu_global_size[1] = globalWorkSize[1];

    gpu_global_size[0] = globalWorkSize[0] - cpu_global_size[0];
    gpu_global_size[1] = globalWorkSize[1];

    global_offset[0] = cpu_global_size[0];
    global_offset[1] = 0;

    dim3 grid(gpu_global_size[0] / localWorkSize[0], gpu_global_size[1] / localWorkSize[1]);
    dim3 block(localWorkSize[0], localWorkSize[1]);

    bool cpu_run = false, gpu_run = false;
    if (cpu_global_size[0] > 0) {
      cpu_run = true;
    }
    if (gpu_global_size[0] > 0) {
      gpu_run = true;
    }

    // co-execution of host and device
    auto co_start = std::chrono::steady_clock::now();

    if (gpu_run) {
      port::launch("winograd_conv2d",grid,block,d_A, d_C, d_B, global_offset[0], global_offset[1]);
    }

    if (cpu_run) {
      //printf("CPU size: %d\n", cpu_global_size[0]);
      WinogradConv2D_2x2_omp(A, B, C, cpu_global_size);

      port::copy(d_B, B, gpu_run ? global_offset[0]*2*(MAP_SIZE-2)*sizeof(DATA_TYPE) :
          (MAP_SIZE-2)*(MAP_SIZE-2)*sizeof(DATA_TYPE), port::h2d);
    }

    port::copy(B_outputFromGpu, d_B, (MAP_SIZE-2) * (MAP_SIZE-2) * sizeof(DATA_TYPE), port::d2h);

    co_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - co_start).count();

#ifdef VERBOSE
    if (cpu_run) printf("run on host\n");
    if (gpu_run) printf("run on device\n");
    printf("CPU workload size : %d\n", cpu_offset);
#endif

    WinogradConv2D_2x2(A, B, C);
    pass &= compareResults(B, B_outputFromGpu);

  } // sweep

  printf("%s\n", pass ? "PASS" : "FAIL");

  port::release(d_A);
  port::release(d_B);
  port::release(d_C);
  free(A);
  free(B);
  free(B_outputFromGpu);
  free(C);

  auto end = std::chrono::steady_clock::now();
  long total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Co-execution time: %lf s\n", co_time * 1e-9);
  printf("Total time: %lf s\n", total_time * 1e-9);
  printf("Ratio of co-execution time to total time: %.2lf%%\n",
         100.0 * co_time / (double)total_time);

  return 0;
}

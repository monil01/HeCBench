/***********************************************************************
 * PathFinder uses dynamic programming to find a path on a 2-D grid from
 * the bottom row to the top row with the smallest accumulated weights,
 * where each step of the path moves straight ahead or diagonally ahead.
 * It iterates row by row, each node picks a neighboring node in the
 * previous row that has the smallest accumulated weight, and adds its
 * own weight to the sum.
 *
 * This kernel uses the technique of ghost zone optimization
 ***********************************************************************/

// Other header files.
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <chrono>
#include <iostream>
#include "iris_support.hpp"
#include <vector>
#include <cstring>


// halo width along one direction when advancing to the next iteration
#define HALO     1
#define STR_SIZE 256
#define DEVICE   0
#define M_SEED   9
#define IN_RANGE(x, min, max)  ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))


void fatal(char *s)
{
  fprintf(stderr, "error: %s\n", s);
}

int main(int argc, char** argv)
{
  // Program variables.
  int   rows, cols;
  int*  data;
  int** wall;
  int*  result;
  int   pyramid_height;

  if (argc == 4)
  {
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
    pyramid_height = atoi(argv[3]);
  }
  else
  {
    printf("Usage: %s <column length> <row length> <pyramid_height>\n", argv[0]);
    exit(0);
  }

  iris::Platform platform;
  int device = init_iris(platform, argc, argv);
  data = new int[rows * cols];
  wall = new int*[rows];
  for (int n = 0; n < rows; n++)
  {
    // wall[n] is set to be the nth row of the data array.
    wall[n] = data + cols * n;
  }
  result = new int[cols];

  int seed = M_SEED;
  srand(seed);

  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      wall[i][j] = rand() % 10;
    }
  }
#ifdef BENCH_PRINT
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%d ", wall[i][j]);
    }
    printf("\n");
  }
#endif

  // Pyramid parameters.
  const int borderCols = (pyramid_height) * HALO;

  /* printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
     pyramid_height, cols, borderCols, NUMBER_THREADS, blockCols, smallBlockCol); */

  int size = rows * cols; // the size (global work size) is a multiple of lws

  // running the opencl application shows lws=4000 (cpu) and lws=250 (gpu)
  int lws = 250;
  int* outputBuffer = (int*)calloc(16384, sizeof(int));
  int theHalo = HALO;

  auto start = std::chrono::steady_clock::now();

  std::vector<int> initial(data, data+cols), scratch(cols);
  iris::DMem wall_mem(data+cols, sizeof(int)*(size-cols));
  iris::DMem first(initial.data(), sizeof(int)*cols), second(scratch.data(), sizeof(int)*cols);
  iris::DMem debug(outputBuffer, sizeof(int)*16384);
  iris::DMem* d_gpuSrc = &first;
  iris::DMem* d_gpuResult = &second;
  auto kstart = std::chrono::steady_clock::now();

  for (int t = 0; t < rows - 1; t += pyramid_height)
  {
    // Calculate this for the kernel argument...
    int iteration = MIN(pyramid_height, rows-t-1);

    void* params[] = { &wall_mem, d_gpuSrc, d_gpuResult, &debug,
                       &iteration, &theHalo, const_cast<int*>(&borderCols), &cols, &t };
    int info[] = { iris_r, iris_r, iris_w, iris_rw,
                   sizeof(int), sizeof(int), sizeof(int), sizeof(int), sizeof(int) };
    launch_iris(device, "pathfinder", (size/lws)*lws, lws, 9, params, info);
    iris::DMem* temp = d_gpuResult;
    d_gpuResult = d_gpuSrc;
    d_gpuSrc = temp;
  }


  auto kend = std::chrono::steady_clock::now();
  auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
  printf("Total kernel execution time: %lf (s)\n", ktime * 1e-9);

  flush_iris(device, *d_gpuSrc);
  flush_iris(device, debug);
  std::memcpy(result, d_gpuSrc == &first ? initial.data() : scratch.data(), sizeof(int)*cols);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Device offloading time = %lf (s)\n", time * 1e-9);

  // add a null terminator at the end of the string.
  outputBuffer[16383] = '\0';

#ifdef BENCH_PRINT
  for (int i = 0; i < cols; i++)
    printf("%d ", data[i]);
  printf("\n");
  for (int i = 0; i < cols; i++)
    printf("%d ", result[i]);
  printf("\n");
#endif

  // Memory cleanup here.
  delete[] data;
  delete[] wall;
  delete[] result;
  free(outputBuffer);

  const int errors = platform.error_count();
  check_iris(platform.finalize());
  return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}

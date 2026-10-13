#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "dmem.hpp"
#include "common.h"
#include "reference.h"
void softmax_forward_baseline(float* out, const float* inp, int N, int C,
                            int warp_size, int block_size) {
  const int grid_size = ceil_div(N * warp_size, block_size);
  if (warp_size == 64)
    port::launch("softmax_baseline64",grid_size,block_size,out, inp, N, C);
  else
    port::launch("softmax_baseline32",grid_size,block_size,out, inp, N, C);
  hipCheck(iris_synchronize());
}

/*
void softmax_forward_online(float* out, const float* inp, int N, int C,
                            int warp_size, int block_size) {
  const int grid_size = ceil_div(N * warp_size, block_size);
  softmax_forward_online_kernel <<<grid_size, block_size >>> (out, inp, N, C);
  hipCheck(iris_synchronize());
}
*/

void softmax_forward_online2(float* out, const float* inp, int N, int C,
                             int warp_size, int block_size) {
  const int grid_size = ceil_div(N * warp_size, block_size);
  if (warp_size == 64)
    port::launch("softmax_online264",grid_size,block_size,out, inp, N, C);
  else
    port::launch("softmax_online232",grid_size,block_size,out, inp, N, C);
  hipCheck(iris_synchronize());
}

void softmax_forward_online3(float* out, const float* inp, int N, int C,
                             int warp_size, int block_size) {
  const int grid_size = N;
  port::launch("softmax_online3",grid_size,block_size,out, inp, N, C);
  hipCheck(iris_synchronize());
}

// kernel version dispatch
void softmax_forward(int kernel_num, float* out, const float* inp, int N, int C,
                     const int warp_size, const int block_size) {
  switch (kernel_num) {
    case 1:
      softmax_forward_baseline(out, inp, N, C, warp_size, block_size);
      break;
    case 2:
      port::launch("softmax_online", ceil_div(N * 32, block_size), block_size, out, inp, N, C);
      break;
    case 3:
      softmax_forward_online2(out, inp, N, C, warp_size, block_size);
      break;
    case 4:
      softmax_forward_online3(out, inp, N, C, warp_size, block_size);
      break;
    default:
      printf("Invalid kernel number\n");
      exit(1);
  }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
  port::Session session(&argc, &argv);
  srand(0);

  int B = getenv("HECBENCH_B") ? atoi(getenv("HECBENCH_B")) : 8;
  int T = getenv("HECBENCH_T") ? atoi(getenv("HECBENCH_T")) : 1024;
  int V = getenv("HECBENCH_V") ? atoi(getenv("HECBENCH_V")) : 50257;

  // create host memory of random numbers
  float* out = (float*)malloc(B * T * V * sizeof(float));
  float* inp = make_random_float(B * T * V);

  // make the input less uniformly random: Otherwise, all probabilities will be basically zero,
  // and the tests are not actually meaningful.
  const int* outliers = make_random_int(B * T * 3, V);
  for(int k = 0; k < 3; ++k) {
    for(int j = 0; j < B * T; ++j) {
      inp[j * V + outliers[j*3 + k]] *= 20;
    }
  }

  // move to GPU
  float* d_out;
  float* d_inp;
  hipCheck(port::allocate((void**)&d_out, B * T * V * sizeof(float)));
  hipCheck(port::allocate((void**)&d_inp, B * T * V * sizeof(float)));
  hipCheck(port::copy(d_inp, inp, B * T * V * sizeof(float), port::h2d));

  // read kernel_num from command line
  int kernel_num = 2;
  if (argc > 1) {
    kernel_num = atoi(argv[1]);
  }
  if (kernel_num > 1)
    printf("Using kernel online %d\n", kernel_num);
  else
    printf("Using kernel baseline %d\n", kernel_num);


  // query the warp size
  int warp_size;
  warp_size = 32; // Preserve CUDA logical warp size on HIP wave64.

  softmax_forward_cpu(out, inp, B * T, V);
  {
    float max_el = -INFINITY;
    for(int i = 0; i <  B * T * V; ++i) {
      max_el = fmaxf(max_el, out[i]);
    }
    assert(max_el > 1e-4);
    printf("Largest output is: %f\n", max_el);
  }

  // first check the correctness of the kernel
  for (int j = warp_size; j <= 1024; j = j * 2) {
    int block_size = j;
    printf("Checking block size %d.\n", block_size);
    softmax_forward(kernel_num, d_out, d_inp, B * T, V, warp_size, block_size);
    validate_result(d_out, out, "out", B * T * V, 1e-4f);
  }

  printf("All results match. Starting benchmarks.\n\n");

  // time the kernel at different block sizes
  for (int j = warp_size; j <= 1024; j = j * 2) {
    int block_size = j;
    int repeat_times = getenv("HECBENCH_REPEAT") ? atoi(getenv("HECBENCH_REPEAT")) : 100;
    float elapsed_time = benchmark_kernel(repeat_times, softmax_forward,
                                          kernel_num, d_out, d_inp, B * T, V,
                                          warp_size, block_size);
    printf("block_size %4d | time %.4f ms | per token %.2f µs\n", block_size, elapsed_time, elapsed_time * 1'000 / (B*T));
  }

  // free memory
  free(out);
  free(inp);
  free((void*)outliers);
  hipCheck(port::release(d_out));
  hipCheck(port::release(d_inp));

  return 0;
}

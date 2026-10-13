#include "dmem_runtime.hpp"
/*
Kernels for layernorm forward pass.
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "common.h"
#include "reference.h"
#include "dispatch.h"

void layernorm_forward0(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = N;
    constexpr int UNROLL = sizeof(float4) / sizeof(float);
    if (C % UNROLL == 0)
      dispatch_layernorm_forward_kernel0_4(dim3(grid_size),dim3(block_size),out, mean, rstd, inp, weight, bias, N, C);
    else
      dispatch_layernorm_forward_kernel0_1(dim3(grid_size),dim3(block_size),out, mean, rstd, inp, weight, bias, N, C);
    iris_check(iris_sync());
}

void layernorm_forward1(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    dispatch_layernorm_forward_kernel1( dim3(grid_size), dim3(block_size), out, mean, rstd, inp, weight, bias, N, C);
    iris_check(iris_sync());
}

void layernorm_forward2(float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int B, int T, int C,
                       const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    dispatch_layernorm_forward_kernel2( dim3(grid_size), dim3(block_size), out, mean, rstd, inp, weight, bias, N, C);
    iris_check(iris_sync());
}


// kernel version dispatch
void layernorm_forward(int kernel_num,
                    float* out, float* mean, float* rstd,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C,
                    const int block_size) {
    switch (kernel_num) {
        case 0:
            layernorm_forward0(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 1:
            layernorm_forward1(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        case 2:
            layernorm_forward2(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
  if (argc != 5) {
    printf("Usage: %s <batch size> <sequence length> <channel length> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t B = atoi(argv[1]);
  const size_t T = atoi(argv[2]);
  const size_t C = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  // create host memory of random numbers
  srand(0);
  float* out = (float*)malloc(B * T * C * sizeof(float));
  float* mean = (float*)malloc(B * T * sizeof(float));
  float* rstd = (float*)malloc(B * T * sizeof(float));
  float* inp = make_random_float(B * T * C);
  float* weight = make_random_float(C);
  float* bias = make_random_float(C);

  // move to GPU
  float* d_out;
  float* d_mean;
  float* d_rstd;
  float* d_inp;
  float* d_weight;
  float* d_bias;
  iris_check(iris_alloc(&d_out, B * T * C * sizeof(float)));
  iris_check(iris_alloc(&d_mean, B * T * sizeof(float)));
  iris_check(iris_alloc(&d_rstd, B * T * sizeof(float)));
  iris_check(iris_alloc(&d_inp, B * T * C * sizeof(float)));
  iris_check(iris_alloc(&d_weight, C * sizeof(float)));
  iris_check(iris_alloc(&d_bias, C * sizeof(float)));
  iris_check(iris_copy(d_inp, inp, B * T * C * sizeof(float), iris_h2d));
  iris_check(iris_copy(d_weight, weight, C * sizeof(float), iris_h2d));
  iris_check(iris_copy(d_bias, bias, C * sizeof(float), iris_h2d));

  const int block_sizes[] = {32, 64, 128, 256, 512, 1024};

  // evaluate kernels
  for (int kernel_num = 0; kernel_num < 3; kernel_num++) {
    printf("Using kernel %d\n", kernel_num);

    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    // check the correctness of the kernel at all block sizes
    for (int block_size : block_sizes) {
      printf("Checking block size %d.\n", block_size);

      layernorm_forward(kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);

      validate_result(d_out, out, "out", B * T * C, 1e-5f);
      validate_result(d_mean, mean, "mean", B * T, 1e-5f);
      validate_result(d_rstd, rstd, "rstd", B * T, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int block_size : block_sizes) {
      float elapsed_time = benchmark_kernel(repeat, layernorm_forward,
                                            kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias,
                                            B, T, C, block_size);

      // estimate the memory bandwidth achieved
      // e.g. A100 40GB PCIe is advertised at 1,555GB/s
      long memory_ops = (2 * B * T * C) * 4; // *4 for float
      float memory_bandwidth = memory_ops / elapsed_time / 1e6;

      printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }
  }

  // free memory
  free(out);
  free(mean);
  free(rstd);
  free(inp);
  free(weight);
  free(bias);
  iris_check(iris_free(d_out));
  iris_check(iris_free(d_mean));
  iris_check(iris_free(d_rstd));
  iris_check(iris_free(d_inp));
  iris_check(iris_free(d_weight));
  iris_check(iris_free(d_bias));

  iris_finish();
  return 0;
}

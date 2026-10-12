#include <hip/hip_runtime.h>

#include <cstdint>

__device__ double LCG_random_double(uint64_t* seed)
{
  const uint64_t m = 9223372036854775808ULL;
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return static_cast<double>(*seed) / static_cast<double>(m);
}

extern "C" __global__ void exact_rows_kernel(float* __restrict__ x,
                                             int x_base,
                                             int nrows_x,
                                             int ncols,
                                             const float* __restrict__ background,
                                             int nrows_background,
                                             float* __restrict__ dataset,
                                             int dataset_base,
                                             const float* __restrict__ observation)
{
  (void)nrows_x;
  const int row_block = blockIdx.x;
  int col = threadIdx.x;
  while (col < ncols) {
    const int curr_x = static_cast<int>(x[x_base + row_block * ncols + col]);
    for (int row_idx = row_block * nrows_background;
         row_idx < row_block * nrows_background + nrows_background;
         row_idx++) {
      const int out = dataset_base + row_idx * ncols + col;
      if (curr_x == 0) {
        dataset[out] = background[(row_idx % nrows_background) * ncols + col];
      } else {
        dataset[out] = observation[col];
      }
    }
    col += blockDim.x;
  }
}

extern "C" __global__ void sampled_rows_kernel(const int* __restrict__ nsamples,
                                               float* __restrict__ x,
                                               int x_base,
                                               int nrows_x,
                                               int ncols,
                                               const float* __restrict__ background,
                                               int nrows_background,
                                               float* __restrict__ dataset,
                                               int dataset_base,
                                               const float* __restrict__ observation,
                                               uint64_t seed)
{
  (void)nrows_x;
  const int k_blk = nsamples[blockIdx.x];
  if (threadIdx.x < k_blk) {
    int rand_idx = static_cast<int>(LCG_random_double(&seed) * ncols);
    while (atomicExch(&x[x_base + 2 * blockIdx.x * ncols + rand_idx], 1.0f) == 1.0f) {
      rand_idx = static_cast<int>(LCG_random_double(&seed) * ncols);
    }
  }
  __syncthreads();

  int col_idx = threadIdx.x;
  while (col_idx < ncols) {
    const int curr_x = static_cast<int>(x[x_base + 2 * blockIdx.x * ncols + col_idx]);
    x[x_base + (2 * blockIdx.x + 1) * ncols + col_idx] = 1.0f - curr_x;

    for (int bg_row_idx = 2 * blockIdx.x * nrows_background;
         bg_row_idx < 2 * blockIdx.x * nrows_background + nrows_background;
         bg_row_idx++) {
      const int out = dataset_base + bg_row_idx * ncols + col_idx;
      if (curr_x == 0) {
        dataset[out] = background[(bg_row_idx % nrows_background) * ncols + col_idx];
      } else {
        dataset[out] = observation[col_idx];
      }
    }

    for (int bg_row_idx = (2 * blockIdx.x + 1) * nrows_background;
         bg_row_idx < (2 * blockIdx.x + 1) * nrows_background + nrows_background;
         bg_row_idx++) {
      const int out = dataset_base + bg_row_idx * ncols + col_idx;
      if (curr_x == 0) {
        dataset[out] = observation[col_idx];
      } else {
        dataset[out] = background[(bg_row_idx % nrows_background) * ncols + col_idx];
      }
    }

    col_idx += blockDim.x;
  }
}

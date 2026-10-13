#include <hip/hip_runtime.h>
#include "constants.h"
namespace mean_shift::gpu {
  extern "C" __global__ void mean_shift(const float *data, float *data_next) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
      size_t row = tid * D;
      float new_position[D] = {0.f};
      float tot_weight = 0.f;
      for (size_t i = 0; i < N; ++i) {
        size_t row_n = i * D;
        float sq_dist = 0.f;
        for (size_t j = 0; j < D; ++j) {
          sq_dist += (data[row + j] - data[row_n + j]) * (data[row + j] - data[row_n + j]);
        }
        if (sq_dist <= RADIUS) {
          float weight = expf(-sq_dist / DBL_SIGMA_SQ);
          for (size_t j = 0; j < D; ++j) {
            new_position[j] += weight * data[row_n + j];
          }
          tot_weight += weight;
        }
      }
      for (size_t j = 0; j < D; ++j) {
        data_next[row + j] = new_position[j] / tot_weight;
      }
    }
  }

  extern "C" __global__ void mean_shift_tiling(const float* data, float* data_next) {

    // Shared memory allocation
    __shared__ float local_data[TILE_WIDTH * D];
    __shared__ float valid_data[TILE_WIDTH];
    // A few convenient variables
    int lid = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + lid;
    int row = (tid < N ? tid : 0) * D;
    int local_row = lid * D;
    float new_position[D] = {0.f};
    float tot_weight = 0.f;
    // Load data in shared memory
    for (int t = 0; t < BLOCKS; ++t) {
      int tid_in_tile = t * TILE_WIDTH + lid;
      if (tid_in_tile < N) {
        int row_in_tile = tid_in_tile * D;
        for (int j = 0; j < D; ++j) {
          local_data[local_row + j] = data[row_in_tile + j];
        }
        valid_data[lid] = 1;
      }
      else {
        for (int j = 0; j < D; ++j) {
          local_data[local_row + j] = 0;
        }
        valid_data[lid] = 0;
      }
      __syncthreads();
      for (int i = 0; i < TILE_WIDTH; ++i) {
        int local_row_tile = i * D;
        float valid_radius = RADIUS * valid_data[i];
        float sq_dist = 0.;
        for (int j = 0; j < D; ++j) {
          sq_dist += (data[row + j] - local_data[local_row_tile + j]) *
                     (data[row + j] - local_data[local_row_tile + j]);
        }
        if (sq_dist <= valid_radius) {
          float weight = expf(-sq_dist / DBL_SIGMA_SQ);
          for (int j = 0; j < D; ++j) {
            new_position[j] += (weight * local_data[local_row_tile + j]);
          }
          tot_weight += (weight * valid_data[i]);
        }
      }
      __syncthreads();
    }
    if (tid < N) {
      for (int j = 0; j < D; ++j) {
        data_next[row + j] = new_position[j] / tot_weight;
      }
    }
  }
}

#include <float.h>
#include <stdint.h>

__device__ float positiveInfinity(float) { return FLT_MAX; }
__device__ double positiveInfinity(double) { return DBL_MAX; }
__device__ uint32_t positiveInfinity(uint32_t) { return 0xFFFFFFFFUL; }
__device__ uint64_t positiveInfinity(uint64_t) { return 0xFFFFFFFFFFFFFFFFUL; }

__device__ float negativeInfinity(float) { return FLT_MIN; }
__device__ double negativeInfinity(double) { return DBL_MIN; }
__device__ uint32_t negativeInfinity(uint32_t) { return 0; }
__device__ uint64_t negativeInfinity(uint64_t) { return 0; }

template <typename vec_t>
__device__ vec_t getPositiveInfinity() {
  vec_t tmp = 0;
  return positiveInfinity(tmp);
}

template <typename vec_t>
__device__ vec_t getNegativeInfinity() {
  vec_t tmp = 0;
  return negativeInfinity(tmp);
}

#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

template <typename vec_t>
__device__ void workloadDiagonals_impl(
    const vec_t* __restrict__ A, uint32_t A_length,
    const vec_t* __restrict__ B, uint32_t B_length,
    uint32_t* __restrict__ diagonal_path_intersections) {
  volatile __shared__ int32_t x_top, y_top, x_bottom, y_bottom, found;
  __shared__ int32_t oneorzero[32];

  int threadOffset = threadIdx.x - 16;

  if (threadIdx.x == 0) {
    int32_t combinedIndex =
        (uint64_t)blockIdx.x * ((uint64_t)A_length + (uint64_t)B_length) /
        (uint64_t)gridDim.x;

    x_top = MIN(combinedIndex, A_length);
    y_top = combinedIndex > (A_length) ? combinedIndex - (A_length) : 0;
    x_bottom = y_top;
    y_bottom = x_top;

    found = 0;
  }
  __syncthreads();

  while (!found) {
    int32_t current_x = x_top - ((x_top - x_bottom) >> 1) - threadOffset;
    int32_t current_y = y_top + ((y_bottom - y_top) >> 1) + threadOffset;

    if (current_x >= (int32_t)A_length || current_y < 0) {
      oneorzero[threadIdx.x] = 0;
    } else if (current_y >= (int32_t)B_length || current_x < 1) {
      oneorzero[threadIdx.x] = 1;
    } else {
      oneorzero[threadIdx.x] = (A[current_x - 1] <= B[current_y]) ? 1 : 0;
    }

    __syncthreads();

    if (threadIdx.x > 0 &&
        (oneorzero[threadIdx.x] != oneorzero[threadIdx.x - 1])) {
      found = 1;
      diagonal_path_intersections[blockIdx.x] = current_x;
      diagonal_path_intersections[blockIdx.x + gridDim.x + 1] = current_y;
    }

    __syncthreads();

    if (threadIdx.x == 16) {
      if (oneorzero[31] != 0) {
        x_bottom = current_x;
        y_bottom = current_y;
      } else {
        x_top = current_x;
        y_top = current_y;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    diagonal_path_intersections[0] = 0;
    diagonal_path_intersections[gridDim.x + 1] = 0;
    diagonal_path_intersections[gridDim.x] = A_length;
    diagonal_path_intersections[gridDim.x + gridDim.x + 1] = B_length;
  }
}

#define K 512
template <typename vec_t>
__device__ void mergeSinglePath_impl(
    const vec_t* __restrict__ A, uint32_t A_length,
    const vec_t* __restrict__ B, uint32_t B_length,
    const uint32_t* __restrict__ diagonal_path_intersections,
    vec_t* __restrict__ C, uint32_t C_length) {
  __shared__ vec_t A_shared[(K + 2) << 1];
  vec_t* B_shared = A_shared + K + 2;

  volatile __shared__ uint32_t x_block_top, y_block_top, x_block_stop,
      y_block_stop;

  uint32_t threadIdX4 = threadIdx.x + threadIdx.x;
  threadIdX4 = threadIdX4 + threadIdX4;
  uint32_t threadIdX4p1 = threadIdX4 + 1;
  uint32_t threadIdX4p2 = threadIdX4p1 + 1;
  uint32_t threadIdX4p3 = threadIdX4p2 + 1;

  switch (threadIdx.x) {
    case 0:
      x_block_top = diagonal_path_intersections[blockIdx.x];
      A_shared[0] = getNegativeInfinity<vec_t>();
      break;
    case 64:
      y_block_top = diagonal_path_intersections[blockIdx.x + gridDim.x + 1];
      A_shared[K + 1] = getPositiveInfinity<vec_t>();
      break;
    case 32:
      x_block_stop = diagonal_path_intersections[blockIdx.x + 1];
      B_shared[0] = getNegativeInfinity<vec_t>();
      break;
    case 96:
      y_block_stop = diagonal_path_intersections[blockIdx.x + gridDim.x + 2];
      B_shared[K + 1] = getPositiveInfinity<vec_t>();
      break;
    default:
      break;
  }

  A--;
  B--;

  __syncthreads();

  while (((x_block_top < x_block_stop) || (y_block_top < y_block_stop))) {
    const vec_t* Atemp = A + x_block_top;
    const vec_t* Btemp = B + y_block_top;
    uint32_t sharedX = threadIdx.x + 1;

    A_shared[sharedX] = Atemp[sharedX];
    B_shared[sharedX] = Btemp[sharedX];
    sharedX += blockDim.x;
    A_shared[sharedX] = Atemp[sharedX];
    B_shared[sharedX] = Btemp[sharedX];
    sharedX += blockDim.x;
    A_shared[sharedX] = Atemp[sharedX];
    B_shared[sharedX] = Btemp[sharedX];
    sharedX += blockDim.x;
    A_shared[sharedX] = Atemp[sharedX];
    B_shared[sharedX] = Btemp[sharedX];

    vec_t* Ctemp = C + x_block_top + y_block_top;

    __syncthreads();

    int32_t offset = threadIdX4 >> 1;
    uint32_t Ax = offset + 1;
    vec_t* BSm1 = B_shared + threadIdX4p2;
    vec_t* BS = BSm1 + 1;
    while (true) {
      offset = ((offset + 1) >> 1);
      if (A_shared[Ax] > BSm1[~Ax]) {
        if (A_shared[Ax - 1] <= BS[~Ax]) {
          break;
        }
        Ax -= offset;
      } else {
        Ax += offset;
      }
    }

    uint32_t Bx = threadIdX4p2 - Ax;

    vec_t Ai, Bi, Ci;
    Ai = A_shared[Ax];
    Bi = B_shared[Bx];
    if (Ai > Bi) {
      Ci = Bi;
      Bx++;
      Bi = B_shared[Bx];
    } else {
      Ci = Ai;
      Ax++;
      Ai = A_shared[Ax];
    }
    Ctemp[threadIdX4] = Ci;
    if (Ai > Bi) {
      Ci = Bi;
      Bx++;
      Bi = B_shared[Bx];
    } else {
      Ci = Ai;
      Ax++;
      Ai = A_shared[Ax];
    }
    Ctemp[threadIdX4p1] = Ci;
    if (Ai > Bi) {
      Ci = Bi;
      Bx++;
      Bi = B_shared[Bx];
    } else {
      Ci = Ai;
      Ax++;
      Ai = A_shared[Ax];
    }
    Ctemp[threadIdX4p2] = Ci;
    Ctemp[threadIdX4p3] = Ai > Bi ? Bi : Ai;

    if (threadIdx.x == 127) {
      x_block_top += Ax - 1;
      y_block_top += Bx - 1;
    }
    __syncthreads();
  }
}

extern "C" __global__ void workloadDiagonals_u32(
    const uint32_t* A, uint32_t A_length, const uint32_t* B, uint32_t B_length,
    uint32_t* diagonal_path_intersections) {
  workloadDiagonals_impl<uint32_t>(A, A_length, B, B_length,
                                   diagonal_path_intersections);
}

extern "C" __global__ void workloadDiagonals_f32(
    const float* A, uint32_t A_length, const float* B, uint32_t B_length,
    uint32_t* diagonal_path_intersections) {
  workloadDiagonals_impl<float>(A, A_length, B, B_length,
                                diagonal_path_intersections);
}

extern "C" __global__ void workloadDiagonals_u64(
    const uint64_t* A, uint32_t A_length, const uint64_t* B, uint32_t B_length,
    uint32_t* diagonal_path_intersections) {
  workloadDiagonals_impl<uint64_t>(A, A_length, B, B_length,
                                   diagonal_path_intersections);
}

extern "C" __global__ void workloadDiagonals_f64(
    const double* A, uint32_t A_length, const double* B, uint32_t B_length,
    uint32_t* diagonal_path_intersections) {
  workloadDiagonals_impl<double>(A, A_length, B, B_length,
                                 diagonal_path_intersections);
}

extern "C" __global__ void mergeSinglePath_u32(
    const uint32_t* A, uint32_t A_length, const uint32_t* B, uint32_t B_length,
    const uint32_t* diagonal_path_intersections, uint32_t* C,
    uint32_t C_length) {
  mergeSinglePath_impl<uint32_t>(A, A_length, B, B_length,
                                 diagonal_path_intersections, C, C_length);
}

extern "C" __global__ void mergeSinglePath_f32(
    const float* A, uint32_t A_length, const float* B, uint32_t B_length,
    const uint32_t* diagonal_path_intersections, float* C, uint32_t C_length) {
  mergeSinglePath_impl<float>(A, A_length, B, B_length,
                              diagonal_path_intersections, C, C_length);
}

extern "C" __global__ void mergeSinglePath_u64(
    const uint64_t* A, uint32_t A_length, const uint64_t* B, uint32_t B_length,
    const uint32_t* diagonal_path_intersections, uint64_t* C,
    uint32_t C_length) {
  mergeSinglePath_impl<uint64_t>(A, A_length, B, B_length,
                                 diagonal_path_intersections, C, C_length);
}

extern "C" __global__ void mergeSinglePath_f64(
    const double* A, uint32_t A_length, const double* B, uint32_t B_length,
    const uint32_t* diagonal_path_intersections, double* C,
    uint32_t C_length) {
  mergeSinglePath_impl<double>(A, A_length, B, B_length,
                               diagonal_path_intersections, C, C_length);
}

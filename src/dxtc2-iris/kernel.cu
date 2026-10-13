#include <cuda_runtime.h>
#include <float.h>
#include <cooperative_groups.h>
#include "helper_math.h"
namespace cg=cooperative_groups;
#define NUM_THREADS 64
// Use power method to find the first eigenvector.
// https://en.wikipedia.org/wiki/Power_iteration
inline __device__ __host__ float3 firstEigenVector(float matrix[6]) {
  // 8 iterations seems to be more than enough.

  float3 v = make_float3(1.0f, 1.0f, 1.0f);

  for (int i = 0; i < 8; i++) {
    float x = v.x * matrix[0] + v.y * matrix[1] + v.z * matrix[2];
    float y = v.x * matrix[1] + v.y * matrix[3] + v.z * matrix[4];
    float z = v.x * matrix[2] + v.y * matrix[4] + v.z * matrix[5];
    float m = max(max(x, y), z);
    float iv = 1.0f / m;
    v = make_float3(x * iv, y * iv, z * iv);
  }

  return v;
}

template <class T>
__device__ inline void swap(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

////////////////////////////////////////////////////////////////////////////////
// Round color to RGB565 and expand
////////////////////////////////////////////////////////////////////////////////
inline __device__ float3 roundAndExpand(float3 v, ushort *w) {
  v.x = rintf(__saturatef(v.x) * 31.0f);
  v.y = rintf(__saturatef(v.y) * 63.0f);
  v.z = rintf(__saturatef(v.z) * 31.0f);

  *w = ((ushort)v.x << 11) | ((ushort)v.y << 5) | (ushort)v.z;
  v.x *= 0.03227752766457f;  // approximate integer bit expansion.
  v.y *= 0.01583151765563f;
  v.z *= 0.03227752766457f;
  return v;
}

__constant__ float alphaTable4[4] = {9.0f, 0.0f, 6.0f, 3.0f};
__constant__ float alphaTable3[4] = {4.0f, 0.0f, 2.0f, 2.0f};
__constant__ const int prods4[4] = {0x090000, 0x000900, 0x040102, 0x010402};
__constant__ const int prods3[4] = {0x040000, 0x000400, 0x040101, 0x010401};

#define USE_TABLES 1

////////////////////////////////////////////////////////////////////////////////
// Evaluate permutations
////////////////////////////////////////////////////////////////////////////////
static __device__ float evalPermutation4(const float3 *colors, uint permutation,
                                         ushort *start, ushort *end,
                                         float3 color_sum) {
// Compute endpoints using least squares.
#if USE_TABLES
  float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

  int akku = 0;

  // Compute alpha & beta for this permutation.
  for (int i = 0; i < 16; i++) {
    const uint bits = permutation >> (2 * i);

    alphax_sum += alphaTable4[bits & 3] * colors[i];
    akku += prods4[bits & 3];
  }

  float alpha2_sum = float(akku >> 16);
  float beta2_sum = float((akku >> 8) & 0xff);
  float alphabeta_sum = float((akku >> 0) & 0xff);
  float3 betax_sum = (9.0f * color_sum) - alphax_sum;
#else
  float alpha2_sum = 0.0f;
  float beta2_sum = 0.0f;
  float alphabeta_sum = 0.0f;
  float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

  // Compute alpha & beta for this permutation.
  for (int i = 0; i < 16; i++) {
    const uint bits = permutation >> (2 * i);

    float beta = (bits & 1);

    if (bits & 2) {
      beta = (1 + beta) * (1.0f / 3.0f);
    }

    float alpha = 1.0f - beta;

    alpha2_sum += alpha * alpha;
    beta2_sum += beta * beta;
    alphabeta_sum += alpha * beta;
    alphax_sum += alpha * colors[i];
  }

  float3 betax_sum = color_sum - alphax_sum;
#endif

  // alpha2, beta2, alphabeta and factor could be precomputed for each
  // permutation, but it's faster to recompute them.
  const float factor =
      1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

  float3 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
  float3 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

  // Round a, b to the closest 5-6-5 color and expand...
  a = roundAndExpand(a, start);
  b = roundAndExpand(b, end);

  // compute the error
  float3 e = a * a * alpha2_sum + b * b * beta2_sum +
             2.0f * (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum);

  return (0.111111111111f) * (e.x + e.y + e.z);
}

static __device__ float evalPermutation3(const float3 *colors, uint permutation,
                                         ushort *start, ushort *end,
                                         float3 color_sum) {
// Compute endpoints using least squares.
#if USE_TABLES
  float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

  int akku = 0;

  // Compute alpha & beta for this permutation.
  for (int i = 0; i < 16; i++) {
    const uint bits = permutation >> (2 * i);

    alphax_sum += alphaTable3[bits & 3] * colors[i];
    akku += prods3[bits & 3];
  }

  float alpha2_sum = float(akku >> 16);
  float beta2_sum = float((akku >> 8) & 0xff);
  float alphabeta_sum = float((akku >> 0) & 0xff);
  float3 betax_sum = (4.0f * color_sum) - alphax_sum;
#else
  float alpha2_sum = 0.0f;
  float beta2_sum = 0.0f;
  float alphabeta_sum = 0.0f;
  float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

  // Compute alpha & beta for this permutation.
  for (int i = 0; i < 16; i++) {
    const uint bits = permutation >> (2 * i);

    float beta = (bits & 1);

    if (bits & 2) {
      beta = 0.5f;
    }

    float alpha = 1.0f - beta;

    alpha2_sum += alpha * alpha;
    beta2_sum += beta * beta;
    alphabeta_sum += alpha * beta;
    alphax_sum += alpha * colors[i];
  }

  float3 betax_sum = color_sum - alphax_sum;
#endif

  const float factor =
      1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

  float3 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
  float3 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

  // Round a, b to the closest 5-6-5 color and expand...
  a = roundAndExpand(a, start);
  b = roundAndExpand(b, end);

  // compute the error
  float3 e = a * a * alpha2_sum + b * b * beta2_sum +
             2.0f * (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum);

  return (0.25f) * (e.x + e.y + e.z);
}

__device__ void evalAllPermutations(const float3 *colors,
                                    const uint *permutations, ushort &bestStart,
                                    ushort &bestEnd, uint &bestPermutation,
                                    float *errors, float3 color_sum,
                                    cg::thread_block cta) {
  const int idx = threadIdx.x;

  float bestError = FLT_MAX;

  __shared__ uint s_permutations[160];

  for (int i = 0; i < 16; i++) {
    int pidx = idx + NUM_THREADS * i;

    if (pidx >= 992) {
      break;
    }

    ushort start, end;
    uint permutation = permutations[pidx];

    if (pidx < 160) {
      s_permutations[pidx] = permutation;
    }

    float error =
        evalPermutation4(colors, permutation, &start, &end, color_sum);

    if (error < bestError) {
      bestError = error;
      bestPermutation = permutation;
      bestStart = start;
      bestEnd = end;
    }
  }

  if (bestStart < bestEnd) {
    swap(bestEnd, bestStart);
    bestPermutation ^= 0x55555555;  // Flip indices.
  }

  cg::sync(cta);  // Sync here to ensure s_permutations is valid going forward

  for (int i = 0; i < 3; i++) {
    int pidx = idx + NUM_THREADS * i;

    if (pidx >= 160) {
      break;
    }

    ushort start, end;
    uint permutation = s_permutations[pidx];
    float error =
        evalPermutation3(colors, permutation, &start, &end, color_sum);

    if (error < bestError) {
      bestError = error;
      bestPermutation = permutation;
      bestStart = start;
      bestEnd = end;

      if (bestStart > bestEnd) {
        swap(bestEnd, bestStart);
        bestPermutation ^=
            (~bestPermutation >> 1) & 0x55555555;  // Flip indices.
      }
    }
  }

  errors[idx] = bestError;
}

////////////////////////////////////////////////////////////////////////////////
// Find index with minimum error
////////////////////////////////////////////////////////////////////////////////
__device__ int findMinError(float *errors, cg::thread_block cta) {
  const int idx = threadIdx.x;
  __shared__ int indices[NUM_THREADS];
  indices[idx] = idx;

  cg::sync(cta);

  for (int d = NUM_THREADS / 2; d > 0; d >>= 1) {
    float err0 = errors[idx];
    float err1 = (idx + d) < NUM_THREADS ? errors[idx + d] : FLT_MAX;
    int index1 = (idx + d) < NUM_THREADS ? indices[idx + d] : 0;

    cg::sync(cta);

    if (err1 < err0) {
      errors[idx] = err1;
      indices[idx] = index1;
    }

    cg::sync(cta);
  }

  return indices[0];
}

////////////////////////////////////////////////////////////////////////////////
// Save DXT block
////////////////////////////////////////////////////////////////////////////////
__device__ void saveBlockDXT1(ushort start, ushort end, uint permutation,
                              int xrefs[16], uint2 *result, int blockOffset) {
  const int bid = blockIdx.x + blockOffset;

  if (start == end) {
    permutation = 0;
  }

  // Reorder permutation.
  uint indices = 0;

  for (int i = 0; i < 16; i++) {
    int ref = xrefs[i];
    indices |= ((permutation >> (2 * ref)) & 3) << (2 * i);
  }

  // Write endpoints.
  result[bid].x = (end << 16) | start;

  // Write palette indices.
  result[bid].y = indices;
}

////////////////////////////////////////////////////////////////////////////////
// Compress color block
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void compress(const uint *permutations, const uint *image,
                         uint2 *result, int blockOffset) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  const int idx = threadIdx.x;
  const int bid = blockIdx.x + blockOffset;

  __shared__ float3 sums;
  __shared__ float3 colors[16];
  __shared__ int xrefs[16];
  __shared__ float dps[16];
  __shared__ float covariance[16 * 6];

  if (idx < 16) {
    // Read color and copy to shared mem.
    uint c = image[(bid)*16 + idx];

    colors[idx].x = ((c >> 0) & 0xFF) * (1.0f / 255.0f);
    colors[idx].y = ((c >> 8) & 0xFF) * (1.0f / 255.0f);
    colors[idx].z = ((c >> 16) & 0xFF) * (1.0f / 255.0f);
  }

  cg::sync(cta);

  if (idx == 0) {
    sums = colors[0];
    for (int i = 1; i < 16; i++)
      sums += colors[i];
  }

  cg::sync(cta);

  if (idx < 16) {
    // Sort colors along the best fit line.
    float3 diff = colors[idx] - sums * (1.0f / 16.0f);
    covariance[6 * idx + 0] = diff.x * diff.x;  // 0, 6, 12, 2, 8, 14, 4, 10, 0
    covariance[6 * idx + 1] = diff.x * diff.y;
    covariance[6 * idx + 2] = diff.x * diff.z;
    covariance[6 * idx + 3] = diff.y * diff.y;
    covariance[6 * idx + 4] = diff.y * diff.z;
    covariance[6 * idx + 5] = diff.z * diff.z;
  }

  cg::sync(cta);

  for (int d = 8; d > 0; d >>= 1) {
    if (idx < d) {
      covariance[6 * idx + 0] += covariance[6 * (idx + d) + 0];
      covariance[6 * idx + 1] += covariance[6 * (idx + d) + 1];
      covariance[6 * idx + 2] += covariance[6 * (idx + d) + 2];
      covariance[6 * idx + 3] += covariance[6 * (idx + d) + 3];
      covariance[6 * idx + 4] += covariance[6 * (idx + d) + 4];
      covariance[6 * idx + 5] += covariance[6 * (idx + d) + 5];
    }
    cg::sync(cta);
  }
   
  if (idx < 16) {
    // Compute first eigen vector.
    float3 axis = firstEigenVector(covariance);

    dps[idx] = colors[idx].x * axis.x + 
               colors[idx].y * axis.y +
               colors[idx].z * axis.z;
  }
  cg::sync(cta);

  if (idx < 16) {
    int rank = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
      rank += (dps[i] < dps[idx]);
    }
  
    xrefs[idx] = rank;
  }
  cg::sync(cta);
  
  // Resolve elements with the same index.
  for (int i = 0; i < 15; i++) {
    if (idx < 16 && idx > i && xrefs[idx] == xrefs[i]) {
      ++xrefs[idx];
    }
    cg::sync(cta);
  }
  
  if (idx < 16) {
    colors[xrefs[idx]] = colors[idx];
  }
  cg::sync(cta);

  ushort bestStart, bestEnd;
  uint bestPermutation;

  __shared__ float errors[NUM_THREADS];

  evalAllPermutations(colors, permutations, bestStart, bestEnd, bestPermutation,
                      errors, sums, cta);

  // Use a parallel reduction to find minimum error.
  const int minIdx = findMinError(errors, cta);

  cg::sync(cta);

  // Only write the result of the winner thread.
  if (idx == minIdx) {
    saveBlockDXT1(bestStart, bestEnd, bestPermutation, xrefs, result,
                  blockOffset);
  }
}

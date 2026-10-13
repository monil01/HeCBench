#include <stdio.h>
#include "rescue_prime.hpp"

extern "C" __global__ void iris_MerklizeRescuePrimeApproach1Phase0 (
  const size_t output_offset,
  const ulong* __restrict__ leaves,
        ulong* __restrict__ intermediates,
  const ulong4* __restrict__ mds,
  const ulong4* __restrict__ ark1,
  const ulong4* __restrict__ ark2)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  merge(leaves + idx * (DIGEST_SIZE >> 1),
        intermediates + (output_offset + idx) * DIGEST_SIZE,
        mds,
        ark1,
        ark2);
}

extern "C" __global__ void iris_MerklizeRescuePrimeApproach1Phase1(
  const size_t offset,
        ulong* __restrict__ intermediates,
  const ulong4* __restrict__ mds,
  const ulong4* __restrict__ ark1,
  const ulong4* __restrict__ ark2)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  merge(intermediates + (offset << 1) * DIGEST_SIZE +
          idx * (DIGEST_SIZE >> 1),
        intermediates + (offset + idx) * DIGEST_SIZE,
        mds,
        ark1,
        ark2);
}

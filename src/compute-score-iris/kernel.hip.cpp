#include <hip/hip_runtime.h>
#define MANUAL_VECTOR 8
#define NUM_THREADS_PER_WG 64
#define BLOOM_1 5
#define BLOOM_2 0x7FFFF
#define docEndingTag 0xFFFFFFFF
__device__
ulong mulfp( ulong weight, uint freq )
{
  uint part1 = weight & 0xFFFFF;         // lower 24-bits of weight
  uint part2 = (weight >> 24) & 0xFFFF;  // next 16-bits

  uint res1 = part1 * freq;
  uint res2 = part2 * freq;

  return (ulong)res1 + (((ulong)res2) << 24);
}

extern "C" __global__ void
compute ( const uint* __restrict__ docWordFrequencies_dimm1, 
          const uint* __restrict__ docWordFrequencies_dimm2,
          const ulong*__restrict__ profileWeights_dimm1,
          const ulong*__restrict__ profileWeights_dimm2,
          const uint* __restrict__ isWordInProfileHash,
                uint* __restrict__ profileScorePerGroup_highbits_dimm1,
                uint* __restrict__ profileScorePerGroup_lowbits_dimm2 )
{
  uint curr_entry[MANUAL_VECTOR];
  uint word_id[MANUAL_VECTOR];
  uint freq[MANUAL_VECTOR];
  uint hash1[MANUAL_VECTOR];
  uint hash2[MANUAL_VECTOR];
  bool is_end[MANUAL_VECTOR];
  bool make_access[MANUAL_VECTOR];

  __shared__ ulong partial[NUM_THREADS_PER_WG/MANUAL_VECTOR];

  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  ulong sum = 0;
  //#pragma unroll
  for (uint i=0; i<MANUAL_VECTOR; i++) {
    curr_entry[i] = docWordFrequencies_dimm1[gid*MANUAL_VECTOR + i]; 
    freq[i] = curr_entry[i] & 0xff;
    word_id[i] = curr_entry[i] >> 8;
    is_end[i] = curr_entry[i] == docEndingTag;
    hash1[i] = word_id[i] >> BLOOM_1;
    hash2[i] = word_id[i] & BLOOM_2;
    make_access[i] = !is_end[i] && ((isWordInProfileHash[ hash1[i] >> 5 ] >> (hash1[i] & 0x1f)) & 0x1) 
      && ((isWordInProfileHash[ hash2[i] >> 5 ] >> (hash2[i] & 0x1f)) & 0x1); 
    if (make_access[i]) {
      sum += mulfp(profileWeights_dimm1[word_id[i]],freq[i]);
    }
  }

  //#pragma unroll
  for (uint i=0; i<MANUAL_VECTOR; i++) {
    curr_entry[i] = docWordFrequencies_dimm2[gid*MANUAL_VECTOR + i]; 
    freq[i] = curr_entry[i] & 0xff;
    word_id[i] = curr_entry[i] >> 8;
    is_end[i] = curr_entry[i] == docEndingTag;
    hash1[i] = word_id[i] >> BLOOM_1;
    hash2[i] = word_id[i] & BLOOM_2;
    make_access[i] = !is_end[i] && ((isWordInProfileHash[ hash1[i] >> 5 ] >> (hash1[i] & 0x1f)) & 0x1) 
      && ((isWordInProfileHash[ hash2[i] >> 5 ] >> (hash2[i] & 0x1f)) & 0x1); 
    if (make_access[i]) {
      sum += mulfp(profileWeights_dimm2[word_id[i]],freq[i]);
    }
  }

  partial[threadIdx.x] = sum;
  __syncthreads();

  if (threadIdx.x == 0) {
    ulong4 res= *(ulong4*)&partial[0];
    ulong4 res2= *(ulong4*)&partial[4];
    ulong final_result = res.x + res.y + res.z + res.w + res2.x + res2.y + res2.z + res2.w;
    profileScorePerGroup_highbits_dimm1[blockIdx.x] = (uint) (final_result >> 32); 
    profileScorePerGroup_lowbits_dimm2[blockIdx.x] = (uint) (final_result & 0xFFFFFFFF); 
  }
}

extern "C" __global__ void
reduction( const ulong* __restrict__ docInfo,
           const uint*  __restrict__ partial_highbits_dimm1,
           const uint*  __restrict__ partial_lowbits_dimm2,
                 ulong* __restrict__ result)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  ulong info = docInfo[gid];
  unsigned start = info >> 32;
  unsigned end = info & 0xFFFFFFFF;

  ulong total = 0;
  #pragma unroll 2
  for (unsigned i=start; i<=end; i++) {
    ulong upper = partial_highbits_dimm1[i];
    ulong lower = partial_lowbits_dimm2[i];
    ulong sum = (upper << 32) | lower;
    total += sum;
  }
  result[gid] = total;
}

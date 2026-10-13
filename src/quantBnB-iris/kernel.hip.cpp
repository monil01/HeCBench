#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include "reference.h"
#include "block_load.h"
#include "block_store.h"
#define NUM 4

static constexpr int TH=256, BLOCK_SIZE=256;
extern "C" __global__ void kQuantize(const float *__restrict__ code,
                          const float * __restrict__ A,
                          uint8_t *out, const int n)
{
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int n_full = gridDim.x * BLOCK_SIZE;
  const int base_idx = bid * BLOCK_SIZE;

  float vals[NUM];
  uint8_t qvals[NUM];

  // 1D block of TH threads owning NUM items each
  typedef BlockLoad<float, TH, NUM> LoadFloat;
  typedef BlockStore<uint8_t, TH, NUM> StoreChar;

  __shared__ typename LoadFloat::TempStorage loadf_storage;
  __shared__ typename StoreChar::TempStorage storec_storage;
  __shared__ float smem_code[256];

  for (int i = tid; i < 256; i += blockDim.x)
  {
    smem_code[i] = code[i];
  }

  for (int i = base_idx; i < n; i += n_full)
  {
      int valid_items = min(n - i, BLOCK_SIZE);

      LoadFloat(loadf_storage).Load(&(A[i]), vals, valid_items);

      __syncthreads();

      #pragma unroll
      for(int j = 0; j < NUM; j++)
          qvals[j] = dQuantize<0>(smem_code, 0.0f, vals[j]);

      StoreChar(storec_storage).Store(&(out[i]), qvals, valid_items);
  }
}

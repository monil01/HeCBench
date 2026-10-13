#include <hip/hip_runtime.h>
#include "kernel_types.h"
__device__ uint32_t hash(uint32_t k)
{
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;
  return k & (kHashTableCapacity-1);
}
extern "C" __global__ void
k_hashtable_insert(KeyValue*__restrict__ hashtable,
                   const KeyValue*__restrict__ kvs,
                   unsigned int numkvs)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numkvs)
  {
    uint32_t key = kvs[tid].key;
    uint32_t value = kvs[tid].value;
    uint32_t slot = hash(key);

    while (true)
    {
      uint32_t prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
      if (prev == kEmpty || prev == key)
      {
        //hashtable[slot].value = value;
        atomicExch(&hashtable[slot].value, value);
        return;
      }

      slot = (slot + 1) & (kHashTableCapacity-1);
    }
  }
}
extern "C" __global__ void 
k_hashtable_delete(KeyValue*__restrict__ hashtable, 
                   const KeyValue*__restrict__ kvs,
                   unsigned int numkvs)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numkvs)
  {
    uint32_t key = kvs[tid].key;
    uint32_t slot = hash(key);

    while (true)
    {
      if (hashtable[slot].key == key)
      {
        hashtable[slot].value = kEmpty;
        return;
      }
      if (hashtable[slot].key == kEmpty)
      {
        return;
      }
      slot = (slot + 1) & (kHashTableCapacity - 1);
    }
  }
}
extern "C" __global__ void
k_iterate_hashtable(KeyValue*__restrict__ pHashTable,
                    KeyValue*__restrict__ kvs,
                    uint32_t* kvs_size)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < kHashTableCapacity) 
  {
    if (pHashTable[tid].key != kEmpty) 
    {
      uint32_t value = pHashTable[tid].value;
      if (value != kEmpty)
      {
        uint32_t size = atomicAdd(kvs_size, 1);
        kvs[size] = pHashTable[tid];
      }
    }
  }
}

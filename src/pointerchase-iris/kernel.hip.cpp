#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdint>

extern "C" __global__ void init_pointer_chase(uintptr_t* data,
                                               const uint64_t n_ptrs,
                                               const uint32_t stride)
{
  const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_ptrs) return;

  const uint64_t next = (i + stride) % n_ptrs;
  data[i] = reinterpret_cast<uintptr_t>(&data[next]);
}

extern "C" __global__ void ptrChasingKernel(uintptr_t* data,
                                             const uint64_t size,
                                             const uint32_t accesses,
                                             const uint32_t targetBlock,
                                             uint32_t* status)
{
  if (blockIdx.x != targetBlock) return;

  uintptr_t* p = data;
  for (uint32_t i = 0; i < accesses; ++i) {
    p = reinterpret_cast<uintptr_t*>(*p);
  }

  if (p == nullptr) {
    assert(0);
  }

  const uintptr_t base = reinterpret_cast<uintptr_t>(data);
  const uintptr_t end = base + size;
  const uintptr_t value = reinterpret_cast<uintptr_t>(p);
  status[0] = (value >= base && value < end) ? 1u : 0u;
}

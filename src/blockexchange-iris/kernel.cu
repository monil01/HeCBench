#include <cuda_runtime.h>

extern "C" __global__ void blockexchange_kernel(const int* __restrict__ input,
                                                 int* __restrict__ output,
                                                 const int n)
{
  constexpr int block_size = 256;
  constexpr int items_per_thread = 4;
  constexpr int items_per_block = block_size * items_per_thread;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int dim = gridDim.x;
  const int n_full = (items_per_block * (n / items_per_block)) +
                     (n % items_per_block == 0 ? 0 : items_per_block);

  for (int base = bid * items_per_block; base < n_full; base += dim * items_per_block) {
    for (int item = 0; item < items_per_thread; item++) {
      const int src = base + item * block_size + tid;
      const int dst = base + tid * items_per_thread + item;
      if (src < n && dst < n) output[dst] = input[src];
    }
  }
}

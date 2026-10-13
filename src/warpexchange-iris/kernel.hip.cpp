#include <hip/hip_runtime.h>
#include "utils.h"
#include "block_load.h"
#include "block_store.h"
#include "warp_exchange.h"

constexpr int items_per_thread = 4;
constexpr int warp_threads = 32;

extern "C" __global__ void warp_exchange(const int *d, int *o, const int n)
{
  constexpr int BLOCK_SIZE=256, NUM_BLOCK=1024;
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int dim = gridDim.x;
  constexpr int block_threads = BLOCK_SIZE;
  constexpr int warps_per_block = block_threads / warp_threads;
  const int warp_id = tid / warp_threads;

  typedef WarpExchange<int, items_per_thread, warp_threads> WarpExchangeT;
  typedef BlockLoad<int, block_threads, items_per_thread> LoadInteger;
  typedef BlockStore<int, block_threads, items_per_thread> StoreInteger;

  // Allocate shared memory
  __shared__ typename WarpExchangeT::TempStorage temp_storage[warps_per_block];
  __shared__ typename LoadInteger::TempStorage loadi;
  __shared__ typename StoreInteger::TempStorage storei;

  // Obtain a segment of consecutive items that are blocked across threads
  int thread_data[items_per_thread];

  const int n_full = (NUM_BLOCK*(n/NUM_BLOCK)) + (n % NUM_BLOCK == 0 ? 0 : NUM_BLOCK);
  const int base_idx = (bid * NUM_BLOCK);
  for (unsigned int i = base_idx; i < n_full; i += dim*NUM_BLOCK) {
    unsigned int valid_items = n - i > NUM_BLOCK ? NUM_BLOCK : n - i;
    LoadInteger(loadi).Load(&(d[i]), thread_data, valid_items);

    // Collectively exchange data into a striped arrangement across threads
    WarpExchangeT(temp_storage[warp_id]).BlockedToStriped(thread_data, thread_data);

    StoreInteger(storei).Store(&(o[i]), thread_data, valid_items);
  }
}

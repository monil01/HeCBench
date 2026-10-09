#define ITEMS_PER_THREAD 4
#define MAX_BLOCK_THREADS 1024

extern "C" __global__ void adjacent_diff(const int* d_in, int* d_out, int subtract_left)
{
  __shared__ int tile[MAX_BLOCK_THREADS * ITEMS_PER_THREAD];
  if (blockDim.x > MAX_BLOCK_THREADS) return;

  const int items_per_block = blockDim.x * ITEMS_PER_THREAD;
  const int thread_base = threadIdx.x * ITEMS_PER_THREAD;
  const int global_base = (blockIdx.x * blockDim.x + threadIdx.x) * ITEMS_PER_THREAD;

  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    tile[thread_base + i] = d_in[global_base + i];
  }
  __syncthreads();

  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    const int local = thread_base + i;
    int value;
    if (subtract_left) {
      value = local == 0 ? tile[local] : tile[local] - tile[local - 1];
    } else {
      value = local + 1 >= items_per_block ? tile[local] : tile[local] - tile[local + 1];
    }
    d_out[global_base + i] = value;
  }
}

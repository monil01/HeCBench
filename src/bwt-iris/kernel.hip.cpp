#include <hip/hip_runtime.h>

extern "C" __global__ void generate_table(int* table, int table_size, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < table_size; i += stride) table[i] = (i < n) ? i : -1;
}

__device__ bool compare_rotations(const int& a, const int& b,
                                  const char* genome, int n) {
  if (a < 0) return false;
  if (b < 0) return true;
  for (int i = 0; i < n; i++) {
    if (genome[(a + i) % n] != genome[(b + i) % n]) {
      return genome[(a + i) % n] < genome[(b + i) % n];
    }
  }
  return false;
}

extern "C" __global__ void bitonic_sort_step(int* __restrict__ table,
                                             int table_size, int j, int k,
                                             const char* __restrict__ genome,
                                             int n) {
  unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int ixj = i ^ j;
  if (i < static_cast<unsigned int>(table_size) && ixj > i) {
    bool f = (i & k) == 0;
    int t1 = table[i];
    int t2 = table[ixj];
    if (compare_rotations(f ? t2 : t1, f ? t1 : t2, genome, n)) {
      table[i] = t2;
      table[ixj] = t1;
    }
  }
}

extern "C" __global__ void reconstruct_sequence(
    const int* __restrict__ table, const char* __restrict__ sequence,
    char* __restrict__ transformed_sequence, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    transformed_sequence[i] = sequence[(n + table[i] - 1) % n];
  }
}

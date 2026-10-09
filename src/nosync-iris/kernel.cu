extern "C" __global__ void sequence_kernel(int* data, const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] = i;
  }
}

extern "C" __global__ void partial_sum_kernel(const int* data,
                                               int* reduce_partials,
                                               int* scan_partials,
                                               const int n)
{
  __shared__ int scratch[256];

  const int tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + tid;
  int value = 0;
  if (i < n) {
    value = data[i];
  }

  scratch[tid] = value;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      scratch[tid] += scratch[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    reduce_partials[blockIdx.x] = scratch[0];
    scan_partials[blockIdx.x] = scratch[0];
  }
}

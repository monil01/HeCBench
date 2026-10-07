#define GPU_NUM_THREADS 256

__device__ int block_reduce_sum(int value) {
  __shared__ int scratch[GPU_NUM_THREADS];
  const int tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      scratch[tid] += scratch[tid + offset];
    }
    __syncthreads();
  }
  return scratch[0];
}

extern "C" __global__ void accuracy_kernel(
    const int N,
    const int D,
    const int top_k,
    const float* __restrict__ Xdata,
    const int* __restrict__ labelData,
    int* accuracy)
{
  int count = 0;

  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    const int label = labelData[row];
    const float label_pred = Xdata[row * D + label];
    int ngt = 0;

    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      const float pred = Xdata[row * D + col];
      if (pred > label_pred || (pred == label_pred && col <= label)) {
        ++ngt;
      }
    }

    ngt = block_reduce_sum(ngt);
    if (threadIdx.x == 0 && ngt <= top_k) {
      ++count;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && count > 0) {
    atomicAdd(accuracy, count);
  }
}

extern "C" __global__ void accuracy_kernel2(
    const int N,
    const int D,
    const int top_k,
    const float* __restrict__ Xdata,
    const int* __restrict__ labelData,
    int* accuracy)
{
  __shared__ float s_label_pred;
  __shared__ int s_label;

  int count = 0;

  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    if (threadIdx.x == 0) {
      s_label = labelData[row];
      s_label_pred = Xdata[row * D + s_label];
    }
    __syncthreads();

    const int label = s_label;
    const float label_pred = s_label_pred;
    const float* row_ptr = Xdata + row * D;
    int ngt = 0;

    int col = threadIdx.x;
    for (; col + 3 * blockDim.x < D; col += 4 * blockDim.x) {
      const float p0 = row_ptr[col];
      const float p1 = row_ptr[col + blockDim.x];
      const float p2 = row_ptr[col + 2 * blockDim.x];
      const float p3 = row_ptr[col + 3 * blockDim.x];

      ngt += (p0 > label_pred || (p0 == label_pred && col <= label));
      ngt += (p1 > label_pred || (p1 == label_pred && col + blockDim.x <= label));
      ngt += (p2 > label_pred || (p2 == label_pred && col + 2 * blockDim.x <= label));
      ngt += (p3 > label_pred || (p3 == label_pred && col + 3 * blockDim.x <= label));
    }

    for (; col < D; col += blockDim.x) {
      const float pred = row_ptr[col];
      ngt += (pred > label_pred || (pred == label_pred && col <= label));
    }

    ngt = block_reduce_sum(ngt);
    if (threadIdx.x == 0 && ngt <= top_k) {
      ++count;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && count > 0) {
    atomicAdd(accuracy, count);
  }
}

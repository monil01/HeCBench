#define NUM_THREADS 256

__device__ void reduce_pair(int* s_sum, int* s_sumsq)
{
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (threadIdx.x < stride) {
      s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
      s_sumsq[threadIdx.x] += s_sumsq[threadIdx.x + stride];
    }
  }
  __syncthreads();
}

extern "C" __global__ void channel_sum_nhwc(const int* X,
                                             int* sum,
                                             int* sumsq,
                                             int N,
                                             int C,
                                             int HxW)
{
  __shared__ int s_sum[NUM_THREADS];
  __shared__ int s_sumsq[NUM_THREADS];

  const int c = blockIdx.x;
  const int inner_size = N * HxW;
  int m_val = 0;
  int v_val = 0;
  for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
    const int value = X[i * C + c];
    m_val += value;
    v_val += value * value;
  }

  s_sum[threadIdx.x] = m_val;
  s_sumsq[threadIdx.x] = v_val;
  reduce_pair(s_sum, s_sumsq);

  if (threadIdx.x == 0) {
    sum[c] = s_sum[0];
    sumsq[c] = s_sumsq[0];
  }
}

extern "C" __global__ void channel_sum_nchw(const int* X,
                                             int* sum,
                                             int* sumsq,
                                             int N,
                                             int C,
                                             int HxW)
{
  __shared__ int s_sum[NUM_THREADS];
  __shared__ int s_sumsq[NUM_THREADS];

  const int c = blockIdx.x;
  const int inner_size = N * HxW;
  int m_val = 0;
  int v_val = 0;
  for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
    const int n = i / HxW;
    const int hw = i - n * HxW;
    const int value = X[(n * C + c) * HxW + hw];
    m_val += value;
    v_val += value * value;
  }

  s_sum[threadIdx.x] = m_val;
  s_sumsq[threadIdx.x] = v_val;
  reduce_pair(s_sum, s_sumsq);

  if (threadIdx.x == 0) {
    sum[c] = s_sum[0];
    sumsq[c] = s_sumsq[0];
  }
}

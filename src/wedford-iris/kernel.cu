template<typename T, typename C>
__device__ __forceinline__
void welford_merge_element(C& count, T& mean, T& m2n, const C& num_new,
                           const T& mean_new, const T& m2n_new) {
  T factor = T(1.0) / max(1, (count + num_new));
  T delta0 = mean - mean_new;
  mean = (mean_new * num_new + mean * count) * factor;
  m2n += m2n_new + delta0 * delta0 * num_new * count * factor;
  count += num_new;
}

template<typename T>
__device__ __forceinline__
void warp_reduce_mean_m2n(T &mean, T &m2n, int &num)
{
  #pragma unroll
  for(int i = warpSize/2; i > 0; i >>= 1) {
    auto num_new = __shfl_down_sync(0xffffffff, num, i);
    auto mean_new = __shfl_down_sync(0xffffffff, mean, i);
    auto m2n_new = __shfl_down_sync(0xffffffff, m2n, i);
    welford_merge_element(num, mean, m2n, num_new, mean_new, m2n_new);
  }
}

template <typename T>
__device__ void welford_reduce_mean_m2n(
      T* __restrict__ x, int* __restrict__ count, T &mean, T &m2n, int &num,
      int block_size, int thread_id)
{
  int lane = thread_id % warpSize;
  int wid = thread_id / warpSize;
  if (block_size > warpSize) {
    warp_reduce_mean_m2n(mean, m2n, num);
    if (lane == 0) {
      x[wid*2] = mean;
      x[wid*2+1] = m2n;
      count[wid] = num;
    }
    __syncthreads();
    if (wid == 0) {
      mean = (thread_id < block_size / warpSize)? x[lane*2] : T(0);
      m2n = (thread_id < block_size / warpSize)? x[lane*2+1] : T(0);
      num = (thread_id < block_size / warpSize)? count[lane] : int(0);
    }
  }
  if (wid==0) warp_reduce_mean_m2n(mean, m2n, num);
}

extern "C" __global__ void welford_kernel_float(
      const float* __restrict__ input,
      float* __restrict__ out_mean,
      float* __restrict__ out_var_biased,
      const int bs,
      const int fs,
      const int ss)
{
  int block_size = blockDim.x * blockDim.y;
  int count = 0;
  float x_mean = 0.0f;
  float m_2_n = 0.0f;
  int thread_id = threadIdx.y*blockDim.x + threadIdx.x;
  for (int batch_id = threadIdx.y; batch_id < bs; batch_id += blockDim.y) {
    int input_base = blockIdx.x*ss + batch_id*ss*fs;
    for (int offset = threadIdx.x; offset < ss ; offset += blockDim.x) {
      count++;
      auto x_n = input[offset+input_base];
      auto d = x_n - x_mean;
      x_mean += d / count;
      m_2_n += d * (x_n - x_mean);
    }
  }
  static __shared__ int s_mem[160];
  float* s_mem_ac = reinterpret_cast<float*>(&s_mem[32]);
  welford_reduce_mean_m2n<float>(s_mem_ac, s_mem, x_mean, m_2_n, count, block_size, thread_id);
  if (thread_id == 0) {
    out_mean[blockIdx.x] = x_mean;
    out_var_biased[blockIdx.x] = m_2_n/count;
  }
}

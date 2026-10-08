#define NUM_THREADS 256

template <bool kNFirst>
__device__ void ChannelShuffleNCHWKernel(const int G,
                                         const int K,
                                         const int HxW,
                                         const float* X,
                                         float* Y)
{
  const int C = G * K;
  const int n = kNFirst ? blockIdx.x : blockIdx.y;
  const int s = kNFirst ? blockIdx.y : blockIdx.x;
  const int g = blockIdx.z % G;
  const int k = blockIdx.z / G;
  const int offset = s * NUM_THREADS + threadIdx.x;
  if (offset < HxW) {
    Y[(n * C + blockIdx.z) * HxW + offset] =
        X[(n * C + g * K + k) * HxW + offset];
  }
}

template <int kSharedSize>
__device__ void ChannelShuffleNHWCKernel(const int G,
                                         const int K,
                                         const float* X,
                                         float* Y)
{
  __shared__ float sdata[kSharedSize];
  const int C = G * K;
  const int offset = blockIdx.x * C;
  for (int i = threadIdx.x; i < C; i += blockDim.x) {
    sdata[i] = X[offset + i];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < C; i += blockDim.x) {
    const int g = i % G;
    const int k = i / G;
    Y[offset + i] = sdata[g * K + k];
  }
}

extern "C" __global__ void channel_shuffle_nchw(float* Y,
                                                 const float* X,
                                                 int G,
                                                 int K,
                                                 int HxW)
{
  ChannelShuffleNCHWKernel<false>(G, K, HxW, X, Y);
}

extern "C" __global__ void channel_shuffle_nhwc_32(float* Y,
                                                    const float* X,
                                                    int G,
                                                    int K)
{
  ChannelShuffleNHWCKernel<32>(G, K, X, Y);
}

extern "C" __global__ void channel_shuffle_nhwc_128(float* Y,
                                                     const float* X,
                                                     int G,
                                                     int K)
{
  ChannelShuffleNHWCKernel<128>(G, K, X, Y);
}

extern "C" __global__ void channel_shuffle_nhwc_512(float* Y,
                                                     const float* X,
                                                     int G,
                                                     int K)
{
  ChannelShuffleNHWCKernel<512>(G, K, X, Y);
}

#include <hip/hip_runtime.h>

#define WARP_SIZE 32

struct WelfordData {
  float mean;
  float m2;
  long long n;
  float nf;
};

__device__ inline WelfordData combine(WelfordData a, WelfordData b)
{
  if (a.nf == 0) return b;
  if (b.nf == 0) return a;
  float delta = b.mean - a.mean;
  float new_count = a.nf + b.nf;
  float nb_over_n = b.nf / new_count;
  return {a.mean + delta * nb_over_n,
          a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
          -1,
          new_count};
}

__device__ inline WelfordData reduce_one(WelfordData acc, float data)
{
  long long new_n = acc.n + 1;
  float new_nf = static_cast<float>(new_n);
  float delta = data - acc.mean;
  float new_mean = acc.mean + delta / new_nf;
  float new_delta = data - new_mean;
  return {new_mean, acc.m2 + delta * new_delta, new_n, new_nf};
}

__device__ inline WelfordData shfl_down(WelfordData acc, int offset)
{
  return {__shfl_down(acc.mean, offset, WARP_SIZE),
          __shfl_down(acc.m2, offset, WARP_SIZE),
          __shfl_down(acc.n, offset, WARP_SIZE),
          __shfl_down(acc.nf, offset, WARP_SIZE)};
}

__device__ inline WelfordData warp_reduce(WelfordData val)
{
  #pragma unroll
  for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
    val = combine(val, shfl_down(val, offset));
  }
  return val;
}

extern "C" __global__ void RowwiseMomentsKernel(long long N,
                                                 float eps,
                                                 const float* X,
                                                 float* mean,
                                                 float* rstd)
{
  const long long row = blockIdx.x;
  WelfordData val{0.0f, 0.0f, 0, 0.0f};

  for (long long j = threadIdx.x; j < N; j += blockDim.x) {
    val = reduce_one(val, X[row * N + j]);
  }

  __shared__ WelfordData shared[WARP_SIZE];
  const int tid = threadIdx.x;
  const int lid = tid % WARP_SIZE;
  const int wid = tid / WARP_SIZE;
  const int warps = blockDim.x / WARP_SIZE;

  val = warp_reduce(val);
  __syncthreads();
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  WelfordData identity{0.0f, 0.0f, 0, 0.0f};
  val = (tid < warps) ? shared[lid] : identity;
  if (wid == 0) {
    val = warp_reduce(val);
  }

  if (threadIdx.x == 0) {
    float variance = val.m2 / val.nf;
    rstd[row] = rsqrtf(variance + eps);
    mean[row] = val.mean;
  }
}

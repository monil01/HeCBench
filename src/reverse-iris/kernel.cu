extern "C" __global__ void reverse(int* d, const int len)
{
  __shared__ int s[256];
  const int t = threadIdx.x;
  if (t < len) {
    s[t] = d[t];
  }
  __syncthreads();
  if (t < len) {
    d[t] = s[len - t - 1];
  }
}

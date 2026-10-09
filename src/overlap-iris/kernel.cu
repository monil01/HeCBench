extern "C" __global__ void incKernel(
    int* g_out,
    const int* g_in,
    int N,
    int inner_reps)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    for (int i = 0; i < inner_reps; ++i) {
      g_out[idx] = (i == 0 ? g_in[idx] : g_out[idx]) + 1;
    }
  }
}

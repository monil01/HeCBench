extern "C" __global__ void lb_keogh(
    const float* __restrict__ subject,
    const float* __restrict__ avgs,
    const float* __restrict__ stds,
    float* __restrict__ lb_keogh,
    const float* __restrict__ lower_bound,
    const float* __restrict__ upper_bound,
    const int M,
    const int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N - M + 1) {
    float residues = 0.0f;
    float avg = avgs[idx];
    float std = stds[idx];

    for (int i = 0; i < M; ++i) {
      float value = (subject[idx + i] - avg) / std;
      float lower = value - lower_bound[i];
      float upper = value - upper_bound[i];
      residues += upper * upper * (upper > 0) + lower * lower * (lower < 0);
    }

    lb_keogh[idx] = residues;
  }
}

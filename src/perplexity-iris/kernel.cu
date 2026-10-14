#include <cfloat>
#include <cmath>

extern "C" __global__ void sigmas_kernel(
    const float* __restrict__ distances,
    float* __restrict__ P,
    const float perplexity,
    const float desired_entropy,
    const int epochs,
    const float tol,
    const int n)
{
  const int k = 4 * static_cast<int>(perplexity);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float beta_min = -INFINITY;
  float beta_max = INFINITY;
  float beta = 1;
  const int ik = i * k;

  for (int step = 0; step < epochs; step++) {
    float sum_Pi = FLT_EPSILON;

    for (int j = 0; j < k; j++) {
      P[ik + j] = __expf(-distances[ik + j] * beta);
      sum_Pi += P[ik + j];
    }

    float sum_disti_Pi = 0;
    const float div = __fdividef(1.0f, sum_Pi);
    for (int j = 0; j < k; j++) {
      P[ik + j] *= div;
      sum_disti_Pi += distances[ik + j] * P[ik + j];
    }

    const float entropy = __logf(sum_Pi) + beta * sum_disti_Pi;
    const float entropy_diff = entropy - desired_entropy;
    if (fabsf(entropy_diff) <= tol) {
      break;
    }

    if (entropy_diff > 0) {
      beta_min = beta;
      if (isinf(beta_max)) {
        beta *= 2.0f;
      } else {
        beta = (beta + beta_max) * 0.5f;
      }
    } else {
      beta_max = beta;
      if (isinf(beta_min)) {
        beta *= 0.5f;
      } else {
        beta = (beta + beta_min) * 0.5f;
      }
    }
  }
}

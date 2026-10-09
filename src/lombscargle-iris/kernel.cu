#include <math.h>

extern "C" __global__ void lombscargle_kernel(const int x_shape,
                                               const int freqs_shape,
                                               const float* __restrict__ x,
                                               const float* __restrict__ y,
                                               const float* __restrict__ freqs,
                                               float* __restrict__ pgram)
{
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const float y_dot = 2.0f / 1.5f;

  for (int tid = tx; tid < freqs_shape; tid += stride) {
    const float freq = freqs[tid];
    float xc = 0.0f;
    float xs = 0.0f;
    float cc = 0.0f;
    float ss = 0.0f;
    float cs = 0.0f;

    for (int j = 0; j < x_shape; j++) {
      float s;
      float c;
      sincosf(freq * x[j], &s, &c);
      xc += y[j] * c;
      xs += y[j] * s;
      cc += c * c;
      ss += s * s;
      cs += c * s;
    }

    float c_tau;
    float s_tau;
    const float tau = atan2f(2.0f * cs, cc - ss) / (2.0f * freq);
    sincosf(freq * tau, &s_tau, &c_tau);
    const float c_tau2 = c_tau * c_tau;
    const float s_tau2 = s_tau * s_tau;
    const float cs_tau = 2.0f * c_tau * s_tau;

    pgram[tid] = (0.5f * ((((c_tau * xc + s_tau * xs) * (c_tau * xc + s_tau * xs) /
                             (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) +
                            ((c_tau * xs - s_tau * xc) * (c_tau * xs - s_tau * xc) /
                             (c_tau2 * ss - cs_tau * cs + s_tau2 * cc))))) * y_dot;
  }
}

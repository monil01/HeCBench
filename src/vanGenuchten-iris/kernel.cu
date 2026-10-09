#include <math.h>

static constexpr double alpha = 0.02;
static constexpr double theta_S = 0.45;
static constexpr double theta_R = 0.1;
static constexpr double n = 1.8;

extern "C" __global__ void vanGenuchten(
    const double* __restrict__ Ksat,
    const double* __restrict__ psi,
    double* __restrict__ C,
    double* __restrict__ theta,
    double* __restrict__ K,
    const int size)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= size) return;

  const double lambda = n - 1.0;
  const double m = lambda / n;

  double _psi = psi[i] * 100.0;
  double _theta;
  if (_psi < 0.0) {
    _theta = (theta_S - theta_R) / pow(1.0 + pow((alpha * (-_psi)), n), m) + theta_R;
  } else {
    _theta = theta_S;
  }

  theta[i] = _theta;

  const double Se = (_theta - theta_R) / (theta_S - theta_R);
  const double t = 1.0 - pow(1.0 - pow(Se, 1.0 / m), m);
  K[i] = Ksat[i] * sqrt(Se) * t * t;

  if (_psi < 0.0) {
    C[i] = 100.0 * alpha * n * (1.0 / n - 1.0) * pow(alpha * fabs(_psi), n - 1.0) *
           (theta_R - theta_S) * pow(pow(alpha * fabs(_psi), n) + 1.0, 1.0 / n - 2.0);
  } else {
    C[i] = 0.0;
  }
}

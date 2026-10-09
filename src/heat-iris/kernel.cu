#include <math.h>

extern "C" __global__ void initial_value(const int n, const double dx, const double length, double* u)
{
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n * n) {
    const int i = idx % n;
    const int j = idx / n;
    const double y = dx * (j + 1);
    const double x = dx * (i + 1);
    u[i + j * n] = sin(acos(-1.0) * x / length) * sin(acos(-1.0) * y / length);
  }
}

extern "C" __global__ void zero(const int n, double* u)
{
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n * n) u[idx] = 0.0;
}

extern "C" __global__ void solve(const int n, const double alpha, const double dx, const double dt,
                                  const double r, const double r2, const double* __restrict__ u,
                                  double* __restrict__ u_tmp)
{
  (void)alpha;
  (void)dx;
  (void)dt;
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n * n) {
    const int i = idx % n;
    const int j = idx / n;
    u_tmp[i + j * n] = r2 * u[i + j * n] +
                       r * ((i < n - 1) ? u[i + 1 + j * n] : 0.0) +
                       r * ((i > 0) ? u[i - 1 + j * n] : 0.0) +
                       r * ((j < n - 1) ? u[i + (j + 1) * n] : 0.0) +
                       r * ((j > 0) ? u[i + (j - 1) * n] : 0.0);
  }
}

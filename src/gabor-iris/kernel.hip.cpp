#include <hip/hip_runtime.h>
#include <math.h>

extern "C" __global__ void gabor(
  double *gabor_spatial,
  const unsigned int height,
  const unsigned int width,
  const double center_y,
  const double center_x,
  const double ctheta,
  const double stheta,
  const double scale,
  const double sx_2,
  const double sy_2,
  const double fx)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    double centered_y = (double)y - center_y;
    double centered_x = (double)x - center_x;
    double u = ctheta * centered_x - stheta * centered_y;
    double v = ctheta * centered_y + stheta * centered_x;
    gabor_spatial[y*width + x] = scale * exp(-0.5*(u*u/sx_2 + v*v/sy_2)) * cos(2.0*M_PI*fx*u);
  }
}

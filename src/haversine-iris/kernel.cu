#include <math.h>

#define DEGREE_TO_RADIAN M_PI / 180.0
#define EARTH_RADIUS_KM 6371.0

struct __attribute__((aligned(16))) HaversinePoint {
  double x, y, z, w;
};

extern "C" __global__ void compute_haversine_distance(
    const HaversinePoint* __restrict__ p,
    double* __restrict__ distance,
    const int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    auto ay = p[i].x * DEGREE_TO_RADIAN;
    auto ax = p[i].y * DEGREE_TO_RADIAN;
    auto by = p[i].z * DEGREE_TO_RADIAN;
    auto bx = p[i].w * DEGREE_TO_RADIAN;

    auto x = (bx - ax) / 2.0;
    auto y = (by - ay) / 2.0;
    auto sinysqrd = sin(y) * sin(y);
    auto sinxsqrd = sin(x) * sin(x);
    auto scale = cos(ay) * cos(by);
    distance[i] = 2.0 * EARTH_RADIUS_KM * asin(sqrt(sinysqrd + sinxsqrd * scale));
  }
}

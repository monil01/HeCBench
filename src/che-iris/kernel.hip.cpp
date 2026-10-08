#include <hip/hip_runtime.h>

#include "che.h"

__device__ int che_index(int x, int y, int z)
{
  return (z * DATAYSIZE + y) * DATAXSIZE + x;
}

__device__ double Laplacian(const double* c, double dx, double dy, double dz,
                            int x, int y, int z)
{
  int xp = x + 1;
  int xn = x - 1;
  int yp = y + 1;
  int yn = y - 1;
  int zp = z + 1;
  int zn = z - 1;

  const int nx = DATAXSIZE - 1;
  const int ny = DATAYSIZE - 1;
  const int nz = DATAZSIZE - 1;

  if (xp > nx) xp = 0;
  if (yp > ny) yp = 0;
  if (zp > nz) zp = 0;
  if (xn < 0) xn = nx;
  if (yn < 0) yn = ny;
  if (zn < 0) zn = nz;

  const double center = c[che_index(x, y, z)];
  const double cxx = (c[che_index(xp, y, z)] + c[che_index(xn, y, z)] - 2.0 * center) / (dx * dx);
  const double cyy = (c[che_index(x, yp, z)] + c[che_index(x, yn, z)] - 2.0 * center) / (dy * dy);
  const double czz = (c[che_index(x, y, zp)] + c[che_index(x, y, zn)] - 2.0 * center) / (dz * dz);
  return cxx + cyy + czz;
}

__device__ double GradientX(const double* phi, double dx, int x, int y, int z)
{
  const int nx = DATAXSIZE - 1;
  int xp = x + 1;
  int xn = x - 1;
  if (xp > nx) xp = 0;
  if (xn < 0) xn = nx;
  return (phi[che_index(xp, y, z)] - phi[che_index(xn, y, z)]) / (2.0 * dx);
}

__device__ double GradientY(const double* phi, double dy, int x, int y, int z)
{
  const int ny = DATAYSIZE - 1;
  int yp = y + 1;
  int yn = y - 1;
  if (yp > ny) yp = 0;
  if (yn < 0) yn = ny;
  return (phi[che_index(x, yp, z)] - phi[che_index(x, yn, z)]) / (2.0 * dy);
}

__device__ double GradientZ(const double* phi, double dz, int x, int y, int z)
{
  const int nz = DATAZSIZE - 1;
  int zp = z + 1;
  int zn = z - 1;
  if (zp > nz) zp = 0;
  if (zn < 0) zn = nz;
  return (phi[che_index(x, y, zp)] - phi[che_index(x, y, zn)]) / (2.0 * dz);
}

extern "C" __global__ void chemicalPotential(const double* c,
                                              double* mu,
                                              double dx,
                                              double dy,
                                              double dz,
                                              double gamma,
                                              double e_AA,
                                              double e_BB,
                                              double e_AB)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx < DATAXSIZE && idy < DATAYSIZE && idz < DATAZSIZE) {
    const int index = che_index(idx, idy, idz);
    const double cv = c[index];
    mu[index] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB) +
                3.0 * cv + cv * cv * cv -
                gamma * Laplacian(c, dx, dy, dz, idx, idy, idz);
  }
}

__device__ double freeEnergy(double c, double e_AA, double e_BB, double e_AB)
{
  return (((9.0 / 4.0) * ((c * c + 2.0 * c + 1.0) * e_AA +
                          (c * c - 2.0 * c + 1.0) * e_BB +
                          2.0 * (1.0 - c * c) * e_AB)) +
          ((3.0 / 2.0) * c * c) + ((3.0 / 12.0) * c * c * c * c));
}

extern "C" __global__ void localFreeEnergyFunctional(const double* c,
                                                      double* f,
                                                      double dx,
                                                      double dy,
                                                      double dz,
                                                      double gamma,
                                                      double e_AA,
                                                      double e_BB,
                                                      double e_AB)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx < DATAXSIZE && idy < DATAYSIZE && idz < DATAZSIZE) {
    const int index = che_index(idx, idy, idz);
    const double cv = c[index];
    const double gx = GradientX(c, dx, idx, idy, idz);
    const double gy = GradientY(c, dy, idx, idy, idz);
    const double gz = GradientZ(c, dz, idx, idy, idz);
    f[index] = freeEnergy(cv, e_AA, e_BB, e_AB) + (gamma / 2.0) * (gx * gx + gy * gy + gz * gz);
  }
}

extern "C" __global__ void cahnHilliard(double* cnew,
                                         const double* cold,
                                         const double* mu,
                                         double D,
                                         double dt,
                                         double dx,
                                         double dy,
                                         double dz)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx < DATAXSIZE && idy < DATAYSIZE && idz < DATAZSIZE) {
    const int index = che_index(idx, idy, idz);
    cnew[index] = cold[index] + dt * D * Laplacian(mu, dx, dy, dz, idx, idy, idz);
  }
}

extern "C" __global__ void Swap(double* cnew, double* cold)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int idy = blockIdx.y * blockDim.y + threadIdx.y;
  const int idz = blockIdx.z * blockDim.z + threadIdx.z;
  if (idx < DATAXSIZE && idy < DATAYSIZE && idz < DATAZSIZE) {
    const int index = che_index(idx, idy, idz);
    const double tmp = cnew[index];
    cnew[index] = cold[index];
    cold[index] = tmp;
  }
}

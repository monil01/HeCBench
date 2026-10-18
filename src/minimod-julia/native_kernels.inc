#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include "constants.h"
#include "grid.h"

#define R 4
#define NDIM 8

__global__ void target_inner_3d_kernel(
    llint nx, llint ny, llint nz, int ldimx, int ldimy, int ldimz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    float hdx_2, float hdy_2, float hdz_2,
    float coef0,
    float coefx_1, float coefx_2, float coefx_3, float coefx_4,
    float coefy_1, float coefy_2, float coefy_3, float coefy_4,
    float coefz_1, float coefz_2, float coefz_3, float coefz_4,
    const float *__restrict__ u, float *__restrict__ v, const float *__restrict__ vp,
    const float *__restrict__ phi, const float *__restrict__ eta)
{
  __shared__ float s_u[NDIM+2*R][NDIM+2*R][NDIM+2*R];

  const llint i0 = x3 + blockIdx.z * blockDim.z;
  const llint j0 = y3 + blockIdx.y * blockDim.y;
  const llint k0 = z3 + blockIdx.x * blockDim.x;
  
  const int ti = threadIdx.z;
  const int tj = threadIdx.y;
  const int tk = threadIdx.x;

  const llint i = i0 + ti;
  const llint j = j0 + tj;
  const llint k = k0 + tk;

  s_u[ti][tj][tk] = 0.f;

  if (ti < 2*R && tj < 2*R && tk< 2*R)
    s_u[NDIM+ti][NDIM+tj][NDIM+tk] = 0.f;

  __syncthreads();

  const llint sui = ti + R;
  const llint suj = tj + R;
  const llint suk = tk + R;

  const int z_side = ti / R;
  s_u[ti+z_side*NDIM][suj][suk] = u[IDX3(i+(z_side*2-1)*R,j,k)];
  const int y_side = tj / R;
  s_u[sui][tj+y_side*NDIM][suk] = u[IDX3(i,j+(y_side*2-1)*R,k)];
  s_u[sui][suj][tk] = u[IDX3(i,j,k-R)];
  s_u[sui][suj][tk+NDIM] = u[IDX3(i,j,k+R)];

  __syncthreads();

  if (i > x4-1 || j > y4-1 || k > z4-1) { return; }

  float lap = coef0 * s_u[sui][suj][suk] + 
            coefx_1 * (s_u[sui+1][suj][suk] + s_u[sui-1][suj][suk]) +
            coefy_1 * (s_u[sui][suj+1][suk] + s_u[sui][suj-1][suk]) +
            coefz_1 * (s_u[sui][suj][suk+1] + s_u[sui][suj][suk-1]) +
            coefx_2 * (s_u[sui+2][suj][suk] + s_u[sui-2][suj][suk]) +
            coefy_2 * (s_u[sui][suj+2][suk] + s_u[sui][suj-2][suk]) +
            coefz_2 * (s_u[sui][suj][suk+2] + s_u[sui][suj][suk-2]) +
            coefx_3 * (s_u[sui+3][suj][suk] + s_u[sui-3][suj][suk]) +
            coefy_3 * (s_u[sui][suj+3][suk] + s_u[sui][suj-3][suk]) +
            coefz_3 * (s_u[sui][suj][suk+3] + s_u[sui][suj][suk-3]) +
            coefx_4 * (s_u[sui+4][suj][suk] + s_u[sui-4][suj][suk]) +
            coefy_4 * (s_u[sui][suj+4][suk] + s_u[sui][suj-4][suk]) +
            coefz_4 * (s_u[sui][suj][suk+4] + s_u[sui][suj][suk-4]);
  v[IDX3(i,j,k)] = 2.f * s_u[sui][suj][suk] + vp[IDX3(i,j,k)] * lap - v[IDX3(i,j,k)];
}

__global__ void target_pml_3d_kernel(
    llint nx, llint ny, llint nz, int ldimx, int ldimy, int ldimz,
    llint x3, llint x4, llint y3, llint y4, llint z3, llint z4,
    llint lx, llint ly, llint lz,
    float hdx_2, float hdy_2, float hdz_2,
    float coef0,
    float coefx_1, float coefx_2, float coefx_3, float coefx_4,
    float coefy_1, float coefy_2, float coefy_3, float coefy_4,
    float coefz_1, float coefz_2, float coefz_3, float coefz_4,
    const float *__restrict__ u, float *__restrict__ v, const float *__restrict__ vp,
    float *__restrict__ phi, const float *__restrict__ eta)
{
  __shared__ float s_u[NDIM+2*R][NDIM+2*R][NDIM+2*R];

  const llint i0 = x3 + blockIdx.z * blockDim.z;
  const llint j0 = y3 + blockIdx.y * blockDim.y;
  const llint k0 = z3 + blockIdx.x * blockDim.x;

  const int ti = threadIdx.z;
  const int tj = threadIdx.y;
  const int tk = threadIdx.x;

  const llint i = i0 + ti;
  const llint j = j0 + tj;
  const llint k = k0 + tk;

  s_u[ti][tj][tk] = 0.f;

  if (ti < 2*R && tj < 2*R && tk< 2*R)
    s_u[NDIM+ti][NDIM+tj][NDIM+tk] = 0.f;

  __syncthreads();

  const llint sui = ti + R;
  const llint suj = tj + R;
  const llint suk = tk + R;

  const int z_side = ti / R;
  s_u[ti+z_side*NDIM][suj][suk] = u[IDX3(i+(z_side*2-1)*R,j,k)];
  const int y_side = tj / R;
  s_u[sui][tj+y_side*NDIM][suk] = u[IDX3(i,j+(y_side*2-1)*R,k)];
  s_u[sui][suj][tk] = u[IDX3(i,j,k-R)];
  s_u[sui][suj][tk+NDIM] = u[IDX3(i,j,k+R)];

  __syncthreads();

  if (i > x4-1 || j > y4-1 || k > z4-1) { return; }

  float lap = coef0 * s_u[sui][suj][suk] + 
            coefx_1 * (s_u[sui+1][suj][suk] + s_u[sui-1][suj][suk]) +
            coefy_1 * (s_u[sui][suj+1][suk] + s_u[sui][suj-1][suk]) +
            coefz_1 * (s_u[sui][suj][suk+1] + s_u[sui][suj][suk-1]) +
            coefx_2 * (s_u[sui+2][suj][suk] + s_u[sui-2][suj][suk]) +
            coefy_2 * (s_u[sui][suj+2][suk] + s_u[sui][suj-2][suk]) +
            coefz_2 * (s_u[sui][suj][suk+2] + s_u[sui][suj][suk-2]) +
            coefx_3 * (s_u[sui+3][suj][suk] + s_u[sui-3][suj][suk]) +
            coefy_3 * (s_u[sui][suj+3][suk] + s_u[sui][suj-3][suk]) +
            coefz_3 * (s_u[sui][suj][suk+3] + s_u[sui][suj][suk-3]) +
            coefx_4 * (s_u[sui+4][suj][suk] + s_u[sui-4][suj][suk]) +
            coefy_4 * (s_u[sui][suj+4][suk] + s_u[sui][suj-4][suk]) +
            coefz_4 * (s_u[sui][suj][suk+4] + s_u[sui][suj][suk-4]);

  const float s_eta_c = eta[IDX3(i,j,k)];

  v[IDX3(i,j,k)] = ((2.f*s_eta_c + 2.f - s_eta_c*s_eta_c)*s_u[sui][suj][suk] + 
      	    (vp[IDX3(i,j,k)] * (lap + phi[IDX3(i,j,k)]) - v[IDX3(i,j,k)])) / 
          (2.f*s_eta_c+1.f);

  phi[IDX3(i,j,k)] = 
   (phi[IDX3(i,j,k)] - 
   ((eta[IDX3(i+1,j,k)]-eta[IDX3(i-1,j,k)]) * 
   (s_u[sui+1][suj][suk]-s_u[sui-1][suj][suk]) * hdx_2 + 
   (eta[IDX3(i,j+1,k)]-eta[IDX3(i,j-1,k)]) *
   (s_u[sui][suj+1][suk]-s_u[sui][suj-1][suk]) * hdy_2 +
   (eta[IDX3(i,j,k+1)]-eta[IDX3(i,j,k-1)]) *
   (s_u[sui][suj][suk+1]-s_u[sui][suj][suk-1]) * hdz_2)) / (1.f + s_eta_c);
}

__global__ void kernel_add_source_kernel(float *g_u, llint idx, float source) {
    g_u[idx] += source;
}


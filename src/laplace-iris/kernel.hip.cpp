#include <hip/hip_runtime.h>
#include "config.h"
/** Function to update temperature for red cells
 *
 * \param[in]     aP          array of self coefficients
 * \param[in]     aW          array of west neighbor coefficients
 * \param[in]     aE          array of east neighbor coefficients
 * \param[in]     aS          array of south neighbor coefficients
 * \param[in]     aN          array of north neighbor coefficients
 * \param[in]     b           right-hand side array
 * \param[in]     temp_black  temperatures of black cells, constant in this function
 * \param[inout]  temp_red    temperatures of red cells
 * \param[out]    bl_norm_L2  array with residual information for blocks
 */
extern "C" __global__
void red_kernel (const Real *__restrict__ aP,
                 const Real *__restrict__ aW,
                 const Real *__restrict__ aE,
                 const Real *__restrict__ aS,
                 const Real *__restrict__ aN,
                 const Real *__restrict__ b,
                 const Real *__restrict__ temp_black,
                       Real *__restrict__ temp_red,
                       Real *__restrict__ norm_L2)
{
  int row = 1 + blockIdx.x * blockDim.x + threadIdx.x;
  int col = 1 + blockIdx.y * blockDim.y + threadIdx.y;
  if (row > NUM / 2 || col > NUM) return;

  int ind_red = col * ((NUM >> 1) + 2) + row; // local (red) index
  int ind = 2 * row - (col & 1) - 1 + NUM * (col - 1); // global index

  Real temp_old = temp_red[ind_red];

  Real res = b[ind]
        + aW[ind] * temp_black[row + (col - 1) * ((NUM >> 1) + 2)]
        + aE[ind] * temp_black[row + (col + 1) * ((NUM >> 1) + 2)]
        + aS[ind] * temp_black[row - (col & 1) + col * ((NUM >> 1) + 2)]
        + aN[ind] * temp_black[row + ((col + 1) & 1) + col * ((NUM >> 1) + 2)];

  Real temp_new = temp_old * (ONE - omega) + omega * (res / aP[ind]);

  temp_red[ind_red] = temp_new;
  res = temp_new - temp_old;

  norm_L2[ind_red] = res * res;

} // end red_kernel

/** Function to update temperature for black cells
 *
 * \param[in]      aP          array of self coefficients
 * \param[in]      aW          array of west neighbor coefficients
 * \param[in]      aE          array of east neighbor coefficients
 * \param[in]      aS          array of south neighbor coefficients
 * \param[in]      aN          array of north neighbor coefficients
 * \param[in]      b           right-hand side array
 * \param[in]      temp_red    temperatures of red cells, constant in this function
 * \param[inout]   temp_black  temperatures of black cells
 * \param[out]     bl_norm_L2  array with residual information for blocks
 */
extern "C" __global__
void black_kernel (const Real *__restrict__ aP,
                   const Real *__restrict__ aW,
                   const Real *__restrict__ aE,
                   const Real *__restrict__ aS,
                   const Real *__restrict__ aN,
                   const Real *__restrict__ b,
                   const Real *__restrict__ temp_red,
                         Real *__restrict__ temp_black,
                         Real *__restrict__ norm_L2)
{
  int row = 1 + blockIdx.x * blockDim.x + threadIdx.x;
  int col = 1 + blockIdx.y * blockDim.y + threadIdx.y;
  if (row > NUM / 2 || col > NUM) return;

  int ind_black = col * ((NUM >> 1) + 2) + row; // local (black) index
  int ind = 2 * row - ((col + 1) & 1) - 1 + NUM * (col - 1); // global index

  Real temp_old = temp_black[ind_black];

  Real res = b[ind]
        + aW[ind] * temp_red[row + (col - 1) * ((NUM >> 1) + 2)]
        + aE[ind] * temp_red[row + (col + 1) * ((NUM >> 1) + 2)]
        + aS[ind] * temp_red[row - ((col + 1) & 1) + col * ((NUM >> 1) + 2)]
        + aN[ind] * temp_red[row + (col & 1) + col * ((NUM >> 1) + 2)];

  Real temp_new = temp_old * (ONE - omega) + omega * (res / aP[ind]);

  temp_black[ind_black] = temp_new;
  res = temp_new - temp_old;

  norm_L2[ind_black] = res * res;
} // end black_kernel

extern "C" __global__ void partial_sum(const Real* input,Real* partial,size_t n) {
  __shared__ Real sums[256];size_t i=blockIdx.x*256+threadIdx.x;
  sums[threadIdx.x]=i<n ? input[i] : ZERO;__syncthreads();
  for(int d=128;d;d/=2){if(threadIdx.x<d)sums[threadIdx.x]+=sums[threadIdx.x+d];__syncthreads();}
  if(!threadIdx.x)partial[blockIdx.x]=sums[0];
}
extern "C" __global__ void finish_sum(const Real* partial,Real* result,size_t n) {
  __shared__ Real sums[256];Real value=ZERO;
  for(size_t i=threadIdx.x;i<n;i+=256)value+=partial[i];
  sums[threadIdx.x]=value;__syncthreads();
  for(int d=128;d;d/=2){if(threadIdx.x<d)sums[threadIdx.x]+=sums[threadIdx.x+d];__syncthreads();}
  if(!threadIdx.x)*result=sums[0];
}

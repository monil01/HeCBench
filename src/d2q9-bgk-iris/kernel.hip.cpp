#include <hip/hip_runtime.h>
#define NSPEEDS 9
#define LOCALSIZEX 128
#define LOCALSIZEY 1
__device__ bool 
isGreater(const float x, const float y) 
{
  return x > y ? 1 : 0;
}

extern "C" __global__ void d2q9_bgk(
  const float* __restrict__ Speed0A,
  const float* __restrict__ Speed1A,
  const float* __restrict__ Speed2A,
  const float* __restrict__ Speed3A,
  const float* __restrict__ Speed4A,
  const float* __restrict__ Speed5A,
  const float* __restrict__ Speed6A,
  const float* __restrict__ Speed7A,
  const float* __restrict__ Speed8A,
  float* __restrict__ Tmp0A,
  float* __restrict__ Tmp1A,
  float* __restrict__ Tmp2A,
  float* __restrict__ Tmp3A,
  float* __restrict__ Tmp4A,
  float* __restrict__ Tmp5A,
  float* __restrict__ Tmp6A,
  float* __restrict__ Tmp7A,
  float* __restrict__ Tmp8A,
  const int* __restrict__ ObstaclesA,
  float* __restrict__ Partial_Sum,
  int* __restrict__ Partial_Sum2,
  const float densityaccel,
  const float omega,
  const int nx,
  const int ny,
  const int tt)
{
  //setup local memory
  __shared__ int local_sum2[LOCALSIZEX*LOCALSIZEY];
  __shared__ float local_sum[LOCALSIZEX*LOCALSIZEY];

  /* get column and row indices */
  const int ii = blockIdx.x * blockDim.x + threadIdx.x;
  const int jj = blockIdx.y * blockDim.y + threadIdx.y;

  const float c_sq_inv = 3.f;
  const float c_sq = 1.f/c_sq_inv; /* square of speed of sound */
  const float temp1 = 4.5f;
  const float w1 = 1.f/9.f;
  const float w0 = 4.f * w1;  /* weighting factor */
  const float w2 = 1.f/36.f; /* weighting factor */
  const float w11 = densityaccel * w1;
  const float w21 = densityaccel * w2;

  /* determine indices of axis-direction neighbours
   ** respecting periodic boundary conditions (wrap around) */
  const int y_n = (jj + 1) % ny;
  const int x_e = (ii + 1) % nx;
  const int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  const int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  /* propagate densities from neighbouring cells, following
   ** appropriate directions of travel and writing into
   ** scratch space grid */

  float tmp_s0 = Speed0A[ii + jj*nx];
  float tmp_s1 = (jj == ny-2 && (!ObstaclesA[x_w + jj*nx] && isGreater((Speed3A[x_w + jj*nx] - w11) , 0.f) && isGreater((Speed6A[x_w + jj*nx] - w21) , 0.f) && isGreater((Speed7A[x_w + jj*nx] - w21) , 0.f))) ? Speed1A[x_w + jj*nx]+w11 : Speed1A[x_w + jj*nx];
  float tmp_s2 = Speed2A[ii + y_s*nx];
  float tmp_s3 = (jj == ny-2 && (!ObstaclesA[x_e + jj*nx] && isGreater((Speed3A[x_e + jj*nx] - w11) , 0.f) && isGreater((Speed6A[x_e + jj*nx] - w21) , 0.f) && isGreater((Speed7A[x_e + jj*nx] - w21) , 0.f))) ? Speed3A[x_e + jj*nx]-w11 : Speed3A[x_e + jj*nx];
  float tmp_s4 = Speed4A[ii + y_n*nx];
  float tmp_s5 = (y_s == ny-2 && (!ObstaclesA[x_w + y_s*nx] && isGreater((Speed3A[x_w + y_s*nx] - w11) , 0.f) && isGreater((Speed6A[x_w + y_s*nx] - w21) , 0.f) && isGreater((Speed7A[x_w + y_s*nx] - w21) , 0.f))) ? Speed5A[x_w + y_s*nx]+w21 : Speed5A[x_w + y_s*nx];
  float tmp_s6 = (y_s == ny-2 && (!ObstaclesA[x_e + y_s*nx] && isGreater((Speed3A[x_e + y_s*nx] - w11) , 0.f) && isGreater((Speed6A[x_e + y_s*nx] - w21) , 0.f) && isGreater((Speed7A[x_e + y_s*nx] - w21) , 0.f))) ? Speed6A[x_e + y_s*nx]-w21 : Speed6A[x_e + y_s*nx];
  float tmp_s7 = (y_n == ny-2 && (!ObstaclesA[x_e + y_n*nx] && isGreater((Speed3A[x_e + y_n*nx] - w11) , 0.f) && isGreater((Speed6A[x_e + y_n*nx] - w21) , 0.f) && isGreater((Speed7A[x_e + y_n*nx] - w21) , 0.f))) ? Speed7A[x_e + y_n*nx]-w21 : Speed7A[x_e + y_n*nx];
  float tmp_s8 = (y_n == ny-2 && (!ObstaclesA[x_w + y_n*nx] && isGreater((Speed3A[x_w + y_n*nx] - w11) , 0.f) && isGreater((Speed6A[x_w + y_n*nx] - w21) , 0.f) && isGreater((Speed7A[x_w + y_n*nx] - w21) , 0.f))) ? Speed8A[x_w + y_n*nx]+w21 : Speed8A[x_w + y_n*nx];

  /* compute local density total */
  float local_density = tmp_s0 + tmp_s1 + tmp_s2 + tmp_s3 + tmp_s4  + tmp_s5  + tmp_s6  + tmp_s7  + tmp_s8;
  const float local_density_recip = 1.f/(local_density);
  /* compute x velocity component */
  float u_x = (tmp_s1
      + tmp_s5
      + tmp_s8
      - tmp_s3
      - tmp_s6
      - tmp_s7)
    * local_density_recip;
  /* compute y velocity component */
  float u_y = (tmp_s2
      + tmp_s5
      + tmp_s6
      - tmp_s4
      - tmp_s8
      - tmp_s7)
    * local_density_recip;

  /* velocity squared */
  const float temp2 = - (u_x * u_x + u_y * u_y)/(2.f * c_sq);

  /* equilibrium densities */
  float d_equ[NSPEEDS];
  /* zero velocity density: weight w0 */
  d_equ[0] = w0 * local_density
    * (1.f + temp2);
  /* axis speeds: weight w1 */
  d_equ[1] = w1 * local_density * (1.f + u_x * c_sq_inv
      + (u_x * u_x) * temp1
      + temp2);
  d_equ[2] = w1 * local_density * (1.f + u_y * c_sq_inv
      + (u_y * u_y) * temp1
      + temp2);
  d_equ[3] = w1 * local_density * (1.f - u_x * c_sq_inv
      + (u_x * u_x) * temp1
      + temp2);
  d_equ[4] = w1 * local_density * (1.f - u_y * c_sq_inv
      + (u_y * u_y) * temp1
      + temp2);
  /* diagonal speeds: weight w2 */
  d_equ[5] = w2 * local_density * (1.f + (u_x + u_y) * c_sq_inv
      + ((u_x + u_y) * (u_x + u_y)) * temp1
      + temp2);
  d_equ[6] = w2 * local_density * (1.f + (-u_x + u_y) * c_sq_inv
      + ((-u_x + u_y) * (-u_x + u_y)) * temp1
      + temp2);
  d_equ[7] = w2 * local_density * (1.f + (-u_x - u_y) * c_sq_inv
      + ((-u_x - u_y) * (-u_x - u_y)) * temp1
      + temp2);
  d_equ[8] = w2 * local_density * (1.f + (u_x - u_y) * c_sq_inv
      + ((u_x - u_y) * (u_x - u_y)) * temp1
      + temp2);

  float tmp;
  int expression = ObstaclesA[ii + jj*nx];
  tmp_s0 = expression ? tmp_s0 : (tmp_s0 + omega * (d_equ[0] - tmp_s0));
  tmp = tmp_s1;
  tmp_s1 = expression ? tmp_s3 : (tmp_s1 + omega * (d_equ[1] - tmp_s1));
  tmp_s3 = expression ? tmp : (tmp_s3 + omega * (d_equ[3] - tmp_s3));
  tmp = tmp_s2;
  tmp_s2 = expression ? tmp_s4 : (tmp_s2 + omega * (d_equ[2] - tmp_s2));
  tmp_s4 = expression ? tmp : (tmp_s4 + omega * (d_equ[4] - tmp_s4));
  tmp = tmp_s5;
  tmp_s5 = expression ? tmp_s7 : (tmp_s5 + omega * (d_equ[5] - tmp_s5));
  tmp_s7 = expression ? tmp : (tmp_s7 + omega * (d_equ[7] - tmp_s7));
  tmp = tmp_s6;
  tmp_s6 = expression ? tmp_s8 : (tmp_s6 + omega * (d_equ[6] - tmp_s6));
  tmp_s8 = expression ? tmp : (tmp_s8 + omega * (d_equ[8] - tmp_s8));

  /* local density total */
  local_density = 1.f/(tmp_s0 + tmp_s1 + tmp_s2 + tmp_s3 + tmp_s4 + tmp_s5 + tmp_s6 + tmp_s7 + tmp_s8);

  /* x-component of velocity */
  u_x = (tmp_s1
      + tmp_s5
      + tmp_s8
      - tmp_s3
      - tmp_s6
      - tmp_s7)
    * local_density;
  /* compute y velocity component */
  u_y = (tmp_s2
      + tmp_s5
      + tmp_s6
      - tmp_s4
      - tmp_s7
      - tmp_s8)
    * local_density;

  Tmp0A[ii + jj*nx] = tmp_s0;
  Tmp1A[ii + jj*nx] = tmp_s1;
  Tmp2A[ii + jj*nx] = tmp_s2;
  Tmp3A[ii + jj*nx] = tmp_s3;
  Tmp4A[ii + jj*nx] = tmp_s4;
  Tmp5A[ii + jj*nx] = tmp_s5;
  Tmp6A[ii + jj*nx] = tmp_s6;
  Tmp7A[ii + jj*nx] = tmp_s7;
  Tmp8A[ii + jj*nx] = tmp_s8;


  int local_idi = threadIdx.x;
  int local_idj = threadIdx.y;
  int local_sizei = blockDim.x;
  int local_sizej = blockDim.y;

  /* accumulate the norm of x- and y- velocity components */
  local_sum[local_idi + local_idj*local_sizei] = (ObstaclesA[ii + jj*nx]) ? 0 : hypotf(u_x,u_y);
  /* increase counter of inspected cells */
  local_sum2[local_idi + local_idj*local_sizei] = (ObstaclesA[ii + jj*nx]) ? 0 : 1 ;

  __syncthreads();

  int group_id = blockIdx.x;
  int group_id2 = blockIdx.y; 
  int group_size = gridDim.x;
  int group_size2 = gridDim.y;
  if(local_idi == 0 && local_idj == 0){
    float sum = 0.0f;
    int sum2 = 0;
    for(int i = 0; i<local_sizei*local_sizej; i++){
      sum += local_sum[i];
      sum2 += local_sum2[i];
    }
    Partial_Sum[group_id+group_id2*group_size+tt*group_size*group_size2] = sum;
    Partial_Sum2[group_id+group_id2*group_size+tt*group_size*group_size2] = sum2;
  }
}

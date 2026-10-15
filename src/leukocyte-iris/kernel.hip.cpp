#include <hip/hip_runtime.h>
#define NCIRCLES 7
#define NPOINTS 150
#define RADIUS 10
#define MIN_RAD (RADIUS - 2)
#define MAX_RAD (RADIUS * 2)
#define MaxR  (MAX_RAD + 2)
#define LOCAL_WORK_SIZE 256
#define FP_TYPE float
#define FP_CONST(num) num##f
#define PI_FP32 FP_CONST(3.14159)
#define ONE_OVER_PI (FP_CONST(1.0) / PI_FP32)
#define MU FP_CONST(0.5)
#define LAMBDA (FP_CONST(8.0) * MU + FP_CONST(1.0))
#define NEXT_LOWEST_POWER_OF_TWO 256
extern "C" __global__ void kernel_GICOV (
    const float*__restrict__ grad_x,
    const float*__restrict__ grad_y,
    const float*__restrict__ sin_angle,
    const float*__restrict__ cos_angle,
    const int*__restrict__ tX,
    const int*__restrict__ tY,
    float*__restrict__ gicov,
    const int local_work_size,
    const int num_work_groups,
    const int grad_m)
{
  int i, j, k, n, x, y;
  int gid = blockIdx.x*blockDim.x+threadIdx.x;

  if (gid >= local_work_size*num_work_groups) return;

  // Determine this thread's pixel
  i = gid/local_work_size + MAX_RAD + 2;
  j = gid%local_work_size + MAX_RAD + 2;

  // Initialize the maximal GICOV score to 0
  float max_GICOV = 0.f;

  // Iterate across each stencil
  for (k = 0; k < NCIRCLES; k++) {
    // Variables used to compute the mean and variance
    //  of the gradients along the current stencil
    float sum = 0.f, M2 = 0.f, mean = 0.f;

    // Iterate across each sample point in the current stencil
    for (n = 0; n < NPOINTS; n++) {
      // Determine the x- and y-coordinates of the current sample point
      y = j + tY[(k * NPOINTS) + n];
      x = i + tX[(k * NPOINTS) + n];

      // Compute the combined gradient value at the current sample point
      int addr = x * grad_m + y;
      float p = grad_x[addr] * cos_angle[n] +
        grad_y[addr] * sin_angle[n];

      // Update the running total
      sum += p;

      // Partially compute the variance
      float delta = p - mean;
      mean = mean + (delta / (float) (n + 1));
      M2 = M2 + (delta * (p - mean));
    }

    // Finish computing the mean
    mean = sum / ((float) NPOINTS);

    // Finish computing the variance
    float var = M2 / ((float) (NPOINTS - 1));

    // Keep track of the maximal GICOV value seen so far
    if (((mean * mean) / var) > max_GICOV) max_GICOV = (mean * mean) / var;
  }

  // Store the maximal GICOV value
  gicov[(i * grad_m) + j] = max_GICOV;
}

extern "C" __global__ void kernel_dilated (
    const float*__restrict__ c_strel,
    const float*__restrict__ img,
    float*__restrict__ dilated,
    const int strel_m,
    const int strel_n,
    const int max_gicov_m,
    const int max_gicov_n)
{
  // Find the center of the structuring element
  int el_center_i = strel_m / 2;
  int el_center_j = strel_n / 2;

  // Determine this thread's location in the matrix
  int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  int i = thread_id % max_gicov_m;
  int j = thread_id / max_gicov_m;

  if(j >= max_gicov_n) return;

  // Initialize the maximum GICOV score seen so far to zero
  float max = 0.0f;

  // Iterate across the structuring element in one dimension
  int el_i, el_j, x, y;

  for (el_i = 0; el_i < strel_m; el_i++) {
    y = i - el_center_i + el_i;
    // Make sure we have not gone off the edge of the matrix
    if ( (y >= 0) && (y < max_gicov_m) ) {
      // Iterate across the structuring element in the other dimension
      for (el_j = 0; el_j < strel_n; el_j++) {
        x = j - el_center_j + el_j;
        // Make sure we have not gone off the edge of the matrix
        //  and that the current structuring element value is not zero
        if ( (x >= 0) &&
            (x < max_gicov_n) &&
            (c_strel[(el_i * strel_n) + el_j] != 0) ) {
          // Determine if this is the maximal value seen so far
          int addr = (x * max_gicov_m) + y;
          float temp = img[addr];
          if (temp > max) max = temp;
        }
      }
    }
  }
  // Store the maximum value found
  dilated[(i * max_gicov_n) + j] = max;
}


__device__
FP_TYPE heaviside(FP_TYPE x) {
  return atan(x) * ONE_OVER_PI + FP_CONST(0.5);
}

extern "C" __global__ void kernel_IMGVF(
          float*__restrict__ IMGVF_array,
    const float*__restrict__ I_array,
    const int*__restrict__ I_offsets,
    const int*__restrict__ m_array,
    const int*__restrict__ n_array,
    const float vx_float,
    const float vy_float,
    const float e_float,
    const float cutoff_float,
    const int max_iterations )
{
  __shared__ float IMGVF [41*81];
  __shared__ float IMGVF_buffer [LOCAL_WORK_SIZE];
  __shared__ int cell_converged;

  // Figure out which cell this thread block is working on
  int cell_num = blockIdx.x;

  // Get pointers to current cell's input image and inital matrix
  int I_offset = I_offsets[cell_num];
  float* IMGVF_global = &(IMGVF_array[I_offset]);
  //auto I = &(I_array[I_offset]);

  // Get current cell's matrix dimensions
  int m = m_array[cell_num];
  int n = n_array[cell_num];

  // Compute the number of virtual thread blocks
  int IMGVF_Size = m * n;
  int tb_count = (m * n + LOCAL_WORK_SIZE - 1) / LOCAL_WORK_SIZE;

  // Load the initial IMGVF matrix into shared memory
  int thread_id = threadIdx.x;
  int thread_block, i, j;
  for (thread_block = 0; thread_block < tb_count; thread_block++) {
    int offset = thread_block * LOCAL_WORK_SIZE;
    i = (thread_id + offset) / n;
    j = (thread_id + offset) % n;
    if (i < m) IMGVF[(i * n) + j] = IMGVF_global[(i * n) + j];
  }
  __syncthreads();

  // Set the converged flag to false
  if (thread_id == 0) cell_converged = 0;
  __syncthreads();

  // Constants used to iterate through virtual thread blocks
  const float one_nth = 1.0f / (float) n;
  const int tid_mod = thread_id % n;
  const int tbsize_mod = LOCAL_WORK_SIZE % n;

  // Constant used in the computation of Heaviside values
  FP_TYPE one_over_e = FP_CONST(1.0) / e_float;

  // Iteratively compute the IMGVF matrix until the computation has
  //  converged or we have reached the tb_countimum number of iterations
  int iterations = 0;
  while ((! cell_converged) && (iterations < max_iterations)) {

    // The total change to this thread's matrix elements in the current iteration
    FP_TYPE total_diff = FP_CONST(0.0);

    int old_i = 0, old_j = 0;
    j = tid_mod - tbsize_mod;

    // Iterate over virtual thread blocks
    for (thread_block = 0; thread_block < tb_count; thread_block++) {
      // Store the index of this thread's previous matrix element
      //  (used in the buffering scheme below)
      old_i = i;
      old_j = j;

      // Determine the index of this thread's current matrix element
      int offset = thread_block * LOCAL_WORK_SIZE;
      i = (thread_id + offset) * one_nth;
      j += tbsize_mod;
      if (j >= n) j -= n;

      FP_TYPE new_val = FP_CONST(0.0);
      FP_TYPE old_val = FP_CONST(0.0);

      // Make sure the thread has not gone off the end of the matrix
      if (i < m) {
        // Compute neighboring matrix element indices
        int rowU = (i == 0) ? 0 : i - 1;
        int rowD = (i == m - 1) ? m - 1 : i + 1;
        int colL = (j == 0) ? 0 : j - 1;
        int colR = (j == n - 1) ? n - 1 : j + 1;

        // Compute the difference between the matrix element and its eight neighbors
        old_val    = IMGVF[(i * n) + j];
        FP_TYPE U  = IMGVF[(rowU * n) + j   ] - old_val;
        FP_TYPE D  = IMGVF[(rowD * n) + j   ] - old_val;
        FP_TYPE L  = IMGVF[(i    * n) + colL] - old_val;
        FP_TYPE R  = IMGVF[(i    * n) + colR] - old_val;
        FP_TYPE UR = IMGVF[(rowU * n) + colR] - old_val;
        FP_TYPE DR = IMGVF[(rowD * n) + colR] - old_val;
        FP_TYPE UL = IMGVF[(rowU * n) + colL] - old_val;
        FP_TYPE DL = IMGVF[(rowD * n) + colL] - old_val;

        // Compute the regularized heaviside value for these differences
        FP_TYPE UHe  = heaviside((U  *       -vy_float)  * one_over_e);
        FP_TYPE DHe  = heaviside((D  *        vy_float)  * one_over_e);
        FP_TYPE LHe  = heaviside((L  *  -vx_float     )  * one_over_e);
        FP_TYPE RHe  = heaviside((R  *   vx_float     )  * one_over_e);
        FP_TYPE URHe = heaviside((UR * ( vx_float - vy_float)) * one_over_e);
        FP_TYPE DRHe = heaviside((DR * ( vx_float + vy_float)) * one_over_e);
        FP_TYPE ULHe = heaviside((UL * (-vx_float - vy_float)) * one_over_e);
        FP_TYPE DLHe = heaviside((DL * (-vx_float + vy_float)) * one_over_e);

        // Update the IMGVF value in two steps:
        // 1) Compute IMGVF += (mu / lambda)(UHe .*U  + DHe .*D  + LHe .*L  + RHe .*R +
        //                                   URHe.*UR + DRHe.*DR + ULHe.*UL + DLHe.*DL);
        new_val = old_val + (MU / LAMBDA) * (UHe  * U  + DHe  * D  + LHe  * L  + RHe  * R +
            URHe * UR + DRHe * DR + ULHe * UL + DLHe * DL);
        // 2) Compute IMGVF -= (1 / lambda)(I .* (IMGVF - I))
        //FP_TYPE vI = I[(i * n) + j];
        FP_TYPE vI = I_array[I_offset+ (i * n) + j];
        new_val -= ((1.0 / LAMBDA) * vI * (new_val - vI));

      }
      // All warps must finish neighbor reads before any shared tile write.
      __syncthreads();
      // Save the previous virtual thread block's value (if it exists)
      if (thread_block > 0) {
        offset = (thread_block - 1) * LOCAL_WORK_SIZE;
        if (old_i < m) IMGVF[(old_i * n) + old_j] = IMGVF_buffer[thread_id];
      }
      if (thread_block < tb_count - 1) {
        // Write the new value to the IMGVF_buffer
        IMGVF_buffer[thread_id] = new_val;
      } else {
        // We've reached the final virtual thread block,
        //  so write directly to the matrix
        if (i < m) IMGVF[(i * n) + j] = new_val;
      }

      // Keep track of the total change of this thread's matrix elements
      total_diff += fabs(new_val - old_val);

      // We need to synchronize between virtual thread blocks to prevent
      //  threads from writing the values from the buffer to the actual
      //  IMGVF matrix too early
      __syncthreads();
    }

    // We need to compute the overall sum of the change at each matrix element
    //  by performing a tree reduction across the whole threadblock
    IMGVF_buffer[thread_id] = total_diff;
    __syncthreads();

    // Account for thread block sizes that are not a power of 2
    if (thread_id >= NEXT_LOWEST_POWER_OF_TWO) {
      IMGVF_buffer[thread_id - NEXT_LOWEST_POWER_OF_TWO] += IMGVF_buffer[thread_id];
    }
    __syncthreads();

    // Perform the tree reduction
    int th;
    for (th = NEXT_LOWEST_POWER_OF_TWO / 2; th > 0; th /= 2) {
      if (thread_id < th) {
        IMGVF_buffer[thread_id] += IMGVF_buffer[thread_id + th];
      }
      __syncthreads();
    }

    // Figure out if we have converged
    if(thread_id == 0) {
      FP_TYPE mean = IMGVF_buffer[thread_id] / (FP_TYPE) (m * n);
      if (mean < cutoff_float) {
        // We have converged, so set the appropriate flag
        cell_converged = 1;
      }
    }

    // We need to synchronize to ensure that all threads
    //  read the correct value of the convergence flag
    __syncthreads();

    // Keep track of the number of iterations we have performed
    iterations++;
  }

  // Save the final IMGVF matrix to global memory
  for (thread_block = 0; thread_block < tb_count; thread_block++) {
    int offset = thread_block * LOCAL_WORK_SIZE + thread_id;
    if (offset < IMGVF_Size)
      IMGVF_global[offset] = IMGVF[offset];
  }
}

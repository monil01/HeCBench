#include <hip/hip_runtime.h>
#include "state.h"
#define slack (state->slack)
#define min_in_rows (state->min_in_rows)
#define min_in_cols (state->min_in_cols)
#define zeros (state->zeros)
#define zeros_size_b (state->zeros_size_b)
#define row_of_star_at_column (state->row_of_star_at_column)
#define column_of_star_at_row (state->column_of_star_at_row)
#define cover_row (state->cover_row)
#define cover_column (state->cover_column)
#define column_of_prime_at_row (state->column_of_prime_at_row)
#define row_of_green_at_column (state->row_of_green_at_column)
#define max_in_mat_row (state->max_in_mat_row)
#define min_in_mat_col (state->min_in_mat_col)
#define d_min_in_mat_vect (state->d_min_in_mat_vect)
#define d_min_in_mat (state->d_min_in_mat)
#define zeros_size (control->zeros_size)
#define n_matches (control->n_matches)
#define goto_5 (control->goto_5)
#define repeat_kernel (control->repeat_kernel)
extern "C" __global__ void init(State *state, Control *control)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // initializations
  //for step 2
  if (i < nrows){
    cover_row[i] = 0;
    column_of_star_at_row[i] = -1;
  }
  if (i < ncols){
    cover_column[i] = 0;
    row_of_star_at_column[i] = -1;
  }
}

// STEP 1.
// a) Subtracting the row by the minimum in each row
const int n_rows_per_block = n / n_blocks_reduction;

__device__ void min_in_rows_warp_reduce(volatile cost_t* sdata, int tid) {
  if (n_threads_reduction >= 64 && n_rows_per_block < 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
  if (n_threads_reduction >= 32 && n_rows_per_block < 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
  if (n_threads_reduction >= 16 && n_rows_per_block < 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
  if (n_threads_reduction >= 8 && n_rows_per_block < 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
  if (n_threads_reduction >= 4 && n_rows_per_block < 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
  if (n_threads_reduction >= 2 && n_rows_per_block < 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

extern "C" __global__ void calc_min_in_rows(State *state, Control *control)
{
  __shared__ cost_t sdata[n_threads_reduction];    // One temporary result for each thread.

  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  // One gets the line and column from the blockID and threadID.
  unsigned int l = bid * n_rows_per_block + tid % n_rows_per_block;
  unsigned int c = tid / n_rows_per_block;
  unsigned int i = c * nrows + l;
  const unsigned int gridSize = n_threads_reduction * n_blocks_reduction;
  cost_t thread_min = MAX_DATA;

  while (i < n * n) {
    thread_min = min(thread_min, slack[i]);
    i += gridSize;  // go to the next piece of the matrix...
    // gridSize = 2^k * n, so that each thread always processes the same line or column
  }
  sdata[tid] = thread_min;

  __syncthreads();
  if (n_threads_reduction >= 1024 && n_rows_per_block < 1024) {if (tid < 512) { sdata[tid] = min(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
  if (n_threads_reduction >= 512 && n_rows_per_block < 512) { if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
  if (n_threads_reduction >= 256 && n_rows_per_block < 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
  if (n_threads_reduction >= 128 && n_rows_per_block < 128) { if (tid <  64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
  if (tid < 32) min_in_rows_warp_reduce(sdata, tid);
  if (tid < n_rows_per_block) min_in_rows[bid*n_rows_per_block + tid] = sdata[tid];
}

// a) Subtracting the column by the minimum in each column
const int n_cols_per_block = n / n_blocks_reduction;

__device__ void min_in_cols_warp_reduce(volatile cost_t* sdata, int tid) {
  if (n_threads_reduction >= 64 && n_cols_per_block < 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
  if (n_threads_reduction >= 32 && n_cols_per_block < 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
  if (n_threads_reduction >= 16 && n_cols_per_block < 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
  if (n_threads_reduction >= 8 && n_cols_per_block < 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
  if (n_threads_reduction >= 4 && n_cols_per_block < 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
  if (n_threads_reduction >= 2 && n_cols_per_block < 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

extern "C" __global__ void calc_min_in_cols(State *state, Control *control)
{
  __shared__ cost_t sdata[n_threads_reduction];    // One temporary result for each thread

  unsigned int tid = threadIdx.x;
  unsigned int bid = blockIdx.x;
  // One gets the line and column from the blockID and threadID.
  unsigned int c = bid * n_cols_per_block + tid % n_cols_per_block;
  unsigned int l = tid / n_cols_per_block;
  const unsigned int gridSize = n_threads_reduction * n_blocks_reduction;
  cost_t thread_min = MAX_DATA;

  while (l < n) {
    unsigned int i = c * nrows + l;
    thread_min = min(thread_min, slack[i]);
    l += gridSize / n;  // go to the next piece of the matrix...
    // gridSize = 2^k * n, so that each thread always processes the same line or column
  }
  sdata[tid] = thread_min;

  __syncthreads();
  if (n_threads_reduction >= 1024 && n_cols_per_block < 1024) {
    if (tid < 512) { sdata[tid] = min(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
  if (n_threads_reduction >= 512 && n_cols_per_block < 512) {
    if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
  if (n_threads_reduction >= 256 && n_cols_per_block < 256) {
    if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
  if (n_threads_reduction >= 128 && n_cols_per_block < 128) {
    if (tid <  64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
  if (tid < 32) min_in_cols_warp_reduce(sdata, tid);
  if (tid < n_cols_per_block) min_in_cols[bid*n_cols_per_block + tid] = sdata[tid];
}

extern "C" __global__ void step_1_row_sub(State *state, Control *control)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l = i & row_mask;
  slack[i] = slack[i] - min_in_rows[l];  // subtract the minimum in row from that row
}

extern "C" __global__ void step_1_col_sub(State *state, Control *control)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int c = i >> log2_n;
  slack[i] = slack[i] - min_in_cols[c]; // subtract the minimum in row from that row

  if (i == 0) zeros_size = 0;
  if (i < n_blocks_step_4) zeros_size_b[i] = 0;
}

// Compress matrix
extern "C" __global__ void compress_matrix(State *state, Control *control){
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (slack[i] == 0) {
    atomicAdd(&zeros_size, 1);
    int b = i >> log2_data_block_size;
    int i0 = i & ~(data_block_size - 1);    // == b << log2_data_block_size
    int j = atomicAdd(zeros_size_b + b, 1);
    zeros[i0 + j] = i;
  }
}

// STEP 2
// Find a zero of slack. If there are no starred zeros in its
// column or row star the zero. Repeat for each zero.

// The zeros are split through blocks of cost_t so we run step 2 with several thread blocks and rerun the kernel if repeat was set to true.
extern "C" __global__ void step_2(State *state, Control *control)
{
  int i = threadIdx.x;
  int b = blockIdx.x;
  __shared__ bool repeat;
  __shared__ bool s_repeat_kernel;

  if (i == 0) s_repeat_kernel = false;

  do {
    __syncthreads();
    if (i == 0) repeat = false;
    __syncthreads();

    for (int j = i; j < zeros_size_b[b]; j += blockDim.x)
    {
      int z = zeros[(b << log2_data_block_size) + j];
      int l = z & row_mask;
      int c = z >> log2_n;

      if (cover_row[l] == 0 && cover_column[c] == 0) {
        // thread trys to get the line
        if (!atomicExch((int *)&(cover_row[l]), 1)){
          // only one thread gets the line
          if (!atomicExch((int *)&(cover_column[c]), 1)){
            // only one thread gets the column
            row_of_star_at_column[c] = l;
            column_of_star_at_row[l] = c;
          }
          else {
            cover_row[l] = 0;
            repeat = true;
            s_repeat_kernel = true;
          }
        }
      }
    }
    __syncthreads();
  } while (repeat);

  if (s_repeat_kernel) repeat_kernel = true;
}

// STEP 3
// uncover all the rows and columns before going to step 3
extern "C" __global__ void step_3ini(State *state, Control *control)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  cover_row[i] = 0;
  cover_column[i] = 0;
  if (i == 0) n_matches = 0;
}

// Cover each column with a starred zero. If all the columns are
// covered then the matching is maximum
extern "C" __global__ void step_3(State *state, Control *control)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (row_of_star_at_column[i]>=0)
  {
    cover_column[i] = 1;
    atomicAdd((int*)&n_matches, 1);
  }
}

// STEP 4
// Find a noncovered zero and prime it. If there is no starred
// zero in the row containing this primed zero, go to Step 5.
// Otherwise, cover this row and uncover the column containing
// the starred zero. Continue in this manner until there are no
// uncovered zeros left. Save the smallest uncovered value and
// Go to Step 6.

extern "C" __global__ void step_4_init(State *state, Control *control)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  column_of_prime_at_row[i] = -1;
  row_of_green_at_column[i] = -1;
}

extern "C" __global__ void step_4(State *state, Control *control) {
  __shared__  bool s_found;
  __shared__  bool s_goto_5;
  __shared__  bool s_repeat_kernel;
  volatile int *v_cover_row = cover_row;
  volatile int *v_cover_column = cover_column;

  int i = threadIdx.x;
  int b = blockIdx.x;

  if (i == 0) {
    s_repeat_kernel = false;
    s_goto_5 = false;
  }

  do {
    __syncthreads();
    if (i == 0) s_found = false;
    __syncthreads();

    for (int j = i; j < zeros_size_b[b]; j += blockDim.x)
    {
      int z = zeros[(b << log2_data_block_size) + j];
      int l = z & row_mask;
      int c = z >> log2_n;
      int c1 = column_of_star_at_row[l];

      for (int n = 0; n < 10; n++) {

        if (!v_cover_column[c] && !v_cover_row[l]) {
          s_found = true; s_repeat_kernel = true;
          column_of_prime_at_row[l] = c;

          if (c1 >= 0) {
            v_cover_row[l] = 1;
            __threadfence();
            v_cover_column[c1] = 0;
          }
          else {
            s_goto_5 = true;
          }
        }
      } // for(int n

    } // for(int j
    __syncthreads();
  } while (s_found && !s_goto_5);

  if (i == 0 && s_repeat_kernel) repeat_kernel = true;
  if (i == 0 && s_goto_5) goto_5 = true;
}

/* STEP 5:
Construct a series of alternating primed and starred zeros as follows:
Let Z0 represent the uncovered primed zero found in Step 4.
Let Z1 denote the starred zero in the column of Z0(if any).
Let Z2 denote the primed zero in the row of Z1(there will always
be one). Continue until the series terminates at a primed zero
that has no starred zero in its column. Unstar each starred
zero of the series, star each primed zero of the series, erase
all primes and uncover every line in the matrix. Return to Step 3.*/

// Eliminates joining paths
extern "C" __global__ void step_5a(State *state, Control *control)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  int r_Z0, c_Z0;

  c_Z0 = column_of_prime_at_row[i];
  if (c_Z0 >= 0 && column_of_star_at_row[i] < 0) {
    row_of_green_at_column[c_Z0] = i;

    while ((r_Z0 = row_of_star_at_column[c_Z0]) >= 0) {
      c_Z0 = column_of_prime_at_row[r_Z0];
      row_of_green_at_column[c_Z0] = r_Z0;
    }
  }
}

// Applies the alternating paths
extern "C" __global__ void step_5b(State *state, Control *control)
{
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  int r_Z0, c_Z0, c_Z2;

  r_Z0 = row_of_green_at_column[j];

  if (r_Z0 >= 0 && row_of_star_at_column[j] < 0) {

    c_Z2 = column_of_star_at_row[r_Z0];

    column_of_star_at_row[r_Z0] = j;
    row_of_star_at_column[j] = r_Z0;

    while (c_Z2 >= 0) {
      r_Z0 = row_of_green_at_column[c_Z2];  // row of Z2
      c_Z0 = c_Z2;              // col of Z2
      c_Z2 = column_of_star_at_row[r_Z0];    // col of Z4

      // star Z2
      column_of_star_at_row[r_Z0] = c_Z0;
      row_of_star_at_column[c_Z0] = r_Z0;
    }
  }
}

// STEP 6
// Add the minimum uncovered value to every element of each covered
// row, and subtract it from every element of each uncovered column.
// Return to Step 4 without altering any stars, primes, or covered lines.

template <unsigned int blockSize>
__device__ void min_warp_reduce(volatile cost_t* sdata, int tid) {
  if (blockSize >= 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
  if (blockSize >= 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
  if (blockSize >= 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
  if (blockSize >= 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
  if (blockSize >= 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
  if (blockSize >= 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

template <unsigned int blockSize>  // blockSize is the size of a block of threads
__device__ void min_reduce1(volatile cost_t *g_idata, volatile cost_t *g_odata, unsigned int n, State *state)
{
  __shared__ cost_t sdata[n_threads_reduction];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize * 2) + tid;
  unsigned int gridSize = blockSize * 2 * gridDim.x;
  sdata[tid] = MAX_DATA;

  while (i < n) {
    int i1 = i;
    int i2 = i + blockSize;
    int l1 = i1 & row_mask;
    int c1 = i1 >> log2_n;
    cost_t g1;
    if (cover_row[l1] == 1 || cover_column[c1] == 1) g1 = MAX_DATA;
    else g1 = g_idata[i1];
    int l2 = i2 & row_mask;
    int c2 = i2 >> log2_n;
    cost_t g2;
    if (cover_row[l2] == 1 || cover_column[c2] == 1) g2 = MAX_DATA;
    else g2 = g_idata[i2];
    sdata[tid] = min(sdata[tid], min(g1, g2));
    i += gridSize;
  }

  __syncthreads();
  if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = min(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
  if (blockSize >= 512) { if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
  if (tid < 32) min_warp_reduce<blockSize>(sdata, tid);
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__device__ void min_reduce2(volatile cost_t *g_idata, volatile cost_t *g_odata, unsigned int n)
{
  __shared__ cost_t sdata[n_threads_reduction];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize * 2) + tid;

  sdata[tid] = min(g_idata[i], g_idata[i + blockSize]);

  __syncthreads();
  if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = min(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
  if (blockSize >= 512) { if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
  if (tid < 32) min_warp_reduce<blockSize>(sdata, tid);
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

extern "C" __global__ void step_6_add_sub(State *state, Control *control)
{
  // STEP 6:
  //  /*STEP 6: Add the minimum uncovered value to every element of each covered
  //  row, and subtract it from every element of each uncovered column.
  //  Return to Step 4 without altering any stars, primes, or covered lines. */
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int l = i & row_mask;
  int c = i >> log2_n;
  if (cover_row[l] == 1 && cover_column[c] == 1)
    slack[i] += d_min_in_mat;
  if (cover_row[l] == 0 && cover_column[c] == 0)
    slack[i] -= d_min_in_mat;

  if (i == 0) zeros_size = 0;
  if (i < n_blocks_step_4) zeros_size_b[i] = 0;
}

extern "C" __global__ void min_reduce_kernel1(State *state, Control *control) {
  min_reduce1<n_threads_reduction>(slack, d_min_in_mat_vect, nrows*ncols, state);
}

extern "C" __global__ void min_reduce_kernel2(State *state, Control *control) {
  min_reduce2<n_threads_reduction / 2>(d_min_in_mat_vect, &d_min_in_mat, n_blocks_reduction);
}

// -------------------------------------------------------------------------------------

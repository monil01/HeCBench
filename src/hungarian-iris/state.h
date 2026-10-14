#pragma once
#include <climits>
#define klog2(n) ((n<8)?2:((n<16)?3:((n<32)?4:((n<64)?5:((n<128)?6:((n<256)?7:((n<512)?8:\
                ((n<1024)?9:((n<2048)?10:((n<4096)?11:((n<8192)?12:((n<16384)?13:0))))))))))))



#define kmin(x,y) ((x<y)?x:y)
#define kmax(x,y) ((x>y)?x:y)

#ifndef USE_TEST_MATRIX
#ifdef _n_
// These values are meant to be changed by scripts
const int n = _n_;          // size of the cost/pay matrix
const int range = _range_;  // defines the range of the random matrix.
const int user_n = n;
const int n_tests = 100;
#else
// User inputs: These values should be changed by the user
const int user_n = 1000;    // This is the size of the cost matrix as supplied by the user
const int n = 1<<(klog2(user_n)+1);    // The size of the cost/pay matrix used in the algorithm that is increased to a power of two
const int range = n;        // defines the range of the random matrix.
const int n_tests = 10;     // defines the number of tests performed
#endif

// End of user inputs

const int log2_n = klog2(n);
const int n_threads = kmin(n,64);    // Number of threads used in small kernels grid size (typically grid size equal to n)
// Used in steps 3ini, 3, 4ini, 4a, 4b, 5a and 5b (64)
const int n_threads_reduction = kmin(n, 256); // Number of threads used in the redution kernels in step 1 and 6 (256)
const int n_blocks_reduction = kmin(n, 256);  // Number of blocks used in the redution kernels in step 1 and 6 (256)
const int n_threads_full = kmin(n, 256);      // Number of threads used the largest grids sizes (typically grid size equal to n*n)
// Used in steps 2 and 6 (512)
const int seed = 45345; // Initialization for the random number generator

#else
const int n = 4;
const int log2_n = 2;
const int n_threads = 2;
const int n_threads_reduction = 2;
const int n_blocks_reduction = 2;
const int n_threads_full = 2;
#endif

const int n_blocks = n / n_threads;  // Number of blocks used in small kernels grid size (typically grid size equal to n)
const int n_blocks_full = n * n / n_threads_full; // Number of blocks used the largest gris sizes (typically grid size equal to n*n)
const int row_mask = (1 << log2_n) - 1; // Used to extract the row from tha matrix position index (matrices are column wise)
const int nrows = n, ncols = n; // The matrix is square so the number of rows and columns is equal to n
const int max_threads_per_block = 256; // The maximum number of threads per block
const int columns_per_block_step_4 = 512; // Number of columns per block in step 4
const int n_blocks_step_4 = kmax(n / columns_per_block_step_4, 1);  // Number of blocks in step 4 and 2
const int data_block_size = columns_per_block_step_4 * n; // The size of a cost_t block. Note that this can be bigger than the matrix size.
const int log2_data_block_size = log2_n + klog2(columns_per_block_step_4);  // log2 of the size of a cost_t block. Note that klog2 cannot handle very large sizes

// For the selection of the cost_t type used
#ifndef CHAR_DATA_TYPE
typedef int cost_t;
#define MAX_DATA INT_MAX
#define MIN_DATA INT_MIN
#else
typedef unsigned char cost_t;
#define MAX_DATA 255
#define MIN_DATA 0
#endif


struct State {
  cost_t slack[nrows*ncols];           // The slack matrix
  cost_t min_in_rows[nrows];           // Minimum in rows
  cost_t min_in_cols[ncols];           // Minimum in columns
  int zeros[nrows*ncols];            // A vector with the position of the zeros in the slack matrix
  int zeros_size_b[n_blocks_step_4]; // The number of zeros in block i
  int row_of_star_at_column[ncols];  // A vector that given the column j gives the row of the star at that column (or -1, no star)
  int column_of_star_at_row[nrows];  // A vector that given the row i gives the column of the star at that row (or -1, no star)
  int cover_row[nrows];              // A vector that given the row i indicates if it is covered (1- covered, 0- uncovered)
  int cover_column[ncols];           // A vector that given the column j indicates if it is covered (1- covered, 0- uncovered)
  int column_of_prime_at_row[nrows]; // A vector that given the row i gives the column of the prime at that row  (or -1, no prime)
  int row_of_green_at_column[ncols]; // A vector that given the row j gives the column of the green at that row (or -1, no green)
  cost_t max_in_mat_row[nrows];        // Used in step 1 to stores the maximum in rows
  cost_t min_in_mat_col[ncols];        // Used in step 1 to stores the minimums in columns
  cost_t d_min_in_mat_vect[n_blocks_reduction];  // Used in step 6 to stores the intermediate results from the first reduction kernel
  cost_t d_min_in_mat;                 // Used in step 6 to store the minimum
};
struct Control {
  int zeros_size;            // The number fo zeros
  int n_matches;             // Used in step 3 to count the number of matches found
  bool goto_5;               // After step 4, goto step 5?
  bool repeat_kernel;        // Needs to repeat the step 2 and step 4 kernel?
};

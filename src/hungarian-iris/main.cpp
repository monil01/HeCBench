#include <iris/iris.hpp>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cassert>
#include <random>
#include <chrono>
#include <cstring>
#include "state.h"
// Host Variables

// Some host variables start with h_ to distinguish them from the corresponding device variables
// Device variables have no prefix.

#ifndef USE_TEST_MATRIX
cost_t h_cost[ncols][nrows];
#else
cost_t h_cost[n][n] = { { 1, 2, 3, 4 }, { 2, 4, 6, 8 }, { 3, 6, 9, 12 }, { 4, 8, 12, 16 } };
#endif
int h_column_of_star_at_row[nrows];
int h_zeros_vector_size;
int h_n_matches;
bool h_found;
bool h_goto_5;


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
// Used to make sure some constants are properly set
void check(bool val, const char *str){
  if (!val) {
    printf("Check failed: %s!\n", str);
    exit(-1);
  }
}

static State state_storage{};
static Control control_storage{};
static State *state = &state_storage;
static Control *control = &control_storage;
static iris::DMem *state_mem, *control_mem;

static void launch(const char *name, size_t blocks, size_t threads) {
  check(threads > 0 && threads <= max_threads_per_block, "Invalid launch");
  check(control_mem->update(control) == IRIS_SUCCESS, "control update");
  iris::Task task;
  void *params[] = {state_mem, control_mem};
  int info[] = {iris_rw, iris_rw};
  size_t gws[] = {blocks * threads}, lws[] = {threads};
  check(task.kernel(name, 1, nullptr, gws, lws, 2, params, info) == IRIS_SUCCESS, "kernel");
  check(task.flush_out(*control_mem) == IRIS_SUCCESS, "flush control");
  check(task.submit(iris_gpu, nullptr, 1) == IRIS_SUCCESS, "submit");
}
#define call_kernel(k, blocks, threads) launch(#k, blocks, threads)
#define call_kernel_s(k, blocks, threads, shared) launch(#k, blocks, threads)
// Hungarian_Algorithm
void Hungarian_Algorithm()
{
  // Initialization
  call_kernel(init, n_blocks, n_threads);

  // Step 1 kernels
  call_kernel(calc_min_in_rows, n_blocks_reduction, n_threads_reduction);
  call_kernel(step_1_row_sub, n_blocks_full, n_threads_full);
  call_kernel(calc_min_in_cols, n_blocks_reduction, n_threads_reduction);
  call_kernel(step_1_col_sub, n_blocks_full, n_threads_full);

  // compress_matrix
  call_kernel(compress_matrix, n_blocks_full, n_threads_full);

  // Step 2 kernels
  do {
    repeat_kernel = false;
    call_kernel(step_2, n_blocks_step_4, (n_blocks_step_4 > 1 || zeros_size > max_threads_per_block) ? max_threads_per_block : zeros_size);
    // If we have more than one block it means that we have 512 lines per block so 1024 threads should be adequate.
  } while (repeat_kernel);

  while (1) {  // repeat steps 3 to 6

    // Step 3 kernels
    call_kernel(step_3ini, n_blocks, n_threads);
    call_kernel(step_3, n_blocks, n_threads);

    if (n_matches >= ncols) break;      // It's done

    //step 4_kernels
    call_kernel(step_4_init, n_blocks, n_threads);

    while (1) // repeat step 4 and 6
    {
      do {  // step 4 loop
        goto_5 = false; repeat_kernel = false;

        call_kernel(step_4, n_blocks_step_4, (n_blocks_step_4 > 1 || zeros_size > max_threads_per_block) ? max_threads_per_block : zeros_size);
        // If we have more than one block it means that we have 512 lines per block so 1024 threads should be adequate.

      } while (repeat_kernel && !goto_5);

      if (goto_5) break;

      //step 6_kernel
      call_kernel_s(min_reduce_kernel1, n_blocks_reduction, n_threads_reduction, n_threads_reduction*sizeof(int));
      call_kernel_s(min_reduce_kernel2, 1, n_blocks_reduction / 2, (n_blocks_reduction / 2) * sizeof(int));
      call_kernel(step_6_add_sub, n_blocks_full, n_threads_full);

      //compress_matrix
      call_kernel(compress_matrix, n_blocks_full, n_threads_full);

    } // repeat step 4 and 6

    call_kernel(step_5a, n_blocks, n_threads);
    call_kernel(step_5b, n_blocks, n_threads);

  }  // repeat steps 3 to 6
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <output file>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  check(platform.init(&argc, &argv, 1) == IRIS_SUCCESS, "IRIS init");
  iris::DMem sm(state, sizeof(State)), cm(control, sizeof(Control));
  state_mem = &sm; control_mem = &cm;

  // Constant checks:
  check(n == (1 << log2_n), "Incorrect log2_n!");
  check(n_threads*n_blocks == n, "n_threads*n_blocks != n\n");
  // step 1
  check(n_blocks_reduction <= n, "Step 1: Should have several lines per block!");
  check(n % n_blocks_reduction == 0, "Step 1: Number of lines per block should be integer!");
  check((n_blocks_reduction*n_threads_reduction) % n == 0,
        "Step 1: The grid size must be a multiple of the line size!");
  check(n_threads_reduction*n_blocks_reduction <= n*n,
        "Step 1: The grid size is bigger than the matrix size!");
  // step 6
  check(n_threads_full*n_blocks_full <= n*n,
        "Step 6: The grid size is bigger than the matrix size!");
  check(columns_per_block_step_4*n == (1 << log2_data_block_size),
        "Columns per block of step 4 is not a power of two!");

  // Open text file
  FILE *file = freopen(argv[1], "w", stdout);
  if (file == NULL)
  {
    perror("Error opening the output file!\n");
    return 1;
  };

  // Prints the current time
  time_t current_time;
  time(&current_time);
  printf("%s\n", ctime(&current_time));
  fflush(file);

  // total kernel time for all testcases
  long total_time = 0;

#ifndef USE_TEST_MATRIX
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<int> distribution(0, range-1);

  for (int test = 0; test < n_tests; test++) {
    printf("\n\n\n\ntest %d\n", test);
    fflush(file);

    for (int c = 0; c < ncols; c++)
      for (int r = 0; r < nrows; r++) {
        if (c < user_n && r < user_n)
          h_cost[c][r] = distribution(generator);
        else {
          if (c == r) h_cost[c][r] = 0;
          else h_cost[c][r] = MAX_DATA;
        }
      }
#endif

    // Copy vectors from host memory to device memory
    memcpy(slack, h_cost, sizeof(cost_t)*nrows*ncols);
    check(sm.update(state) == IRIS_SUCCESS, "state update");

    // Invoke kernels

    auto start = std::chrono::steady_clock::now();

    Hungarian_Algorithm();


    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
    printf("Total kernel execution time of the Hungarian algorithm %f (s)\n", time * 1e-9f);

    fflush(file);

    // Copy assignments from Device to Host and calculate the total Cost
    iris::Task flush;
    check(flush.flush_out(sm) == IRIS_SUCCESS, "flush state");
    check(flush.submit(iris_gpu, nullptr, 1) == IRIS_SUCCESS, "submit flush");
    memcpy(h_column_of_star_at_row, column_of_star_at_row, nrows * sizeof(int));

    bool seen[ncols] = {};
    int total_cost = 0;
    for (int r = 0; r < nrows; r++) {
      int c = h_column_of_star_at_row[r];
      check(c >= 0 && c < ncols && !seen[c], "assignment is not a permutation");
      seen[c] = true;
      total_cost += h_cost[c][r];
    }

    printf("Total cost is \t %d \n", total_cost);

#ifndef USE_TEST_MATRIX
  }
#endif

  check(platform.finalize() == IRIS_SUCCESS, "IRIS finalize");
  fclose(file);
  fprintf(stderr, "Total kernel time for all test cases %lf (s)\n", total_time * 1e-9);
  return 0;
}

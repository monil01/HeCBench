#include <iris/iris.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define GPU_NUM_THREADS 256

template <typename T>
static void reference_reverse(std::vector<T>& out,
                              const std::vector<T>& in,
                              int nrows,
                              int ncols,
                              bool row_major,
                              bool along_rows)
{
  for (int row = 0; row < nrows; ++row) {
    for (int col = 0; col < ncols; ++col) {
      const int dst_row = along_rows ? (nrows - row - 1) : row;
      const int dst_col = along_rows ? col : (ncols - col - 1);
      const size_t src_idx = row_major
        ? static_cast<size_t>(row) * ncols + col
        : static_cast<size_t>(col) * nrows + row;
      const size_t dst_idx = row_major
        ? static_cast<size_t>(dst_row) * ncols + dst_col
        : static_cast<size_t>(dst_col) * nrows + dst_row;
      out[dst_idx] = in[src_idx];
    }
  }
}

template <typename T>
static unsigned long long checksum(const std::vector<T>& values)
{
  unsigned long long sum = 1469598103934665603ull;
  for (T value : values) {
    sum ^= static_cast<unsigned long long>(value);
    sum *= 1099511628211ull;
  }
  return sum;
}

template <typename T>
static int eval_case(const char* kernel_name,
                     const std::vector<T>& input,
                     bool row_major,
                     bool along_rows,
                     int nrows,
                     int ncols,
                     int repeat)
{
  std::printf("\nInput matrix is %s major and reverse along %s\n",
              row_major ? "row" : "column",
              along_rows ? "rows" : "columns");

  const size_t matrix_size = static_cast<size_t>(nrows) * ncols;
  const size_t bytes = sizeof(T) * matrix_size;
  std::vector<T> output(matrix_size);
  std::vector<T> expected(matrix_size);
  reference_reverse(expected, input, nrows, ncols, row_major, along_rows);

  iris::DMem mem_in(const_cast<T*>(input.data()), bytes);
  iris::DMem mem_out(output.data(), bytes);

  const int row_major_arg = row_major ? 1 : 0;
  const int along_rows_arg = along_rows ? 1 : 0;
  const int total = static_cast<int>(matrix_size);
  const size_t gws[1] = {
    ((matrix_size + GPU_NUM_THREADS - 1) / GPU_NUM_THREADS) * GPU_NUM_THREADS
  };
  const size_t lws[1] = { GPU_NUM_THREADS };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    iris::Task task;
    void* params[7] = {
      &mem_out,
      &mem_in,
      &nrows,
      &ncols,
      const_cast<int*>(&row_major_arg),
      const_cast<int*>(&along_rows_arg),
      const_cast<int*>(&total)
    };
    int pinfo[7] = {
      iris_w,
      iris_r,
      sizeof(nrows),
      sizeof(ncols),
      sizeof(row_major_arg),
      sizeof(along_rows_arg),
      sizeof(total)
    };
    task.kernel(kernel_name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 7, params, pinfo);
    task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (ms)\n", elapsed * 1e-6f / repeat);

#ifdef DEBUG
  for (size_t i = 1; i <= matrix_size; ++i) {
    std::printf("%d ", static_cast<int>(output[i - 1]));
    if (i % static_cast<size_t>(ncols) == 0) std::printf("\n");
  }
#endif

  size_t mismatches = 0;
  for (size_t i = 0; i < matrix_size; ++i) {
    mismatches += output[i] == expected[i] ? 0 : 1;
  }
  const unsigned long long out_sum = checksum(output);
  if (const char* path = getenv("HECBENCH_DUMP")) {
    FILE* f = fopen(path, "ab");
    if (!f || fwrite(output.data(), sizeof(T), matrix_size, f) != matrix_size || fclose(f)) return 2;
  }
  const unsigned long long ref_sum = checksum(expected);
  std::printf("checksum %llu reference %llu mismatches %zu\n", out_sum, ref_sum, mismatches);
  std::printf("%s\n", mismatches == 0 ? "PASS" : "FAIL");
  return mismatches == 0 ? 0 : 1;
}

template <typename T>
static int eval_type(const char* kernel_name, int nrows, int ncols, int repeat)
{
  const size_t matrix_size = static_cast<size_t>(nrows) * ncols;
  std::vector<T> input(matrix_size);

  std::default_random_engine generator(123);
  std::uniform_int_distribution<int> distribution(0, 255);
#ifdef DEBUG
  std::printf("Input matrix:\n");
#endif
  for (size_t i = 0; i < matrix_size; ++i) {
    input[i] = static_cast<T>(distribution(generator));
#ifdef DEBUG
    std::printf("%d ", static_cast<int>(input[i]));
    if ((i + 1) % static_cast<size_t>(ncols) == 0) std::printf("\n");
#endif
  }

  int errors = 0;
  errors += eval_case(kernel_name, input, true, true, nrows, ncols, repeat);
  errors += eval_case(kernel_name, input, true, false, nrows, ncols, repeat);
  errors += eval_case(kernel_name, input, false, true, nrows, ncols, repeat);
  errors += eval_case(kernel_name, input, false, false, nrows, ncols, repeat);
  return errors;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: ./%s <nrows> <ncols> <iterations>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int nrows = std::atoi(argv[1]);
  const int ncols = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  int errors = 0;
  std::printf("\nThe size of each matrix element is %zu byte\n", sizeof(unsigned char));
  errors += eval_type<unsigned char>("reverse2d_u8", nrows, ncols, repeat);
  std::printf("\nThe size of each matrix element is %zu bytes\n", sizeof(uint16_t));
  errors += eval_type<uint16_t>("reverse2d_u16", nrows, ncols, repeat);
  std::printf("\nThe size of each matrix element is %zu bytes\n", sizeof(uint32_t));
  errors += eval_type<uint32_t>("reverse2d_u32", nrows, ncols, repeat);
  std::printf("\nThe size of each matrix element is %zu bytes\n", sizeof(uint64_t));
  errors += eval_type<uint64_t>("reverse2d_u64", nrows, ncols, repeat);

  platform.finalize();
  return errors == 0 ? 0 : 1;
}

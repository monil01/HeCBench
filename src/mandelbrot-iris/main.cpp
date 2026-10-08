#include <iris/iris.hpp>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

constexpr int row_size = 1080;
constexpr int col_size = 1920;
constexpr int max_iterations = 100;
constexpr int threads_per_block_x = 16;
constexpr int threads_per_block_y = 16;

struct ComplexF {
  float real;
  float imag;
};

static float scale_row(int i, int rows)
{
  return -1.5f + (i * (2.0f / rows));
}

static float scale_col(int i, int cols)
{
  return -1.0f + (i * (2.0f / cols));
}

static int mandel_point(const ComplexF& c, int max_iters)
{
  int count = 0;
  ComplexF z = {0.0f, 0.0f};
  for (int i = 0; i < max_iters; ++i) {
    const float r = z.real;
    const float im = z.imag;
    if (((r * r) + (im * im)) >= 4.0f) {
      break;
    }
    z.real = r * r - im * im + c.real;
    z.imag = 2.0f * r * im + c.imag;
    count++;
  }
  return count;
}

static void evaluate_serial(std::vector<int>& data, int rows, int cols, int max_iters)
{
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      ComplexF c = {scale_row(i, rows), scale_col(j, cols)};
      data[static_cast<size_t>(i) * cols + j] = mandel_point(c, max_iters);
    }
  }
}

static void print_data(const std::vector<int>& data, int rows, int cols, int max_iters)
{
  if (rows > 128 || cols > 128) {
    std::cout << "No Print() output due to size too large" << std::endl;
    return;
  }
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << std::setw(1)
                << ((data[static_cast<size_t>(i) * cols + j] >= max_iters) ? "x" : " ");
    }
    std::cout << std::endl;
  }
}

static void verify(const std::vector<int>& parallel, const std::vector<int>& serial)
{
  int diff = 0;
  for (size_t i = 0; i < parallel.size(); ++i) {
    if (parallel[i] != serial[i]) diff++;
  }
  const double ratio = static_cast<double>(diff) / static_cast<double>(parallel.size());
  if (ratio > 0.05) {
    std::cout << "Fail verification - diff larger than tolerance" << std::endl;
    throw std::runtime_error("Verification failure");
  }
}

static double launch_mandel(iris::DMem& mem_data, int rows, int cols, int max_iters)
{
  const size_t gws[2] = {
    static_cast<size_t>((rows + threads_per_block_x - 1) / threads_per_block_x) *
        threads_per_block_x,
    static_cast<size_t>((cols + threads_per_block_y - 1) / threads_per_block_y) *
        threads_per_block_y,
  };
  const size_t lws[2] = {threads_per_block_x, threads_per_block_y};

  iris::Task task;
  void* params[4] = {&mem_data, &rows, &cols, &max_iters};
  int pinfo[4] = {iris_w, sizeof(rows), sizeof(cols), sizeof(max_iters)};

  auto start = std::chrono::steady_clock::now();
  task.kernel("mandel", 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
              4, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::cout << " Incorrect parameters\n";
    std::cout << " Usage: " << argv[0] << " <repeat>\n\n";
    return -1;
  }

  try {
    iris::Platform platform;
    platform.init(&argc, &argv, 1);

    const int repetitions = std::atoi(argv[1]);
    std::vector<int> parallel(static_cast<size_t>(row_size) * col_size, 0);
    std::vector<int> serial(static_cast<size_t>(row_size) * col_size, 0);
    iris::DMem mem_parallel(parallel.data(), parallel.size() * sizeof(int));

    launch_mandel(mem_parallel, row_size, col_size, max_iterations);

    double kernel_time = 0.0;
    auto parallel_start = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i) {
      kernel_time += launch_mandel(mem_parallel, row_size, col_size, max_iterations);
    }
    iris::Task flush;
    flush.flush_out(mem_parallel);
    flush.submit(iris_gpu, nullptr, 1);
    mem_parallel.fetch(parallel.data());
    auto parallel_end = std::chrono::steady_clock::now();

    print_data(parallel, row_size, col_size, max_iterations);

    auto serial_start = std::chrono::steady_clock::now();
    evaluate_serial(serial, row_size, col_size, max_iterations);
    auto serial_end = std::chrono::steady_clock::now();

    const auto parallel_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(parallel_end - parallel_start);
    const auto serial_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(serial_end - serial_start);

    std::cout << std::setw(20) << "Serial time: " << serial_time.count() << " s\n";
    std::cout << std::setw(20) << "Average parallel time: "
              << (parallel_time / repetitions).count() * 1e3 << " ms\n";
    std::cout << std::setw(20) << "Average kernel execution time: "
              << kernel_time / repetitions * 1e3 << " ms\n";

    verify(parallel, serial);
    platform.finalize();
  } catch (...) {
    std::cout << "Failure\n";
    std::terminate();
  }

  std::cout << "Success\n";
  return 0;
}

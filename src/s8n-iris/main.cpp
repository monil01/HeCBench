#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "reference.h"

static double launch_select(const char* kernel_name, int b, int n, int radius,
                            int* h_xyz, int* h_out, size_t out_bytes, int repeat) {
  constexpr size_t block_size = 512;
  size_t gws[] = {static_cast<size_t>(b) * block_size};
  size_t lws[] = {block_size};
  iris::DMem mem_xyz(h_xyz, static_cast<size_t>(b) * n * 3 * sizeof(int));
  iris::DMem mem_out(h_out, out_bytes);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    iris::Task task;
    void* params[] = {&b, &n, &radius, &mem_xyz, &mem_out};
    int pinfo[] = {
        static_cast<int>(sizeof(b)),
        static_cast<int>(sizeof(n)),
        static_cast<int>(sizeof(radius)),
        iris_r,
        iris_w,
    };
    task.kernel(kernel_name, 1, nullptr, gws, lws, 5, params, pinfo);
    task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() *
         1e-3 / repeat;
}

int main(int argc, char** argv) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  if (argc != 4) {
    printf("Usage: %s <number of batches> <number of points> <repeat>\n", argv[0]);
    platform.finalize();
    return 1;
  }
  const int b = atoi(argv[1]);
  const int n = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  if (b <= 0 || n <= 0 || repeat <= 0) {
    fprintf(stderr, "b, n, and repeat must be positive\n");
    platform.finalize();
    return 1;
  }

  const int input_size = b * n * 3;
  const size_t input_size_bytes = static_cast<size_t>(input_size) * sizeof(int);
  const size_t output_size = static_cast<size_t>(b) * n * 8;
  const size_t output_size_bytes = output_size * sizeof(int);
  const int radius = 512;

  int* h_xyz = static_cast<int*>(malloc(input_size_bytes));
  int* h_out = static_cast<int*>(malloc(output_size_bytes));
  int* r_out = static_cast<int*>(malloc(output_size_bytes));
  int* h_out2 = static_cast<int*>(malloc(2 * output_size_bytes));
  int* r_out2 = static_cast<int*>(malloc(2 * output_size_bytes));
  int* h_out4 = static_cast<int*>(malloc(4 * output_size_bytes));
  int* r_out4 = static_cast<int*>(malloc(4 * output_size_bytes));
  if (!h_xyz || !h_out || !r_out || !h_out2 || !r_out2 || !h_out4 || !r_out4) {
    fprintf(stderr, "host allocation failed\n");
    platform.finalize();
    return 1;
  }

  std::default_random_engine g(123);
  std::uniform_int_distribution<> distr(-256, 255);
  for (int i = 0; i < input_size; i++) {
    h_xyz[i] = distr(g);
  }

  double time =
      launch_select("k_cube_select", b, n, radius, h_xyz, h_out, output_size_bytes, repeat);
  printf("Average execution time of select kernel: %f (us)\n", time);
  cube_select(b, n, radius, h_xyz, r_out);
  int error = memcmp(h_out, r_out, output_size_bytes);

  time = launch_select("k_cube_select_two", b, n, radius, h_xyz, h_out2,
                       2 * output_size_bytes, repeat);
  printf("Average execution time of select2 kernel: %f (us)\n", time);
  cube_select_two(b, n, radius, h_xyz, r_out2);
  error += memcmp(h_out2, r_out2, 2 * output_size_bytes);

  time = launch_select("k_cube_select_four", b, n, radius, h_xyz, h_out4,
                       4 * output_size_bytes, repeat);
  printf("Average execution time of select4 kernel: %f (us)\n", time);
  cube_select_four(b, n, radius, h_xyz, r_out4);
  error += memcmp(h_out4, r_out4, 4 * output_size_bytes);

  printf("%s\n", error ? "FAIL" : "PASS");

  free(h_xyz);
  free(h_out);
  free(h_out2);
  free(h_out4);
  free(r_out);
  free(r_out2);
  free(r_out4);
  platform.finalize();
  return 0;
}

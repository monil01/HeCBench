#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include "reference.h"

#define BLOCK_SIZE 256

static void run_glu(
    int split_index,
    int m,
    int split_dim_size,
    int n,
    int repeat,
    iris::DMem& mem_x,
    iris::DMem& mem_y)
{
  const int total = m * split_dim_size * n;
  const size_t blocks = (static_cast<size_t>(total) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const size_t gws[1] = { blocks * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[5] = { &m, &split_dim_size, &n, &mem_x, &mem_y };
    int pinfo[5] = { sizeof(m), sizeof(split_dim_size), sizeof(n), iris_r, iris_w };
    task.kernel("glu_kernel", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 5, params, pinfo);
    if (i == repeat - 1) {
      task.flush_out(mem_y);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of GLU kernel (split dimension = %d): %f (us)\n",
              split_index, (time * 1e-3f) / repeat);
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <number of dimensions> <size of each dimension> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int ndims = std::atoi(argv[1]);
  const int dim_size = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  std::vector<int> Xshape;
  for (int i = 0; i < ndims; i++) {
    Xshape.push_back(dim_size);
  }

  std::vector<int> Yshape;
  Yshape.insert(Yshape.end(), Xshape.begin(), Xshape.end());

  std::printf("Shape of input tensor: ( ");
  for (int i = 0; i < ndims; i++) {
    std::printf("%d ", Xshape[i]);
  }
  std::printf(")\n");

  const uint64_t nelems = size_from_dim(0, Xshape);
  const uint64_t nelems_bytes = nelems * sizeof(float);

  float* X = static_cast<float*>(std::malloc(nelems_bytes));
  float* Y = static_cast<float*>(std::malloc(nelems_bytes));
  float* Y_ref = static_cast<float*>(std::malloc(nelems_bytes));
  if (!X || !Y || !Y_ref) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(X);
    std::free(Y);
    std::free(Y_ref);
    platform.finalize();
    return 1;
  }

  for (uint64_t i = 0; i < nelems; i++) {
    X[i] = static_cast<float>(i);
    Y[i] = 0.0f;
    Y_ref[i] = 0.0f;
  }

  iris::DMem mem_x(X, nelems_bytes);
  iris::DMem mem_y(Y, nelems_bytes);

  for (int input_dim = -1; input_dim < 3 * (ndims - 1); input_dim++) {
    const int split_index = (input_dim == -1) ? ndims - 1 : (input_dim % ndims);

    if (Yshape[split_index] % 2 != 0) {
      std::printf("Split dimension %d should be divided by two. Skip\n", Yshape[split_index]);
      continue;
    }

    const int split_dim_size = Yshape[split_index] / 2;
    const int m = size_to_dim(split_index, Xshape);
    const int n = size_from_dim(split_index + 1, Xshape);

    ComputeGlu(m, split_dim_size, n, X, Y_ref);
    run_glu(split_index, m, split_dim_size, n, repeat, mem_x, mem_y);
    mem_y.fetch(Y);

    bool ok = true;
    for (uint64_t i = 0; i < nelems / 2; i++) {
      if (std::fabs(Y[i] - Y_ref[i]) > 1e-3f) {
        ok = false;
        break;
      }
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
  }

  std::free(X);
  std::free(Y);
  std::free(Y_ref);
  platform.finalize();
  return 0;
}

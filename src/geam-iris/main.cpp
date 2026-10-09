#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#define BLOCK_SIZE 16

template <typename T>
int transpose_matrix(const char* kernel_name, const char* label, int nrow, int ncol, int repeat)
{
  int error = 0;
  const size_t size = static_cast<size_t>(nrow) * static_cast<size_t>(ncol);
  const size_t size_byte = size * sizeof(T);

  T* matrix = static_cast<T*>(std::malloc(size_byte));
  T* matrixT = static_cast<T*>(std::malloc(size_byte));
  T* h_matrixT = static_cast<T*>(std::malloc(size_byte));
  if (!matrix || !matrixT || !h_matrixT) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(h_matrixT);
    std::free(matrixT);
    std::free(matrix);
    return 1;
  }

  for (size_t i = 0; i < size; i++) {
    matrix[i] = static_cast<T>(std::rand() % 13);
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      matrixT[(j * nrow) + i] = matrix[(i * ncol) + j];
    }
  }
  auto end = std::chrono::steady_clock::now();
  double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Host: serial matrix transpose time = %f (ms)\n", time * 1e-6f);

  iris::DMem mem_matrix(matrix, size_byte);
  iris::DMem mem_matrixT(h_matrixT, size_byte);

  time = 0.0;
  const int warmup = 4;
  const size_t gws[2] = {
    static_cast<size_t>(((ncol + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE),
    static_cast<size_t>(((nrow + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE)
  };
  const size_t lws[2] = { BLOCK_SIZE, BLOCK_SIZE };

  for (int i = 0; i < repeat + warmup; i++) {
    if (i >= warmup) {
      start = std::chrono::steady_clock::now();
    }

    iris::Task task;
    void* params[4] = { &mem_matrixT, &mem_matrix, &nrow, &ncol };
    int pinfo[4] = { iris_w, iris_r, sizeof(nrow), sizeof(ncol) };
    task.kernel(kernel_name, 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 4, params, pinfo);
    if (i == repeat + warmup - 1) {
      task.flush_out(mem_matrixT);
    }
    task.submit(iris_gpu, nullptr, 1);

    if (i >= warmup) {
      end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
  }

  std::printf("Device: average matrix transpose time = %f (ms)\n", (time * 1e-6f) / repeat);

  if (error == 0) {
    error = std::memcmp(h_matrixT, matrixT, size_byte);
  }

  std::printf("%s\n", error ? "FAIL" : "PASS");

  std::free(h_matrixT);
  std::free(matrixT);
  std::free(matrix);
  (void)label;
  return error ? 1 : 0;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage %s <matrix row> <matrix col> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int nrow = std::atoi(argv[1]);
  const int ncol = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);
  if (nrow <= 0 || ncol <= 0 || repeat < 0) {
    std::printf("Error: invalid inputs\n");
    platform.finalize();
    return 1;
  }

  int errors = 0;
  std::printf("----------------FP32 transpose matrix (%d x %d)----------------\n", nrow, ncol);
  errors += transpose_matrix<float>("transpose_f32_kernel", "FP32", nrow, ncol, repeat);

  std::printf("----------------FP64 transpose matrix (%d x %d)----------------\n", nrow, ncol);
  errors += transpose_matrix<double>("transpose_f64_kernel", "FP64", nrow, ncol, repeat);

  platform.finalize();
  return errors == 0 ? 0 : 1;
}

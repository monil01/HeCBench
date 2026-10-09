#include <iris/iris.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "reference.h"

static void property(const char* name, const std::vector<int64_t>& p)
{
  std::printf("%s: ( ", name);
  for (uint64_t i = 0; i < p.size(); i++) {
    std::printf("%ld ", p[i]);
  }
  std::printf(")\n");
}

template <typename scalar_t>
static void launch_flip(
    const char* kernel_name,
    int64_t n,
    int64_t num_flip_dims,
    int64_t num_dims,
    bool flush_output,
    iris::DMem& mem_input,
    iris::DMem& mem_output,
    iris::DMem& mem_flip_dims,
    iris::DMem& mem_stride,
    iris::DMem& mem_stride_contiguous,
    iris::DMem& mem_shape)
{
  const size_t threads_per_block = 256;
  const size_t blocks = (static_cast<size_t>(n) + threads_per_block - 1) / threads_per_block;
  size_t gws[1] = { blocks * threads_per_block };
  size_t lws[1] = { threads_per_block };

  iris::Task task;
  void* params[9] = {
    &mem_input,
    &mem_output,
    &n,
    &mem_flip_dims,
    &num_flip_dims,
    &mem_stride,
    &mem_stride_contiguous,
    &mem_shape,
    &num_dims
  };
  int pinfo[9] = {
    iris_r,
    iris_w,
    sizeof(n),
    iris_r,
    sizeof(num_flip_dims),
    iris_r,
    iris_r,
    iris_r,
    sizeof(num_dims)
  };
  task.kernel(kernel_name, 1, nullptr, gws, lws, 9, params, pinfo);
  if (flush_output) {
    task.flush_out(mem_output);
  }
  task.submit(iris_gpu, nullptr, 1);
}

template <typename scalar_t>
static bool flip(
    iris::Platform& platform,
    const char* kernel_name,
    const int64_t num_dims,
    const int64_t num_flip_dims,
    const int32_t dim_size,
    const int32_t repeat)
{
  std::vector<int64_t> flip_dims;
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;

  for (int64_t i = 0; i < num_dims; i++) {
#ifdef EXAMPLE
    shape.push_back(2);
#else
    shape.push_back(dim_size);
#endif
  }

  int64_t n = 1;
  for (int64_t i = 0; i < num_dims; i++) {
    n = n * shape[i];
  }

  for (int64_t i = 0; i < num_flip_dims; i++) {
    flip_dims.push_back(i);
  }

  stride.push_back(shape[1] * shape[2]);
  stride.push_back(shape[2]);
  stride.push_back(1);

  property("shape", shape);
  property("flip_dims", flip_dims);
  property("stride", stride);

  const size_t input_size_bytes = static_cast<size_t>(n) * sizeof(scalar_t);
  const size_t output_size_bytes = input_size_bytes;
  const size_t dims_bytes = static_cast<size_t>(num_dims) * sizeof(int64_t);
  const size_t flip_dims_bytes = static_cast<size_t>(num_flip_dims) * sizeof(int64_t);

  scalar_t* input = static_cast<scalar_t*>(std::malloc(input_size_bytes));
  scalar_t* output = static_cast<scalar_t*>(std::malloc(output_size_bytes));
  scalar_t* output_ref = static_cast<scalar_t*>(std::malloc(output_size_bytes));
  if (!input || !output || !output_ref) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(input);
    std::free(output);
    std::free(output_ref);
    platform.finalize();
    std::exit(1);
  }

  for (int64_t i = 0; i < n; i++) {
    input[i] = static_cast<scalar_t>(i);
    output[i] = static_cast<scalar_t>(0);
    output_ref[i] = static_cast<scalar_t>(0);
  }

  iris::DMem mem_input(input, input_size_bytes);
  iris::DMem mem_output(output, output_size_bytes);
  iris::DMem mem_flip_dims(flip_dims.data(), flip_dims_bytes);
  iris::DMem mem_shape(shape.data(), dims_bytes);
  iris::DMem mem_stride(stride.data(), dims_bytes);
  iris::DMem mem_stride_contiguous(stride.data(), dims_bytes);

  launch_flip<scalar_t>(
      kernel_name,
      n,
      num_flip_dims,
      num_dims,
      true,
      mem_input,
      mem_output,
      mem_flip_dims,
      mem_stride,
      mem_stride_contiguous,
      mem_shape);

  flip_kernel_cpu<scalar_t>(
      input,
      output_ref,
      n,
      flip_dims.data(),
      num_flip_dims,
      stride.data(),
      stride.data(),
      shape.data(),
      num_dims);

  mem_output.fetch(output);
  const int error = std::memcmp(output, output_ref, output_size_bytes);
  std::printf("%s\n", error ? "FAIL" : "PASS");

#ifdef EXAMPLE
  for (int64_t i = 0; i < n; i++) {
    std::printf("%f ", static_cast<double>(output[i]));
  }
  std::printf("\n");
#endif

  auto start = std::chrono::steady_clock::now();
  for (int32_t i = 0; i < repeat; i++) {
    launch_flip<scalar_t>(
        kernel_name,
        n,
        num_flip_dims,
        num_dims,
        false,
        mem_input,
        mem_output,
        mem_flip_dims,
        mem_stride,
        mem_stride_contiguous,
        mem_shape);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of the flip kernel: %f (ms)\n",
              (time * 1e-6f) / repeat);

  std::free(input);
  std::free(output);
  std::free(output_ref);
  return error == 0;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <number of dimensions> <size of each dimension> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int64_t num_dims = std::atoi(argv[1]);
  const int64_t dim_size = std::atoi(argv[2]);
  const int32_t repeat = std::atoi(argv[3]);

#ifdef EXAMPLE
  const int64_t num_flip_dims = 2;
#else
  const int64_t num_flip_dims = num_dims;
#endif

  std::printf("=========== Data type is FP32 ==========\n");
  bool ok = flip<float>(platform, "flip_kernel_float", num_dims, num_flip_dims, dim_size, repeat);

  std::printf("=========== Data type is FP64 ==========\n");
  ok = flip<double>(platform, "flip_kernel_double", num_dims, num_flip_dims, dim_size, repeat) && ok;

  platform.finalize();
  return ok ? 0 : 1;
}

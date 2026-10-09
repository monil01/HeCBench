#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define TILE_SIZE 5900
#define NTHREADS 256

static const int d1 = 41, d2 = 13, d3 = 11, d4 = 9, d5 = 76, d6 = 50;
static const int data_size = d1 * d2 * d3 * d4 * d5 * d6;
static int repeat = 1;

static const int shape_output[] = {d2, d3, d1};
static const int shape_input[] = {d4, d5, d6};
static const float shape_output_r[] = {1.f / d2, 1.f / d3, 1.f / d1};
static const float shape_input_r[] = {1.f / d4, 1.f / d5, 1.f / d6};
static const int stride_output_local[] = {d1, d1 * d2, 1};
static const int stride_output_global[] = {1, d2, d2 * d3 * d4 * d6};
static const int stride_input[] = {d2 * d3, d2 * d3 * d4 * d6 * d1, d2 * d3 * d4};

static bool verify(const std::vector<double>& input, const std::vector<double>& output)
{
  int input_offset = 2 + d1 * (2 + d2 * (2 + d3 * (2 + d4 * (0 + 2 * d5))));
  int output_offset = 2 + d2 * (2 + d3 * (2 + d4 * (2 + d6 * (2 + 0 * d1))));
  bool error = false;
  for (size_t i = 0; i < d5; i++) {
    if (input[input_offset + i * d1 * d2 * d3 * d4] !=
        output[output_offset + i * d2 * d3 * d4 * d6 * d1]) {
      std::printf("FAIL\n");
      error = true;
      break;
    }
  }
  if (!error) std::printf("PASS\n");
  return !error;
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  repeat = std::atoi(argv[1]);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  std::vector<double> input(static_cast<size_t>(data_size), 0.0);
  std::vector<double> output(static_cast<size_t>(data_size), 0.0);

  for (size_t i = 0; i < static_cast<size_t>(data_size); i++) {
    input[i] = static_cast<double>(i);
  }

  int nblocks = d4 * d5 * d6;
  int tile_size = d1 * d2 * d3;
  int dim_output = 3;
  int dim_input = 3;

  iris::DMem d_input(input.data(), input.size() * sizeof(double));
  iris::DMem d_output(output.data(), output.size() * sizeof(double));
  iris::DMem d_shape_input(const_cast<int*>(shape_input), sizeof(shape_input));
  iris::DMem d_shape_input_r(const_cast<float*>(shape_input_r), sizeof(shape_input_r));
  iris::DMem d_shape_output(const_cast<int*>(shape_output), sizeof(shape_output));
  iris::DMem d_shape_output_r(const_cast<float*>(shape_output_r), sizeof(shape_output_r));
  iris::DMem d_stride_input(const_cast<int*>(stride_input), sizeof(stride_input));
  iris::DMem d_stride_output_local(const_cast<int*>(stride_output_local), sizeof(stride_output_local));
  iris::DMem d_stride_output_global(const_cast<int*>(stride_output_global), sizeof(stride_output_global));

  const size_t gws[1] = {static_cast<size_t>(nblocks) * NTHREADS};
  const size_t lws[1] = {NTHREADS};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    iris::Task task;
    void* params[13] = {
        &dim_input,
        &dim_output,
        &nblocks,
        &tile_size,
        &d_shape_input,
        &d_shape_output,
        &d_shape_input_r,
        &d_shape_output_r,
        &d_stride_input,
        &d_stride_output_local,
        &d_stride_output_global,
        &d_input,
        &d_output,
    };
    int pinfo[13] = {
        sizeof(dim_input),
        sizeof(dim_output),
        sizeof(nblocks),
        sizeof(tile_size),
        iris_r,
        iris_r,
        iris_r,
        iris_r,
        iris_r,
        iris_r,
        iris_r,
        iris_r,
        iris_w,
    };
    task.kernel("tensor_transpose", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 13, params, pinfo);
    task.flush_out(d_output);
    task.submit(iris_gpu, nullptr, 1);
  }

  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (ms)\n",
              (time * 1e-6f) / repeat);

  d_output.fetch(output.data());
  bool ok = verify(input, output);

  platform.finalize();
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

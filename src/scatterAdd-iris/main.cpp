#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define MAX_THREADS_PER_BLOCK 512
#define WORK_SIZE 8192

static void scatter_add_reference(
    int batch_size,
    int vector_dim,
    float* out,
    int* idx,
    float* src)
{
  for (int d = 0; d < vector_dim; d++) {
    for (int i = 0; i < batch_size; i++) {
      int index = idx[i];
      out[index * vector_dim + d] += src[i * vector_dim + d];
    }
  }
}

static void launch_scatter(
    const char* kernel_name,
    int batch_size,
    int output_size,
    int vector_dim,
    int warp_size,
    iris::DMem& mem_index,
    iris::DMem& mem_source,
    iris::DMem& mem_output,
    bool fetch_output)
{
  int thread_x = warp_size;
  int thread_y = MAX_THREADS_PER_BLOCK / warp_size;
  int block_x = batch_size / WORK_SIZE + 1;
  size_t gws[2] = {
    static_cast<size_t>(block_x * thread_x),
    static_cast<size_t>(thread_y)
  };
  size_t lws[2] = {
    static_cast<size_t>(thread_x),
    static_cast<size_t>(thread_y)
  };

  iris::Task task;
  void* params[6] = {
    &mem_index,
    &mem_source,
    &mem_output,
    &batch_size,
    &output_size,
    &vector_dim
  };
  int pinfo[6] = {
    iris_r,
    iris_r,
    iris_rw,
    sizeof(batch_size),
    sizeof(output_size),
    sizeof(vector_dim)
  };
  task.kernel(kernel_name, 2, nullptr, gws, lws, 6, params, pinfo);
  if (fetch_output) {
    task.flush_out(mem_output);
  }
  task.submit(iris_gpu, nullptr, 1);
}

static void index_accumulate(int batch_size, int output_size, int vector_dim, int repeat)
{
  const size_t source_size_bytes = static_cast<size_t>(batch_size) * static_cast<size_t>(vector_dim) * sizeof(float);
  const size_t output_size_bytes = static_cast<size_t>(output_size) * static_cast<size_t>(vector_dim) * sizeof(float);
  const size_t index_size_bytes = static_cast<size_t>(batch_size) * sizeof(int);

  int* index = static_cast<int*>(std::malloc(index_size_bytes));
  float* source = static_cast<float*>(std::malloc(source_size_bytes));
  float* output = static_cast<float*>(std::malloc(output_size_bytes));
  float* output_ref = static_cast<float*>(std::malloc(output_size_bytes));

  if (!index || !source || !output || !output_ref) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(source);
    std::free(index);
    std::free(output);
    std::free(output_ref);
    return;
  }

  std::srand(2);
  for (int i = 0; i < batch_size; i++) {
    index[i] = std::rand() % output_size;
  }

  for (int i = 0; i < batch_size * vector_dim; i++) {
    source[i] = -1.f;
  }

  std::memset(output_ref, 0, output_size_bytes);
  scatter_add_reference(batch_size, vector_dim, output_ref, index, source);
  std::memset(output, 0, output_size_bytes);

  iris::DMem mem_index(index, index_size_bytes);
  iris::DMem mem_source(source, source_size_bytes);
  iris::DMem mem_output(output, output_size_bytes);

  const int warp_size = 64;

  for (int i = 0; i < 10; i++) {
    std::memset(output, 0, output_size_bytes);
    mem_output.update(output);
    launch_scatter("scatterAdd2_kernel", batch_size, output_size, vector_dim,
                   warp_size, mem_index, mem_source, mem_output, i == 9);
  }

  mem_output.fetch(output);
  bool ok = true;
  for (int i = 0; i < output_size * vector_dim; i++) {
    if (std::fabs(output[i] - output_ref[i]) > 1e-3f) {
      std::printf("output %d: %f %f\n", i, output[i], output_ref[i]);
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    launch_scatter("scatterAdd_kernel", batch_size, output_size, vector_dim,
                   warp_size, mem_index, mem_source, mem_output, false);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of kernel1: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    launch_scatter("scatterAdd2_kernel", batch_size, output_size, vector_dim,
                   warp_size, mem_index, mem_source, mem_output, false);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of kernel2: %f (us)\n", (time * 1e-3f) / repeat);

  std::free(source);
  std::free(index);
  std::free(output);
  std::free(output_ref);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("Usage: %s <batch size> <output size> <vector dimension> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int batch_size = std::atoi(argv[1]);
  const int output_size = std::atoi(argv[2]);
  const int vector_dim = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);

  std::printf("batch_size: %d\n", batch_size);
  std::printf("output_size (range of index values): %d\n", output_size);
  std::printf("vector_dimension: %d\n", vector_dim);

  index_accumulate(batch_size, output_size, vector_dim, repeat);

  platform.finalize();
  return 0;
}

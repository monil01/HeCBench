#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "reference.h"

#define GPU_NUM_THREADS 256

static int run_kernel(
    const char* name,
    int nrows,
    int ndims,
    int top_k,
    int repeat,
    int ngrid,
    int count_ref,
    iris::DMem& mem_data,
    iris::DMem& mem_label,
    iris::DMem& mem_count)
{
  int count = 0;
  const size_t gws[1] = { static_cast<size_t>(ngrid) * GPU_NUM_THREADS };
  const size_t lws[1] = { GPU_NUM_THREADS };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    count = 0;
    mem_count.update(&count);

    iris::Task task;
    void* params[6] = {
      &nrows,
      &ndims,
      &top_k,
      &mem_data,
      &mem_label,
      &mem_count
    };
    int pinfo[6] = {
      sizeof(nrows),
      sizeof(ndims),
      sizeof(top_k),
      iris_r,
      iris_r,
      iris_rw
    };
    task.kernel(name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 6, params, pinfo);
    task.flush_out(mem_count);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of %s: %f (us)\n", name, (time * 1e-3f) / repeat);

  mem_count.fetch(&count);
  std::printf("%s\n", (count == count_ref) ? "PASS" : "FAIL");
  return count == count_ref ? 0 : 1;
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("Usage: %s <number of rows> <number of columns> <top K> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int nrows = std::atoi(argv[1]);
  const int ndims = std::atoi(argv[2]);
  const int top_k = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);

  const int data_size = nrows * ndims;
  const int label_size_bytes = nrows * static_cast<int>(sizeof(int));
  const size_t data_size_bytes = static_cast<size_t>(data_size) * sizeof(float);

  int* label = static_cast<int*>(std::malloc(label_size_bytes));
  float* data = static_cast<float*>(std::malloc(data_size_bytes));
  if (!label || !data) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(label);
    std::free(data);
    platform.finalize();
    return 1;
  }

  std::srand(123);
  for (int i = 0; i < nrows; i++) {
    label[i] = std::rand() % ndims;
  }

  std::default_random_engine g(123);
  std::uniform_real_distribution<float> distr(0.f, 1.f);
  for (int i = 0; i < data_size; i++) {
    data[i] = distr(g);
  }

  const int count_ref = reference(nrows, ndims, top_k, data, label);
  int count = 0;

  iris::DMem mem_label(label, label_size_bytes);
  iris::DMem mem_data(data, data_size_bytes);
  iris::DMem mem_count(&count, sizeof(int));

  int errors = 0;
  for (int ngrid = nrows / 4; ngrid <= nrows; ngrid += nrows / 4) {
    std::printf("Grid size is %d\n", ngrid);
    errors += run_kernel("accuracy_kernel", nrows, ndims, top_k, repeat, ngrid, count_ref,
                         mem_data, mem_label, mem_count);
    errors += run_kernel("accuracy_kernel2", nrows, ndims, top_k, repeat, ngrid, count_ref,
                         mem_data, mem_label, mem_count);
  }

  std::free(label);
  std::free(data);
  platform.finalize();
  return errors == 0 ? 0 : 1;
}

#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "reference.h"

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
void srand48(long seed) { std::srand(static_cast<unsigned int>(seed)); }
double drand48() { return double(std::rand()) / RAND_MAX; }
#endif

template <typename T>
struct Task {
  unsigned int size, id;
  T* data;
  T* result;
  T* vector;

  Task() : size(0), id(0), data(nullptr), result(nullptr), vector(nullptr) {}

  void deallocate()
  {
    std::free(data);
    std::free(result);
    std::free(vector);
  }

  void allocate(const unsigned int s, const unsigned int unique_id)
  {
    id = unique_id;
    size = s;
    data = static_cast<T*>(std::malloc(sizeof(T) * size * size));
    result = static_cast<T*>(std::malloc(sizeof(T) * size));
    vector = static_cast<T*>(std::malloc(sizeof(T) * size));
    if (!data || !result || !vector) {
      std::fprintf(stderr, "host allocation failed for task %u size %u\n", id, size);
      std::exit(1);
    }

    for (unsigned int i = 0; i < size * size; i++) {
      data[i] = drand48();
    }
    for (unsigned int i = 0; i < size; i++) {
      result[i] = 0.0;
      vector[i] = drand48();
    }
  }
};

template <typename T>
void execute(Task<T>& t)
{
  if (t.size < 100) {
    gemv(t.size, t.size, static_cast<T>(1.0), t.data, t.vector,
         static_cast<T>(0.0), t.result);
    return;
  }

  const size_t matrix_bytes = static_cast<size_t>(t.size) * t.size * sizeof(T);
  const size_t vector_bytes = static_cast<size_t>(t.size) * sizeof(T);
  iris::DMem mem_data(t.data, matrix_bytes);
  iris::DMem mem_vector(t.vector, vector_bytes);
  iris::DMem mem_result(t.result, vector_bytes);

  const size_t gws[1] = { static_cast<size_t>(((t.size + 255U) / 256U) * 256U) };
  const size_t lws[1] = { 256 };
  iris::Task iris_task;
  void* params[4] = { &mem_result, &mem_data, &mem_vector, &t.size };
  int pinfo[4] = { iris_w, iris_r, iris_r, sizeof(t.size) };
  iris_task.kernel("streamum_dgemv_col_major", 1, nullptr,
                   const_cast<size_t*>(gws), const_cast<size_t*>(lws), 4,
                   params, pinfo);
  iris_task.flush_out(mem_result);
  iris_task.submit(iris_gpu, nullptr, 1);
}

template <typename T>
void initialise_tasks(std::vector<Task<T> >& task_list)
{
  for (unsigned int i = 0; i < task_list.size(); i++) {
    int size = std::max(static_cast<int>(drand48() * 1000.0), 64);
    task_list[i].allocate(size, i);
  }
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    std::printf("Usage: %s <number of host threads> <number of tasks> <verify>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int nthreads = std::atoi(argv[1]);
  const unsigned int N = std::atoi(argv[2]);
  const int verify = std::atoi(argv[3]);
  if (nthreads <= 0 || N == 0) {
    std::fprintf(stderr, "invalid arguments\n");
    platform.finalize();
    return 1;
  }

  srand48(48);
  std::vector<Task<double> > task_list(N);
  initialise_tasks(task_list);

  std::printf("Executing tasks on host / device\n");
  auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < task_list.size(); i++) {
    execute(task_list[i]);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Task execution time : %f (s)\n", (time * 1e-9f));

  if (verify) {
    check(task_list);
  }

  for (size_t i = 0; i < task_list.size(); i++) {
    task_list[i].deallocate();
  }

  platform.finalize();
  return 0;
}

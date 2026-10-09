#include <iris/iris.hpp>

#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static inline float createRoundingFactor(float max, int n)
{
  float delta = (max * static_cast<float>(n)) / (1.f - 2.f * static_cast<float>(n) * FLT_EPSILON);
  int exp;
  std::frexp(delta, &exp);
  return std::ldexp(1.f, exp);
}

static inline float truncateWithRoundingFactor(float roundingFactor, float x)
{
  return (roundingFactor + x) - roundingFactor;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <number of arrays> <length of each array>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int nArrays = std::atoi(argv[1]);
  const int nElems = std::atoi(argv[2]);
  int launch_nArrays = nArrays;
  int launch_nElems = nElems;
  const size_t narray_size = sizeof(float) * static_cast<size_t>(nArrays);
  const size_t array_size = narray_size * static_cast<size_t>(nElems);

  float* arrays = static_cast<float*>(std::malloc(array_size));
  float* maxVal = static_cast<float*>(std::malloc(narray_size));
  float* result = static_cast<float*>(std::malloc(narray_size));
  float* factor = static_cast<float*>(std::malloc(narray_size));
  float* result_ref = static_cast<float*>(std::malloc(narray_size));
  float* zero_result = static_cast<float*>(std::calloc(static_cast<size_t>(nArrays), sizeof(float)));
  if (!arrays || !maxVal || !result || !factor || !result_ref || !zero_result) {
    std::fprintf(stderr, "host allocation failed\n");
    platform.finalize();
    return 1;
  }

  std::srand(123);
  float* arr = arrays;
  for (int n = 0; n < nArrays; n++) {
    float max = 0;
    for (int i = 0; i < nElems; i++) {
      arr[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
      if (std::rand() % 2) arr[i] = -1.f * arr[i];
      max = std::fmax(std::fabs(arr[i]), max);
    }
    factor[n] = createRoundingFactor(max, nElems);
    maxVal[n] = max;
    arr += nElems;
  }

  arr = arrays;
  for (int n = 0; n < nArrays; n++) {
    result_ref[n] = 0;
    for (int i = 0; i < nElems; i++) {
      result_ref[n] += truncateWithRoundingFactor(factor[n], arr[i]);
    }
    arr += nElems;
  }

  iris::DMem mem_arrays(arrays, array_size);
  iris::DMem mem_maxVal(maxVal, narray_size);
  iris::DMem mem_result(result, narray_size);

  const size_t gws[1] = { 256u * 256u };
  const size_t lws[1] = { 256u };

  mem_result.update(zero_result);
  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < nArrays; n++) {
    iris::Task task;
    void* params[6] = { &factor[n], &launch_nElems, &n, &mem_arrays, &mem_result, &launch_nArrays };
    int pinfo[6] = { sizeof(float), sizeof(launch_nElems), sizeof(n), iris_r, iris_rw, sizeof(launch_nArrays) };
    task.kernel("sumArray", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 6, params, pinfo);
    if (n == nArrays - 1) {
      task.flush_out(mem_result);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time (sumArray): %f (s)\n", (time * 1e-9f) / nArrays);

  mem_result.fetch(result);
  bool ok = !std::memcmp(result_ref, result, narray_size);
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  start = std::chrono::steady_clock::now();
  iris::Task task;
  void* params[5] = { &launch_nArrays, &launch_nElems, &mem_arrays, &mem_result, &mem_maxVal };
  int pinfo[5] = { sizeof(launch_nArrays), sizeof(launch_nElems), iris_r, iris_w, iris_r };
  task.kernel("sumArrays", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 5, params, pinfo);
  task.flush_out(mem_result);
  task.submit(iris_gpu, nullptr, 1);
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Kernel execution time (sumArrays): %f (s)\n", time * 1e-9f);

  mem_result.fetch(result);
  ok = !std::memcmp(result_ref, result, narray_size);
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::free(arrays);
  std::free(maxVal);
  std::free(result);
  std::free(factor);
  std::free(result_ref);
  std::free(zero_result);
  platform.finalize();
  return 0;
}

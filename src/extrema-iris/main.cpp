#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void clip_both(bool clip, int n, int& plus, int& minus)
{
  if (clip) {
    if (plus >= n) plus = n - 1;
    if (minus < 0) minus = 0;
  } else {
    if (plus >= n) plus -= n;
    if (minus < 0) minus += n;
  }
}

template <typename T>
static void cpu_relextrema_1d(int n, int order, bool clip, const T* inp, unsigned char* results)
{
  for (int tid = 0; tid < n; tid++) {
    const T data = inp[tid];
    bool temp = true;
    for (int o = 1; o < order + 1; o++) {
      int plus = tid + o;
      int minus = tid - o;
      clip_both(clip, n, plus, minus);
      temp &= data > inp[plus];
      temp &= data >= inp[minus];
    }
    results[tid] = temp ? 1 : 0;
  }
}

template <typename T>
static void cpu_relextrema_2d(int in_x,
                              int in_y,
                              int order,
                              bool clip,
                              int axis,
                              const T* inp,
                              unsigned char* results)
{
  for (int tx = 0; tx < in_y; tx++) {
    for (int ty = 0; ty < in_x; ty++) {
      const int tid = tx * in_x + ty;
      const T data = inp[tid];
      bool temp = true;
      for (int o = 1; o < order + 1; o++) {
        int plus;
        int minus;
        if (axis == 0) {
          plus = tx + o;
          minus = tx - o;
          clip_both(clip, in_y, plus, minus);
          plus = plus * in_x + ty;
          minus = minus * in_x + ty;
        } else {
          plus = ty + o;
          minus = ty - o;
          clip_both(clip, in_x, plus, minus);
          plus = tx * in_x + plus;
          minus = tx * in_x + minus;
        }
        temp &= data > inp[plus];
        temp &= data >= inp[minus];
      }
      results[tid] = temp ? 1 : 0;
    }
  }
}

template <typename T>
static const char* type_suffix();
template <> const char* type_suffix<int>() { return "int"; }
template <> const char* type_suffix<long>() { return "long"; }
template <> const char* type_suffix<float>() { return "float"; }
template <> const char* type_suffix<double>() { return "double"; }

static std::string kernel_name(const char* prefix, const char* type)
{
  return std::string(prefix) + "_" + type;
}

template <typename T>
static long test_1d(iris::Platform&,
                    int length,
                    int order,
                    bool clip,
                    int repeat,
                    const char* type)
{
  std::vector<T> x(length);
  for (int i = 0; i < length; i++) x[i] = static_cast<T>(std::rand() % length);
  std::vector<unsigned char> cpu_r(length);
  std::vector<unsigned char> gpu_r(length);

  iris::DMem mem_x(x.data(), x.size() * sizeof(T));
  iris::DMem mem_result(gpu_r.data(), gpu_r.size() * sizeof(unsigned char));
  int clip_i = clip ? 1 : 0;
  const std::string name = kernel_name("relextrema_1d", type_suffix<T>());
  size_t gws[1] = {((static_cast<size_t>(length) + 255) / 256) * 256};
  size_t lws[1] = {256};

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    void* params[] = {&length, &order, &clip_i, &mem_x, &mem_result};
    int pinfo[] = {sizeof(length), sizeof(order), sizeof(clip_i), iris_r, iris_w};
    iris::Task task;
    task.kernel(name.c_str(), 1, nullptr, gws, lws, 5, params, pinfo);
    task.flush_out(mem_result);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average 1D kernel (type = %s, order = %d, clip = %d) execution time %f (s)\n",
              type, order, clip, (time * 1e-9f) / repeat);

  mem_result.fetch(gpu_r.data());
  cpu_relextrema_1d(length, order, clip, x.data(), cpu_r.data());

  int error = 0;
  for (int i = 0; i < length; i++) {
    if (cpu_r[i] != gpu_r[i]) {
      error = 1;
      break;
    }
  }
  if (error) std::printf("1D test: FAILED\n");
  return time;
}

template <typename T>
static long test_2d(iris::Platform&,
                    int length_x,
                    int length_y,
                    int order,
                    bool clip,
                    int axis,
                    int repeat,
                    const char* type)
{
  const int length = length_x * length_y;
  std::vector<T> x(length);
  for (int i = 0; i < length; i++) x[i] = static_cast<T>(std::rand() % length);
  std::vector<unsigned char> cpu_r(length);
  std::vector<unsigned char> gpu_r(length);

  iris::DMem mem_x(x.data(), x.size() * sizeof(T));
  iris::DMem mem_result(gpu_r.data(), gpu_r.size() * sizeof(unsigned char));
  int clip_i = clip ? 1 : 0;
  const std::string name = kernel_name("relextrema_2d", type_suffix<T>());
  size_t gws[2] = {((static_cast<size_t>(length_x) + 15) / 16) * 16,
                   ((static_cast<size_t>(length_y) + 15) / 16) * 16};
  size_t lws[2] = {16, 16};

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    void* params[] = {&length_x, &length_y, &order, &clip_i, &axis, &mem_x, &mem_result};
    int pinfo[] = {sizeof(length_x), sizeof(length_y), sizeof(order), sizeof(clip_i),
                   sizeof(axis), iris_r, iris_w};
    iris::Task task;
    task.kernel(name.c_str(), 2, nullptr, gws, lws, 7, params, pinfo);
    task.flush_out(mem_result);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average 2D kernel (type = %s, order = %d, clip = %d, axis = %d) execution time %f (s)\n",
              type, order, clip, axis, (time * 1e-9f) / repeat);

  mem_result.fetch(gpu_r.data());
  cpu_relextrema_2d(length_x, length_y, order, clip, axis, x.data(), cpu_r.data());

  int error = 0;
  for (int i = 0; i < length; i++) {
    if (cpu_r[i] != gpu_r[i]) {
      error = 1;
      break;
    }
  }
  if (error) std::printf("2D test: FAILED\n");
  return time;
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::printf("Usage ./%s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = std::atoi(argv[1]);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  long time = 0;
  for (int order = 1; order <= 128; order *= 2) {
    time += test_1d<int>(platform, 1000000, order, true, repeat, "int");
    time += test_1d<long>(platform, 1000000, order, true, repeat, "long");
    time += test_1d<float>(platform, 1000000, order, true, repeat, "float");
    time += test_1d<double>(platform, 1000000, order, true, repeat, "double");
  }
  for (int order = 1; order <= 128; order *= 2) {
    time += test_2d<int>(platform, 1000, 1000, order, true, 1, repeat, "int");
    time += test_2d<long>(platform, 1000, 1000, order, true, 1, repeat, "long");
    time += test_2d<float>(platform, 1000, 1000, order, true, 1, repeat, "float");
    time += test_2d<double>(platform, 1000, 1000, order, true, 1, repeat, "double");
  }
  for (int order = 1; order <= 128; order *= 2) {
    time += test_2d<int>(platform, 1000, 1000, order, true, 0, repeat, "int");
    time += test_2d<long>(platform, 1000, 1000, order, true, 0, repeat, "long");
    time += test_2d<float>(platform, 1000, 1000, order, true, 0, repeat, "float");
    time += test_2d<double>(platform, 1000, 1000, order, true, 0, repeat, "double");
  }

  std::printf("\n-----------------------------------------------\n");
  std::printf("Total kernel execution time: %lf (s)", time * 1e-9);
  std::printf("\n-----------------------------------------------\n");

  platform.finalize();
  return 0;
}

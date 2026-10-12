#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

using host_half = _Float16;

static inline int divup(int count, int size) {
  return (count + size - 1) / size;
}

static void ReluGrad_reference(
    int count,
    const host_half* gradient,
    const host_half* feature,
    host_half* backprop)
{
  for (int i = 0; i < count; i++) {
    float grad_f = static_cast<float>(gradient[i]);
    float feature_f = static_cast<float>(feature[i]);
    float backprop_f = (feature_f > 0) ? grad_f : 0;
    backprop[i] = static_cast<host_half>(backprop_f);
  }
}

static void Relu_reference(int count, const int* input, int* output)
{
  for (int i = 0; i < count; i++) {
    signed char c1, c2, c3, c4;
    unsigned x, y, z, w;
    c1 = input[i] & 0xFF;
    c2 = (input[i] >> 8) & 0xFF;
    c3 = (input[i] >> 16) & 0xFF;
    c4 = (input[i] >> 24) & 0xFF;
    x = c1 > 0 ? c1 : 0;
    y = c2 > 0 ? c2 : 0;
    z = c3 > 0 ? c3 : 0;
    w = c4 > 0 ? c4 : 0;
    output[i] = w << 24 | z << 16 | y << 8 | x;
  }
}

static void run_kernel_1d(
    const char* name,
    int nparams,
    void** params,
    int* pinfo,
    size_t gws,
    size_t lws,
    iris::DMem* flush)
{
  iris::Task task;
  size_t global[1] = {gws};
  size_t local[1] = {lws};
  task.kernel(name, 1, nullptr, global, local, nparams, params, pinfo);
  if (flush) task.flush_out(*flush);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <count> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int count = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  constexpr int kThreadInBlock = 256;
  constexpr int VectorSize = 8;

  size_t size = count * sizeof(host_half);
  host_half *h_gradient = static_cast<host_half*>(std::malloc(size));
  host_half *h_feature = static_cast<host_half*>(std::malloc(size));
  host_half *h_backprop = static_cast<host_half*>(std::malloc(size));
  host_half *r_backprop = static_cast<host_half*>(std::malloc(size));
  if (!h_gradient || !h_feature || !h_backprop || !r_backprop) {
    std::fprintf(stderr, "host half allocation failed\n");
    return 1;
  }

  std::mt19937 engine(19937);
  std::uniform_real_distribution<float> real_dist(-1.f, 1.f);
  for (int i = 0; i < count; i++) {
    h_feature[i] = static_cast<host_half>(real_dist(engine));
    h_gradient[i] = static_cast<host_half>(1.f);
  }

  ReluGrad_reference(count, h_gradient, h_feature, r_backprop);

  iris::DMem mem_gradient(h_gradient, size);
  iris::DMem mem_feature(h_feature, size);
  iris::DMem mem_backprop(h_backprop, size);

  std::printf("16-byte aligned pointers: Yes\n");

  int half_count = divup(count, 2);
  int kBlock = divup(half_count, kThreadInBlock);
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    void* params[4] = {&mem_gradient, &mem_feature, &mem_backprop, const_cast<int*>(&count)};
    int pinfo[4] = {iris_r, iris_r, iris_w, sizeof(count)};
    run_kernel_1d("ReluGrad_impl1", 4, params, pinfo,
                  static_cast<size_t>(kBlock * kThreadInBlock), kThreadInBlock,
                  i == repeat - 1 ? &mem_backprop : nullptr);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of ReluGrad_impl1 Kernel: %f (us)\n",
              (time * 1e-3f) / repeat);
  mem_backprop.fetch(h_backprop);

  int fail = 0;
  for (int i = 0; i < count; i++) {
    if (std::fabs(static_cast<float>(h_backprop[i]) -
                  static_cast<float>(r_backprop[i])) > 1e-3f) {
      fail = 1;
      break;
    }
  }
  std::printf("%s\n", fail ? "FAIL" : "PASS");

  half_count = divup(count, VectorSize);
  kBlock = divup(half_count, kThreadInBlock);
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    void* params[4] = {&mem_gradient, &mem_feature, &mem_backprop, const_cast<int*>(&count)};
    int pinfo[4] = {iris_r, iris_r, iris_w, sizeof(count)};
    run_kernel_1d("ReluGrad_impl2", 4, params, pinfo,
                  static_cast<size_t>(kBlock * kThreadInBlock), kThreadInBlock,
                  i == repeat - 1 ? &mem_backprop : nullptr);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of ReluGrad_impl2 Kernel: %f (us)\n",
              (time * 1e-3f) / repeat);
  mem_backprop.fetch(h_backprop);

  fail = 0;
  for (int i = 0; i < count; i++) {
    if (std::fabs(static_cast<float>(h_backprop[i]) -
                  static_cast<float>(r_backprop[i])) > 1e-3f) {
      fail = 1;
      break;
    }
  }
  std::printf("%s\n", fail ? "FAIL" : "PASS");

  size = count * sizeof(int);
  int *h_in = static_cast<int*>(std::malloc(size));
  int *h_out = static_cast<int*>(std::malloc(size));
  int *r_out = static_cast<int*>(std::malloc(size));
  if (!h_in || !h_out || !r_out) {
    std::fprintf(stderr, "host int allocation failed\n");
    return 1;
  }

  std::uniform_int_distribution<unsigned char> int_dist(0, 255);
  for (int i = 0; i < count; i++) {
    h_in[i] = (unsigned) int_dist(engine) |
              (unsigned) int_dist(engine) << 8 |
              (unsigned) int_dist(engine) << 16 |
              (unsigned) int_dist(engine) << 24;
  }
  Relu_reference(count, h_in, r_out);

  iris::DMem mem_in(h_in, size);
  iris::DMem mem_out(h_out, size);
  kBlock = divup(count, kThreadInBlock);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    void* params[3] = {const_cast<int*>(&count), &mem_in, &mem_out};
    int pinfo[3] = {sizeof(count), iris_r, iris_w};
    run_kernel_1d("Relu_impl1", 3, params, pinfo,
                  static_cast<size_t>(kBlock * kThreadInBlock), kThreadInBlock,
                  i == repeat - 1 ? &mem_out : nullptr);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of Relu_impl1 Kernel : %f (us)\n",
              (time * 1e-3f) / repeat);
  mem_out.fetch(h_out);

  fail = std::memcmp(h_out, r_out, size);
  std::printf("%s\n", fail ? "FAIL" : "PASS");

  std::free(h_in);
  std::free(h_out);
  std::free(r_out);
  std::free(h_gradient);
  std::free(h_feature);
  std::free(h_backprop);
  std::free(r_backprop);
  platform.finalize();
  return 0;
}

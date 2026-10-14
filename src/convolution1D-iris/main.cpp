#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#define MAX_MASK_WIDTH 10
#define MAX_BLOCK_SIZE 1024

template <typename T>
static bool reference(const T* h_in, const T* d_out, const T* mask,
                      int input_width, int mask_width) {
  bool ok = true;
  for (int i = 0; i < input_width; i++) {
    T s = 0;
    int start = i - mask_width / 2;
    for (int j = 0; j < mask_width; j++) {
      if (start + j >= 0 && start + j < input_width) {
        s += h_in[start + j] * mask[j];
      }
    }
    if (std::fabs(static_cast<double>(s - d_out[i])) > 1e-3) {
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok;
}

template <typename T>
static void fill_input(T* a, int input_width) {
  std::srand(123);
  for (int i = 0; i < input_width; i++) {
    a[i] = static_cast<T>(std::rand() % 256);
  }
}

template <typename T>
static bool run_kernel_set(const char* label, const char* kernel_name,
                           iris::DMem& mem_in, iris::DMem& mem_out,
                           iris::DMem& mem_mask, T* h_in, T* h_out,
                           T* h_mask, int input_width, int mask_width,
                           int repeat) {
  bool all_ok = true;
  for (int bs = 64; bs <= MAX_BLOCK_SIZE; bs *= 2) {
    const size_t gws[1] = {static_cast<size_t>(input_width)};
    const size_t lws[1] = {static_cast<size_t>(bs)};
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      int n = input_width;
      int mw = mask_width;
      iris::Task task;
      void* params[] = {&mem_in, &mem_out, &n, &mw, &mem_mask};
      int pinfo[] = {iris_r, iris_w, sizeof(n), sizeof(mw), iris_r};
      task.kernel(kernel_name, 1, nullptr, const_cast<size_t*>(gws),
                  const_cast<size_t*>(lws), 5, params, pinfo);
      if (i == repeat - 1) task.flush_out(mem_out);
      task.submit(iris_gpu, nullptr, 1);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("Average kernel execution time of %s kernel (block size %d): %f (us)\n",
                label, bs, (time * 1e-3f) / repeat);
    all_ok = reference(h_in, h_out, h_mask, input_width, mask_width) && all_ok;
  }
  return all_ok;
}

template <typename T>
static bool conv1D(int input_width, int mask_width, int repeat, const char* kernel_name) {
  const size_t size_bytes = static_cast<size_t>(input_width) * sizeof(T);
  T* a = static_cast<T*>(std::malloc(size_bytes));
  T* b = static_cast<T*>(std::malloc(size_bytes));
  if (a == nullptr || b == nullptr) {
    std::free(a);
    std::free(b);
    std::fprintf(stderr, "Host allocation failed\n");
    return false;
  }

  T h_mask[MAX_MASK_WIDTH];
  for (int i = 0; i < MAX_MASK_WIDTH; i++) h_mask[i] = static_cast<T>(1);
  fill_input(a, input_width);

  iris::DMem mem_in(a, size_bytes);
  iris::DMem mem_out(b, size_bytes);
  iris::DMem mem_mask(h_mask, mask_width * sizeof(T));
  mem_in.update(a);
  mem_mask.update(h_mask);

  bool ok = true;
  ok = run_kernel_set("conv1d", kernel_name, mem_in, mem_out, mem_mask, a, b,
                      h_mask, input_width, mask_width, repeat) && ok;
  std::string tiled = std::string(kernel_name) + "_tiled";
  std::string caching = std::string(kernel_name) + "_caching";
  ok = run_kernel_set("conv1d-tiled", tiled.c_str(), mem_in, mem_out, mem_mask, a, b,
                      h_mask, input_width, mask_width, repeat) && ok;
  ok = run_kernel_set("conv1d-tiled-caching", caching.c_str(), mem_in, mem_out, mem_mask,
                      a, b, h_mask, input_width, mask_width, repeat) && ok;

  std::free(a);
  std::free(b);
  return ok;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::printf("Usage: %s <input_width> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int input_width = std::atoi(argv[1]);
  input_width = (input_width + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE * MAX_BLOCK_SIZE;
  const int repeat = std::atoi(argv[2]);
  bool ok = true;

  for (int mask_width = 3; mask_width < MAX_MASK_WIDTH; mask_width += 2) {
    std::printf("\n---------------------\n");
    std::printf("Mask width: %d\n", mask_width);

    std::printf("1D convolution (FP64)\n");
    ok = conv1D<double>(input_width, mask_width, repeat, "conv1d_f64") && ok;

    std::printf("1D convolution (FP32)\n");
    ok = conv1D<float>(input_width, mask_width, repeat, "conv1d_f32") && ok;

    std::printf("1D convolution (INT16)\n");
    ok = conv1D<int16_t>(input_width, mask_width, repeat, "conv1d_i16") && ok;
  }

  platform.finalize();
  return ok ? 0 : 1;
}

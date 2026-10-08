#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define NUM_THREADS 256
#define GridDimMaxY 65536

template <typename T>
static void ChannelShuffleNCHWKernel_cpu(const int N,
                                         const int G,
                                         const int K,
                                         const int HxW,
                                         const T* X,
                                         T* Y)
{
  const int C = G * K;
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int s = 0; s < HxW; s++) {
        Y[(n * C + c) * HxW + s] =
            X[(n * C + (c % G) * K + c / G) * HxW + s];
      }
    }
  }
}

template <typename T>
static void ChannelShuffleNHWCKernel_cpu(const int O,
                                         const int G,
                                         const int K,
                                         const T* X,
                                         T* Y)
{
  const int C = G * K;
  for (int o = 0; o < O; o++) {
    for (int i = 0; i < C; i++) {
      Y[o * C + i] = X[o * C + (i % G) * K + i / G];
    }
  }
}

template <typename T>
static bool ChannelShuffleNCHW_cpu(T* X, int N, int C, int G, int numel, T* Y)
{
  if (C % G != 0 || numel < N * C) return false;
  const int K = C / G;
  const int HxW = numel / (N * C);
  ChannelShuffleNCHWKernel_cpu<T>(N, G, K, HxW, X, Y);
  return true;
}

template <typename T>
static bool ChannelShuffleNHWC_cpu(T* X, int N, int C, int G, int numel, T* Y)
{
  if (C % G != 0 || numel < N * C) return false;
  const int K = C / G;
  const int HxW = numel / (N * C);
  const int O = N * HxW;
  ChannelShuffleNHWCKernel_cpu<T>(O, G, K, X, Y);
  return true;
}

static const char* nhwc_kernel_name(int C)
{
  if (C <= 32) return "channel_shuffle_nhwc_32";
  if (C <= 128) return "channel_shuffle_nhwc_128";
  if (C <= 512) return "channel_shuffle_nhwc_512";
  return nullptr;
}

static bool ChannelShuffleNHWC(iris::DMem& mem_x,
                               iris::DMem& mem_y,
                               int N,
                               int C,
                               int G,
                               int numel,
                               long& time,
                               int repeat)
{
  if (C % G != 0 || numel < N * C) return false;
  int K = C / G;
  int HxW = numel / (N * C);
  int outer_size = N * HxW;
  const char* kernel = nhwc_kernel_name(C);
  if (!kernel) return false;

  const size_t gws[1] = {static_cast<size_t>(outer_size) * NUM_THREADS};
  const size_t lws[1] = {NUM_THREADS};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[4] = {&mem_y, &mem_x, &G, &K};
    int pinfo[4] = {iris_w, iris_r, sizeof(G), sizeof(K)};
    task.kernel(kernel, 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 4, params, pinfo);
    task.flush_out(mem_y);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return true;
}

static bool ChannelShuffleNCHW(iris::DMem& mem_x,
                               iris::DMem& mem_y,
                               int N,
                               int C,
                               int G,
                               int numel,
                               long& time,
                               int repeat)
{
  if (C % G != 0 || numel < N * C) return false;
  int K = C / G;
  int HxW = numel / (N * C);
  int S = (HxW + NUM_THREADS - 1) / NUM_THREADS;

  if (N > GridDimMaxY) return false;
  const size_t gws[3] = {static_cast<size_t>(S) * NUM_THREADS,
                         static_cast<size_t>(N),
                         static_cast<size_t>(C)};
  const size_t lws[3] = {NUM_THREADS, 1, 1};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[5] = {&mem_y, &mem_x, &G, &K, &HxW};
    int pinfo[5] = {iris_w, iris_r, sizeof(G), sizeof(K), sizeof(HxW)};
    task.kernel("channel_shuffle_nchw", 3, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 5, params, pinfo);
    task.flush_out(mem_y);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return true;
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("Usage: %s <group size> <width> <height> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int G = std::atoi(argv[1]);
  const int W = std::atoi(argv[2]);
  const int H = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);

  int failures = 0;
  long time = 0;

  for (int N = 1; N <= 64; N = N * 4) {
    for (int C = 32; C <= 512; C = C * 4) {
      std::printf("\n(N=%d C=%d W=%d H=%d)\n", N, C, W, H);

      const int numel = N * C * W * H;
      const size_t data_size_bytes = static_cast<size_t>(numel) * sizeof(float);

      float* h_X = static_cast<float*>(std::malloc(data_size_bytes));
      float* h_Y = static_cast<float*>(std::malloc(data_size_bytes));
      float* h_Y_ref = static_cast<float*>(std::malloc(data_size_bytes));
      if (!h_X || !h_Y || !h_Y_ref) {
        std::printf("Host memory allocation failed. Exit\n");
        std::free(h_X);
        std::free(h_Y);
        std::free(h_Y_ref);
        platform.finalize();
        return 1;
      }

      for (int i = 0; i < numel; i++) h_X[i] = static_cast<float>(i) / numel;
      std::memset(h_Y, 0, data_size_bytes);
      std::memset(h_Y_ref, 0, data_size_bytes);

      iris::DMem mem_x(h_X, data_size_bytes);
      iris::DMem mem_y(h_Y, data_size_bytes);

      ChannelShuffleNHWC(mem_x, mem_y, N, C, G, numel, time, repeat);
      ChannelShuffleNHWC_cpu(h_X, N, C, G, numel, h_Y_ref);
      mem_y.fetch(h_Y);
      int error = std::memcmp(h_Y, h_Y_ref, data_size_bytes);
      if (error) {
        std::printf("Failed to pass channel shuffle (NHWC) check\n");
        failures++;
      } else {
        std::printf("Average time of channel shuffle (NHWC): %f (ms)\n",
                    (time * 1e-6f) / repeat);
      }

      ChannelShuffleNCHW(mem_x, mem_y, N, C, G, numel, time, repeat);
      ChannelShuffleNCHW_cpu(h_X, N, C, G, numel, h_Y_ref);
      mem_y.fetch(h_Y);
      error = std::memcmp(h_Y, h_Y_ref, data_size_bytes);
      if (error) {
        std::printf("Failed to pass channel shuffle (NCHW) check\n");
        failures++;
      } else {
        std::printf("Average time of channel shuffle (NCHW): %f (ms)\n",
                    (time * 1e-6f) / repeat);
      }

      std::free(h_X);
      std::free(h_Y);
      std::free(h_Y_ref);
    }
  }

  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  platform.finalize();
  return failures == 0 ? 0 : 1;
}

#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define NUM_THREADS 256

typedef int scalar_t;

template <typename T>
static void ref_nchw(const int N,
                     const int C,
                     const int HxW,
                     const T* X,
                     T* sum,
                     T* sumsq)
{
  for (int c = 0; c < C; c++) {
    T m_val = 0;
    T v_val = 0;
    for (int n = 0; n < N; n++) {
      for (int hw = 0; hw < HxW; hw++) {
        const int index = (n * C + c) * HxW + hw;
        m_val += X[index];
        v_val += X[index] * X[index];
      }
    }
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

template <typename T>
static void ref_nhwc(const int N,
                     const int C,
                     const int HxW,
                     const T* X,
                     T* sum,
                     T* sumsq)
{
  for (int c = 0; c < C; c++) {
    T m_val = 0;
    T v_val = 0;
    for (int i = 0; i < N * HxW; i++) {
      const int index = i * C + c;
      m_val += X[index];
      v_val += X[index] * X[index];
    }
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

template <typename T>
static bool check(int size, T* d, T* h)
{
  for (int i = 0; i < size; i++) {
    if (std::abs(d[i] - h[i]) > 1) return false;
  }
  return true;
}

static void compute_channel_sum(const char* kernel_name,
                                iris::DMem& mem_x,
                                iris::DMem& mem_sum,
                                iris::DMem& mem_sumsq,
                                int N,
                                int C,
                                int HxW,
                                long& time,
                                int repeat)
{
  const size_t gws[1] = {static_cast<size_t>(C) * NUM_THREADS};
  const size_t lws[1] = {NUM_THREADS};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[6] = {&mem_x, &mem_sum, &mem_sumsq, &N, &C, &HxW};
    int pinfo[6] = {iris_r, iris_w, iris_w, sizeof(N), sizeof(C), sizeof(HxW)};
    task.kernel(kernel_name, 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 6, params, pinfo);
    task.flush_out(mem_sum);
    task.flush_out(mem_sumsq);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int W = std::atoi(argv[1]);
  const int H = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  long time = 0;
  int failures = 0;

  for (int N = 1; N <= 64; N = N * 4) {
    for (int C = 32; C <= 512; C = C * 4) {
      std::printf("\n(N=%d C=%d W=%d H=%d)\n", N, C, W, H);

      const int HxW = W * H;
      const int numel = N * C * HxW;
      const size_t data_size_bytes = static_cast<size_t>(numel) * sizeof(scalar_t);
      const size_t output_size_bytes = static_cast<size_t>(C) * sizeof(scalar_t);

      scalar_t* h_X = static_cast<scalar_t*>(std::malloc(data_size_bytes));
      scalar_t* h_sum = static_cast<scalar_t*>(std::malloc(output_size_bytes));
      scalar_t* h_sumsq = static_cast<scalar_t*>(std::malloc(output_size_bytes));
      scalar_t* r_sum = static_cast<scalar_t*>(std::malloc(output_size_bytes));
      scalar_t* r_sumsq = static_cast<scalar_t*>(std::malloc(output_size_bytes));
      if (!h_X || !h_sum || !h_sumsq || !r_sum || !r_sumsq) {
        std::printf("Host memory allocation failed. Exit\n");
        std::free(h_X);
        std::free(h_sum);
        std::free(h_sumsq);
        std::free(r_sum);
        std::free(r_sumsq);
        platform.finalize();
        return 1;
      }

      std::srand(numel);
      for (int i = 0; i < numel; i++) h_X[i] = std::rand() % 256;

      iris::DMem mem_x(h_X, data_size_bytes);
      iris::DMem mem_sum(h_sum, output_size_bytes);
      iris::DMem mem_sumsq(h_sumsq, output_size_bytes);

      compute_channel_sum("channel_sum_nhwc", mem_x, mem_sum, mem_sumsq,
                          N, C, HxW, time, repeat);
      mem_sum.fetch(h_sum);
      ref_nhwc(N, C, HxW, h_X, r_sum, r_sumsq);
      bool ok = check(C, h_sum, r_sum);
      std::printf("Average time of channel sum (nhwc): %f (ms)\n",
                  (time * 1e-6f) / repeat);
      std::printf("Verification %s for channel sum (nhwc)\n", ok ? "PASS" : "FAIL");
      failures += ok ? 0 : 1;

      compute_channel_sum("channel_sum_nchw", mem_x, mem_sum, mem_sumsq,
                          N, C, HxW, time, repeat);
      mem_sum.fetch(h_sum);
      ref_nchw(N, C, HxW, h_X, r_sum, r_sumsq);
      ok = check(C, h_sum, r_sum);
      std::printf("Average time of channel sum (nchw): %f (ms)\n",
                  (time * 1e-6f) / repeat);
      std::printf("Verification %s for channel sum (nchw)\n", ok ? "PASS" : "FAIL");
      failures += ok ? 0 : 1;

      std::free(h_X);
      std::free(h_sum);
      std::free(h_sumsq);
      std::free(r_sum);
      std::free(r_sumsq);
    }
  }

  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  platform.finalize();
  return failures == 0 ? 0 : 1;
}

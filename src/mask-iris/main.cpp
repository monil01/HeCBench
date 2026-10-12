#include <iris/iris.hpp>

#include "reference.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define GPU_THREADS 256

static void print_mask_ratio(int *h_out, int *out, int fill_val, int data_size)
{
  int error = std::memcmp(h_out, out, data_size * sizeof(int));
  int cnt_fill = 0;
  for (int i = 0; i < data_size; i++) {
    if (h_out[i] == fill_val) cnt_fill++;
  }
  std::printf("%s, Mask ratio: %f\n", (error ? "FAIL" : "PASS"),
              static_cast<float>(cnt_fill) / data_size);
}

static void run_kernel(
    const char* name,
    int nparams,
    void** params,
    int* pinfo,
    int nblocks,
    int repeat,
    iris::DMem& mem_out)
{
  size_t gws[1] = {static_cast<size_t>(nblocks * GPU_THREADS)};
  size_t lws[1] = {GPU_THREADS};
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    task.kernel(name, 1, nullptr, gws, lws, nparams, params, pinfo);
    if (i == repeat - 1) task.flush_out(mem_out);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of %s kernel: %f (us)\n",
              name, (time * 1e-3f) / repeat);
}

static void eval_mask(const int M, const int N, const int B, const int repeat)
{
  const int fill_val = -1;
  const int radius = M / 4;
  int batch_dim = (B <= 0) ? 1 : B;

  std::printf("\nM = %d, N = %d, B = %d\n", M, N, batch_dim);

  int data_size = N * M * batch_dim;
  size_t data_size_in_bytes = data_size * sizeof(int);
  size_t window_size_in_bytes = N * sizeof(int);
  size_t seq_len_in_bytes = N * sizeof(int);

  int *h_in = static_cast<int*>(std::malloc(data_size_in_bytes));
  int *h_out = static_cast<int*>(std::malloc(data_size_in_bytes));
  int *out = static_cast<int*>(std::malloc(data_size_in_bytes));
  int *h_seq_len = static_cast<int*>(std::malloc(seq_len_in_bytes));
  int *h_window = static_cast<int*>(std::malloc(window_size_in_bytes));

  srand(123);
  for (int i = 0; i < N; i++) h_seq_len[i] = rand() % (M / 2);
  for (int i = 0; i < N; i++) h_window[i] = rand() % M;
  for (int i = 0; i < data_size; i++) h_in[i] = rand() % (M * N);

  iris::DMem mem_in(h_in, data_size_in_bytes);
  iris::DMem mem_out(out, data_size_in_bytes);
  iris::DMem mem_seq(h_seq_len, seq_len_in_bytes);
  iris::DMem mem_window(h_window, window_size_in_bytes);

  int nblocks = (B <= 0) ? (N * M / GPU_THREADS) : (N * M);
  if (nblocks < 1) nblocks = 1;

  sequenceMaskKernel_cpu(N, M, batch_dim, h_in, h_seq_len, fill_val, h_out);
  {
    void* params[7] = {const_cast<int*>(&N), const_cast<int*>(&M), const_cast<int*>(&batch_dim),
                       &mem_in, &mem_seq, const_cast<int*>(&fill_val), &mem_out};
    int pinfo[7] = {sizeof(N), sizeof(M), sizeof(batch_dim), iris_r, iris_r, sizeof(fill_val), iris_w};
    run_kernel("sequenceMaskKernel_int", 7, params, pinfo, nblocks, repeat, mem_out);
    mem_out.fetch(out);
    print_mask_ratio(h_out, out, fill_val, data_size);
  }

  windowMaskKernel_cpu(N, M, batch_dim, h_in, h_window, radius, fill_val, h_out);
  {
    void* params[8] = {const_cast<int*>(&N), const_cast<int*>(&M), const_cast<int*>(&batch_dim),
                       &mem_in, &mem_window, const_cast<int*>(&radius), const_cast<int*>(&fill_val), &mem_out};
    int pinfo[8] = {sizeof(N), sizeof(M), sizeof(batch_dim), iris_r, iris_r, sizeof(radius), sizeof(fill_val), iris_w};
    run_kernel("windowMaskKernel_int", 8, params, pinfo, nblocks, repeat, mem_out);
    mem_out.fetch(out);
    print_mask_ratio(h_out, out, fill_val, data_size);
  }

#define RUN_SIMPLE(cpu_func, kernel_name) \
  cpu_func(N, M, batch_dim, h_in, fill_val, h_out); \
  { \
    void* params[6] = {const_cast<int*>(&N), const_cast<int*>(&M), const_cast<int*>(&batch_dim), \
                       &mem_in, const_cast<int*>(&fill_val), &mem_out}; \
    int pinfo[6] = {sizeof(N), sizeof(M), sizeof(batch_dim), iris_r, sizeof(fill_val), iris_w}; \
    run_kernel(kernel_name, 6, params, pinfo, nblocks, repeat, mem_out); \
    mem_out.fetch(out); \
    print_mask_ratio(h_out, out, fill_val, data_size); \
  }

  RUN_SIMPLE(upperMaskKernel_cpu, "upperMaskKernel_int")
  RUN_SIMPLE(lowerMaskKernel_cpu, "lowerMaskKernel_int")
  RUN_SIMPLE(upperDiagMaskKernel_cpu, "upperDiagMaskKernel_int")
  RUN_SIMPLE(lowerDiagMaskKernel_cpu, "lowerDiagMaskKernel_int")

#undef RUN_SIMPLE

  std::free(h_in);
  std::free(h_out);
  std::free(out);
  std::free(h_window);
  std::free(h_seq_len);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("Usage: %s <sequence length> <sequence length> <batch size> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int M = std::atoi(argv[1]);
  const int N = std::atoi(argv[2]);
  const int B = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);

  eval_mask(M, N, B, repeat);

  platform.finalize();
  return 0;
}

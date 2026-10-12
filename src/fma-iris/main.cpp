#include <iris/iris.hpp>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

template <typename T>
void reference(const T* a, const T* b, T* c, const int* in_indices,
               const int* out_indices, int num_ops, int C, int N_A, int N_C)
{
  for (int op_idx = 0; op_idx < num_ops; op_idx++) {
    int in_idx = in_indices[op_idx];
    int out_idx = out_indices[op_idx];
    if (in_idx < 0 || in_idx >= N_A || out_idx < 0 || out_idx >= N_C) continue;
    for (int ch_idx = 0; ch_idx < C; ch_idx++) {
      c[out_idx * C + ch_idx] += a[in_idx * C + ch_idx] * b[ch_idx];
    }
  }
}

template <typename T>
const char* kernel_name(const char* mode);

template <>
const char* kernel_name<__half>(const char* mode)
{
  return std::string(mode) == "basic" ? "fma_basic_half" : "fma_rowwise_half";
}

template <>
const char* kernel_name<__hip_bfloat16>(const char* mode)
{
  return std::string(mode) == "basic" ? "fma_basic_bfloat16" : "fma_rowwise_bfloat16";
}

template <>
const char* kernel_name<float>(const char* mode)
{
  return std::string(mode) == "basic" ? "fma_basic_float" : "fma_rowwise_float";
}

template <>
const char* kernel_name<double>(const char* mode)
{
  return std::string(mode) == "basic" ? "fma_basic_double" : "fma_rowwise_double";
}

template <typename T>
void run_kernel(const char* mode, const T* h_a, const T* h_b, T* h_c,
                const int* h_i_idx, const int* h_o_idx, int num_ops,
                int C, int NA, int NC)
{
  const size_t a_size_bytes = static_cast<size_t>(NA) * C * sizeof(T);
  const size_t b_size_bytes = static_cast<size_t>(C) * sizeof(T);
  const size_t c_size_bytes = static_cast<size_t>(NC) * C * sizeof(T);
  const size_t idx_size_bytes = static_cast<size_t>(num_ops) * sizeof(int);

  iris::DMem d_a(const_cast<T*>(h_a), a_size_bytes);
  iris::DMem d_b(const_cast<T*>(h_b), b_size_bytes);
  iris::DMem d_c(h_c, c_size_bytes);
  iris::DMem d_i_idx(const_cast<int*>(h_i_idx), idx_size_bytes);
  iris::DMem d_o_idx(const_cast<int*>(h_o_idx), idx_size_bytes);

  constexpr int threads_per_block = 256;
  const int total_threads = std::string(mode) == "rowwise" ? num_ops : num_ops * C;
  const size_t gws[1] = {static_cast<size_t>(((total_threads + threads_per_block - 1) /
                                             threads_per_block) * threads_per_block)};
  const size_t lws[1] = {threads_per_block};

  void* params[9] = {&d_a, &d_b, &d_c, &d_i_idx, &d_o_idx, &num_ops, &C, &NA, &NC};
  int pinfo[9] = {iris_r, iris_r, iris_rw, iris_r, iris_r,
                  sizeof(num_ops), sizeof(C), sizeof(NA), sizeof(NC)};
  iris::Task task;
  task.kernel(kernel_name<T>(mode), 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws),
              9, params, pinfo);
  task.flush_out(d_c);
  task.submit(iris_gpu, nullptr, 1);
  d_c.fetch(h_c);
}

template <typename T>
void fma(int NA, int NC, int C, int num_ops, int repeat)
{
  size_t a_size = static_cast<size_t>(NA) * C;
  size_t b_size = C;
  size_t c_size = static_cast<size_t>(NC) * C;
  size_t a_size_bytes = a_size * sizeof(T);
  size_t b_size_bytes = b_size * sizeof(T);
  size_t c_size_bytes = c_size * sizeof(T);
  size_t idx_size_bytes = static_cast<size_t>(num_ops) * sizeof(int);

  T* h_a = static_cast<T*>(malloc(a_size_bytes));
  T* h_b = static_cast<T*>(malloc(b_size_bytes));
  T* h_c = static_cast<T*>(malloc(c_size_bytes));
  int* h_i_idx = static_cast<int*>(malloc(idx_size_bytes));
  int* h_o_idx = static_cast<int*>(malloc(idx_size_bytes));
  T* h_r = static_cast<T*>(malloc(c_size_bytes));

  srand(123);
  for (size_t i = 0; i < a_size; i++) h_a[i] = static_cast<T>(rand() % 5);
  for (size_t i = 0; i < b_size; i++) h_b[i] = static_cast<T>(rand() % 3 - 1);
  for (int i = 0; i < num_ops; i++) h_i_idx[i] = rand() % NA;
  for (int i = 0; i < num_ops; i++) h_o_idx[i] = i;

  memset(h_r, 0, c_size_bytes);
  reference<T>(h_a, h_b, h_r, h_i_idx, h_o_idx, num_ops, C, NA, NC);

  bool ok = true;
  memset(h_c, 0, c_size_bytes);
  run_kernel<T>("basic", h_a, h_b, h_c, h_i_idx, h_o_idx, num_ops, C, NA, NC);
  for (size_t i = 0; i < c_size; i++) {
    if ((double)h_c[i] - (double)h_r[i] > 1e-3) ok = false;
  }

  memset(h_c, 0, c_size_bytes);
  run_kernel<T>("rowwise", h_a, h_b, h_c, h_i_idx, h_o_idx, num_ops, C, NA, NC);
  for (size_t i = 0; i < c_size; i++) {
    if ((double)h_c[i] - (double)h_r[i] > 1e-3) ok = false;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  double time = 0;
  for (int i = 0; i < repeat; i++) {
    memset(h_c, 0, c_size_bytes);
    auto start = std::chrono::steady_clock::now();
    run_kernel<T>("basic", h_a, h_b, h_c, h_i_idx, h_o_idx, num_ops, C, NA, NC);
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  printf("Average execution time of basic kernel: %f (us)\n", (time * 1e-3f) / repeat);

  time = 0;
  for (int i = 0; i < repeat; i++) {
    memset(h_c, 0, c_size_bytes);
    auto start = std::chrono::steady_clock::now();
    run_kernel<T>("rowwise", h_a, h_b, h_c, h_i_idx, h_o_idx, num_ops, C, NA, NC);
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  printf("Average execution time of rowwise kernel: %f (us)\n", (time * 1e-3f) / repeat);

  free(h_a);
  free(h_b);
  free(h_c);
  free(h_i_idx);
  free(h_o_idx);
  free(h_r);
}

int main(int argc, char* argv[])
{
  if (argc != 6) {
    printf("Usage: %s <NA> <NC> <C> <num_ops> <repeat>\n", argv[0]);
    printf("<Number of rows in A> <Number of rows in C> ");
    printf("<Number of channels/columns> <Number of operations>\n");
    return 1;
  }
  const int NA = atoi(argv[1]);
  const int NC = atoi(argv[2]);
  const int C = atoi(argv[3]);
  const int num_ops = atoi(argv[4]);
  const int repeat = atoi(argv[5]);

  if (num_ops > NC) {
    printf("Error: Number of operations is larger than number of rows in C\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  for (int i = 0; i < 2; i++) {
    printf("FP16 FMA\n");
    fma<__half>(NA, NC, C, num_ops, repeat);
    printf("BF16 FMA\n");
    fma<__hip_bfloat16>(NA, NC, C, num_ops, repeat);
    printf("FP32 FMA\n");
    fma<float>(NA, NC, C, num_ops, repeat);
    printf("FP64 FMA\n");
    fma<double>(NA, NC, C, num_ops, repeat);
  }

  platform.finalize();
  return 0;
}

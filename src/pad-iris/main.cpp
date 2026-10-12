#include <iris/iris.hpp>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

using T = double;

struct Params {
  int n_gpu_threads = 256;
  int n_gpu_blocks = 8;
  int n_threads = 4;
  int n_warmup = 5;
  int n_reps = 1000;
  float alpha = 0.1f;
  int m = 1000;
  int n = 999;
  int pad = 1;

  Params(int argc, char** argv) {
    int opt;
    while ((opt = getopt(argc, argv, "hi:g:t:w:r:a:m:n:e:")) >= 0) {
      switch (opt) {
        case 'h':
          usage();
          std::exit(0);
        case 'i':
          n_gpu_threads = std::atoi(optarg);
          break;
        case 'g':
          n_gpu_blocks = std::atoi(optarg);
          break;
        case 't':
          n_threads = std::atoi(optarg);
          break;
        case 'w':
          n_warmup = std::atoi(optarg);
          break;
        case 'r':
          n_reps = std::atoi(optarg);
          break;
        case 'a':
          alpha = std::atof(optarg);
          break;
        case 'm':
          m = std::atoi(optarg);
          break;
        case 'n':
          n = std::atoi(optarg);
          break;
        case 'e':
          pad = std::atoi(optarg);
          break;
        default:
          usage();
          std::exit(1);
      }
    }
    assert(n_gpu_threads > 0);
    assert(n_gpu_blocks > 0);
    assert(n_threads > 0);
    assert(alpha >= 0.0f && alpha <= 1.0f);
  }

  static void usage() {
    std::fprintf(stderr,
                 "\nUsage: ./main [options]\n"
                 "  -i <I> device threads per block (default=256)\n"
                 "  -g <G> device blocks (default=8)\n"
                 "  -t <T> host threads, accepted for compatibility (default=4)\n"
                 "  -w <W> warmup iterations (default=5)\n"
                 "  -r <R> timed repetitions (default=1000)\n"
                 "  -a <A> host fraction, accepted for compatibility (default=0.1)\n"
                 "  -m <M> rows (default=1000)\n"
                 "  -n <N> columns (default=999)\n"
                 "  -e <B> extra columns to pad (default=1)\n");
  }
};

static void read_input(T* input, const Params& p) {
  std::srand(123);
  for (int i = 0; i < p.m; i++) {
    for (int j = 0; j < p.n; j++) {
      input[i * p.n + j] = static_cast<T>(std::rand()) / RAND_MAX;
    }
  }
  for (int i = p.n * p.m; i < (p.n + p.pad) * p.m; i++) {
    input[i] = 0.0;
  }
}

static void cpu_padding(T* matrix, int x_size, int y_size, int padded_x_size) {
  for (int y = y_size - 1; y >= 0; --y) {
    for (int x = padded_x_size - 1; x >= 0; --x) {
      if (x < x_size) {
        matrix[y * padded_x_size + x] = matrix[y * x_size + x];
      } else {
        matrix[y * padded_x_size + x] = 0.0;
      }
    }
  }
}

static bool verify(const T* out, T* backup, int n, int m, int n_pad) {
  cpu_padding(backup, n, m, n_pad);
  double sum_delta = 0.0;
  double sum_ref = 0.0;
  for (int i = 0; i < m * n_pad; i++) {
    sum_delta += std::fabs(out[i] - backup[i]);
    sum_ref += std::fabs(backup[i]);
  }
  double l1 = sum_ref == 0.0 ? sum_delta : sum_delta / sum_ref;
  bool ok = l1 < 1e-6;
  std::printf("%s\n", ok ? "Test Passed" : "Test failed");
  return ok;
}

int main(int argc, char** argv) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const Params p(argc, argv);
  const int out_cols = p.n + p.pad;
  const int in_size = p.m * out_cols;
  const size_t bytes = static_cast<size_t>(in_size) * sizeof(T);

  T* h_in_out = static_cast<T*>(std::malloc(bytes));
  T* h_backup = static_cast<T*>(std::malloc(bytes));
  if (!h_in_out || !h_backup) return 1;

  read_input(h_in_out, p);
  std::memcpy(h_backup, h_in_out, bytes);

  size_t gws[1] = {static_cast<size_t>(((in_size + p.n_gpu_threads - 1) / p.n_gpu_threads) *
                                       p.n_gpu_threads)};
  size_t lws[1] = {static_cast<size_t>(p.n_gpu_threads)};

  auto start = std::chrono::steady_clock::now();
  for (int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
    std::memcpy(h_in_out, h_backup, bytes);
    iris::DMem mem_input(h_backup, bytes);
    iris::DMem mem_output(h_in_out, bytes);
    int n = p.n;
    int m = p.m;
    int pad = p.pad;
    int matrix_size = in_size;
    void* params[] = {&mem_output, &mem_input, &n, &m, &pad, &matrix_size};
    int pinfo[] = {iris_w, iris_r, sizeof(n), sizeof(m), sizeof(pad), sizeof(matrix_size)};
    iris::Task task;
    task.kernel("padding_kernel", 1, nullptr, gws, lws, 6, params, pinfo);
    if (rep == p.n_warmup + p.n_reps - 1) task.flush_out(mem_output);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Total padding execution time for %d iterations: %f (ms)\n",
              p.n_reps + p.n_warmup, time * 1e-6f);

  bool ok = verify(h_in_out, h_backup, p.n, p.m, out_cols);
  std::free(h_in_out);
  std::free(h_backup);
  platform.finalize();
  return ok ? 0 : 1;
}

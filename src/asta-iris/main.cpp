#include <iris/iris.hpp>

#include <assert.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#ifdef DOUBLE_PRECISION
using T = double;
#else
using T = float;
#endif

static int divceil(int n, int m) { return (n - 1) / m + 1; }

struct Params {
  int device = 0;
  int n_gpu_threads = 64;
  int n_gpu_blocks = 16;
  int n_warmup = 10;
  int n_reps = 100;
  int m = 197;
  int n = 35588;
  int s = 32;

  Params(int argc, char **argv) {
    int opt;
    while ((opt = getopt(argc, argv, "hd:i:g:w:r:m:n:s:")) >= 0) {
      switch (opt) {
      case 'h':
        usage();
        std::exit(0);
      case 'i': n_gpu_threads = std::atoi(optarg); break;
      case 'g': n_gpu_blocks = std::atoi(optarg); break;
      case 'w': n_warmup = std::atoi(optarg); break;
      case 'r': n_reps = std::atoi(optarg); break;
      case 'm': m = std::atoi(optarg); break;
      case 'n': n = std::atoi(optarg); break;
      case 's': s = std::atoi(optarg); break;
      default:
        std::fprintf(stderr, "\nUnrecognized option!\n");
        usage();
        std::exit(0);
      }
    }
  }

  void usage() const {
    std::fprintf(stderr,
                 "\nUsage: ./main [options]"
                 "\n    -i <I> device threads per block (default=64)"
                 "\n    -g <G> device blocks (default=16)"
                 "\n    -w <W> warmup iterations (default=10)"
                 "\n    -r <R> timed repetitions (default=100)"
                 "\n    -m <M> matrix height (default=197)"
                 "\n    -n <N> matrix width (default=35588)"
                 "\n    -s <S> super-element size (default=32)\n");
  }
};

static void read_input(T *x, const Params &p) {
  const int tiled_n = divceil(p.n, p.s);
  const int in_size = p.m * tiled_n * p.s;
  std::srand(5432);
  for (int i = 0; i < in_size; ++i) x[i] = static_cast<T>(std::rand() % 100) / 100;
}

static void cpu_soa_asta(T *src, T *dst, int height, int width, int tile_size) {
  if ((height / tile_size) * tile_size != height) return;
  for (int k = 0; k < width; ++k)
    for (int i = 0; i < height / tile_size; ++i)
      for (int j = 0; j < tile_size; ++j)
        dst[i * width * tile_size + k * tile_size + j] =
            src[k * height + i * tile_size + j];
}

static int verify(T *input2, T *input, int height, int width, int tile_size) {
  T *output = static_cast<T *>(std::malloc(static_cast<size_t>(width) * height * sizeof(T)));
  cpu_soa_asta(input, output, height, width, tile_size);
  int fail = 0;
  for (int i = 0; i < height * width; ++i) {
    const T diff = std::fabs(output[i] - input2[i]);
    if ((diff > static_cast<T>(0.00001)) &&
        (diff > static_cast<T>(0.01) * std::fabs(output[i]))) {
      std::printf("Failed at line: %d ref: %f actual: %f diff: %f\n", i,
                  static_cast<double>(output[i]), static_cast<double>(input2[i]),
                  static_cast<double>(diff));
      fail = 1;
      break;
    }
  }
  std::free(output);
  return fail;
}

int main(int argc, char **argv) {
  const Params p(argc, argv);
  const int blocks = p.n_gpu_blocks;
  const int threads = p.n_gpu_threads;
  assert(threads <= 256 && "The thread block size is at most 256");

  const int tiled_n = divceil(p.n, p.s);
  const int in_size = p.m * tiled_n * p.s;
  const int finished_size = p.m * tiled_n;
  const size_t in_bytes = static_cast<size_t>(in_size) * sizeof(T);
  const size_t finished_bytes = static_cast<size_t>(finished_size) * sizeof(int);

  T *h_in_out = static_cast<T *>(std::malloc(in_bytes));
  T *h_in_backup = static_cast<T *>(std::malloc(in_bytes));
  int *h_finished = static_cast<int *>(std::malloc(finished_bytes));
  int *h_head = static_cast<int *>(std::malloc(sizeof(int)));
  read_input(h_in_out, p);
  std::memcpy(h_in_backup, h_in_out, in_bytes);
  std::memset(h_finished, 0, finished_bytes);
  h_head[0] = 0;

  iris::Platform platform;
  platform.init(&argc, &argv, 1);
  double kernel_seconds = 0.0;

  for (int rep = 0; rep < p.n_warmup + p.n_reps; ++rep) {
    std::memcpy(h_in_out, h_in_backup, in_bytes);
    std::memset(h_finished, 0, finished_bytes);
    h_head[0] = 0;

    iris::DMem mem_in_out(h_in_out, in_bytes);
    iris::DMem mem_finished(h_finished, finished_bytes);
    iris::DMem mem_head(h_head, sizeof(int));
    iris::Task task;
    int kernel_m = p.m;
    int kernel_tiled_n = tiled_n;
    int kernel_s = p.s;
    void *params[] = {&kernel_m, &kernel_tiled_n, &kernel_s,
                      &mem_in_out, &mem_finished, &mem_head};
    int pinfo[] = {sizeof(kernel_m), sizeof(kernel_tiled_n), sizeof(kernel_s),
                   iris_rw, iris_rw, iris_rw};
    size_t gws[] = {static_cast<size_t>(blocks * threads)};
    size_t lws[] = {static_cast<size_t>(threads)};
    task.kernel("PTTWAC_soa_asta", 1, nullptr, gws, lws, 6, params, pinfo);
    task.flush_out(mem_in_out);

    const auto start = std::chrono::steady_clock::now();
    task.submit(iris_gpu, nullptr, true);
    const auto end = std::chrono::steady_clock::now();
    if (rep >= p.n_warmup)
      kernel_seconds += std::chrono::duration<double>(end - start).count();
  }

  std::printf("Average kernel execution time %lf (s)\n", kernel_seconds / p.n_reps);
  const int status = verify(h_in_out, h_in_backup, tiled_n * p.s, p.m, p.s);
  std::printf("%s\n", status == 0 ? "PASS" : "FAIL");

  std::free(h_in_out);
  std::free(h_in_backup);
  std::free(h_finished);
  std::free(h_head);
  platform.finalize();
  return status;
}

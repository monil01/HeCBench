#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "reference.h"

#define GPU_NUM_THREADS 256

static void launch_concat(
    iris::DMem& mem_inp1,
    iris::DMem& mem_inp2,
    iris::DMem& mem_outp,
    int sz0,
    int sz2,
    int sz1_1,
    int sz1_2)
{
  const size_t n = static_cast<size_t>(sz0) * sz2 * (sz1_1 + sz1_2);
  const size_t nblock = (n + GPU_NUM_THREADS - 1) / GPU_NUM_THREADS;
  const size_t gws[1] = { nblock * GPU_NUM_THREADS };
  const size_t lws[1] = { GPU_NUM_THREADS };

  iris::Task task;
  void* params[7] = {
    &mem_inp1,
    &mem_inp2,
    &mem_outp,
    &sz0,
    &sz2,
    &sz1_1,
    &sz1_2
  };
  int pinfo[7] = {
    iris_r,
    iris_r,
    iris_w,
    sizeof(sz0),
    sizeof(sz2),
    sizeof(sz1_1),
    sizeof(sz1_2)
  };
  task.kernel("concat", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 7, params, pinfo);
  task.flush_out(mem_outp);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  int errors = 0;

  for (int nhead = 6; nhead <= 48; nhead *= 2) {
    std::srand(nhead);

    const int seq_len = 1024;
    const int batch_size = 8;
    const int hidden_dim = nhead * 128;
    const int head_dim = hidden_dim / nhead;

    const int sl1 = std::rand() % (seq_len - 1) + 1;
    const int sl2 = seq_len - sl1;
    const int beam_size = 8;

    std::printf("\n");
    std::printf("num_head = %d\t", nhead);
    std::printf("seq_len = %d\t", seq_len);
    std::printf("batch_size = %d\t", batch_size);
    std::printf("hidden_dimension = %d\t", hidden_dim);
    std::printf("beam_size = %d\n", beam_size);

    const size_t inp1_size = static_cast<size_t>(batch_size) * beam_size * hidden_dim * sl1;
    const size_t inp2_size = static_cast<size_t>(batch_size) * beam_size * hidden_dim * sl2;
    const size_t outp_size = static_cast<size_t>(batch_size) * beam_size * hidden_dim * seq_len;

    const size_t inp1_size_bytes = inp1_size * sizeof(float);
    const size_t inp2_size_bytes = inp2_size * sizeof(float);
    const size_t outp_size_bytes = outp_size * sizeof(float);

    const float size_bytes = 2 * outp_size_bytes * 1e-9f;
    std::printf("Total device memory usage (GB) = %.2f\n", size_bytes);

    float* inp1 = static_cast<float*>(std::malloc(inp1_size_bytes));
    float* inp2 = static_cast<float*>(std::malloc(inp2_size_bytes));
    float* outp = static_cast<float*>(std::malloc(outp_size_bytes));
    float* outp_ref = static_cast<float*>(std::malloc(outp_size_bytes));
    if (!inp1 || !inp2 || !outp || !outp_ref) {
      std::fprintf(stderr, "host allocation failed\n");
      std::free(inp1);
      std::free(inp2);
      std::free(outp);
      std::free(outp_ref);
      platform.finalize();
      return 1;
    }

    for (size_t i = 0; i < inp1_size; i++) {
      inp1[i] = static_cast<float>(std::rand() % inp1_size);
    }

    for (size_t i = 0; i < inp2_size; i++) {
      inp2[i] = static_cast<float>(std::rand() % inp2_size);
    }

    iris::DMem mem_inp1(inp1, inp1_size_bytes);
    iris::DMem mem_inp2(inp2, inp2_size_bytes);
    iris::DMem mem_outp(outp, outp_size_bytes);

    const int sz0 = batch_size * beam_size * nhead;
    const int sz2 = head_dim;

    launch_concat(mem_inp1, mem_inp2, mem_outp, sz0, sz2, sl1, sl2);

    concat_cpu(inp1, inp2, outp_ref, sz0, sz2, sl1, sl2);

    mem_outp.fetch(outp);
    const int error = std::memcmp(outp_ref, outp, outp_size_bytes);
    std::printf("%s\n", error ? "FAIL" : "PASS");
    errors += error ? 1 : 0;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      launch_concat(mem_inp1, mem_inp2, mem_outp, sz0, sz2, sl1, sl2);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    const float avg_time = (time * 1e-3f) / repeat;
    std::printf("Average kernel execution time: %f (us)\n", avg_time);
    std::printf("Average kernel throughput : %f (GB/s)\n", size_bytes / (avg_time * 1e-6f));

    std::free(inp1);
    std::free(inp2);
    std::free(outp);
    std::free(outp_ref);
  }

  platform.finalize();
  return errors == 0 ? 0 : 1;
}

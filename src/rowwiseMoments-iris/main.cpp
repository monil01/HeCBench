#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[])
{
  if (argc != 7) {
    std::printf("Usage: %s <batch> <channel> <width> <height> <group> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int N = std::atoi(argv[1]);
  const int C = std::atoi(argv[2]);
  const int W = std::atoi(argv[3]);
  const int H = std::atoi(argv[4]);
  const int G = std::atoi(argv[5]);
  const int repeat = std::atoi(argv[6]);

  const int64_t D = C / G;
  const float eps = 1e-6f;
  const int64_t row_size = D * H * W;

  const size_t input_size = static_cast<size_t>(N) * C * W * H;
  const size_t output_size = static_cast<size_t>(N) * G;
  const size_t input_size_bytes = input_size * sizeof(float);
  const size_t output_size_bytes = output_size * sizeof(float);

  float* h_X = static_cast<float*>(std::malloc(input_size_bytes));
  float* h_mean = static_cast<float*>(std::malloc(output_size_bytes));
  float* h_rstd = static_cast<float*>(std::malloc(output_size_bytes));
  if (!h_X || !h_mean || !h_rstd) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(h_X);
    std::free(h_mean);
    std::free(h_rstd);
    platform.finalize();
    return 1;
  }

  std::srand(123);
  for (size_t i = 0; i < input_size; i++) {
    h_X[i] = std::rand() / static_cast<float>(RAND_MAX);
  }

  iris::DMem mem_X(h_X, input_size_bytes);
  iris::DMem mem_mean(h_mean, output_size_bytes);
  iris::DMem mem_rstd(h_rstd, output_size_bytes);

  size_t gws[1] = { output_size * 256 };
  size_t lws[1] = { 256 };

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[5] = { const_cast<int64_t*>(&row_size), const_cast<float*>(&eps),
                        &mem_X, &mem_mean, &mem_rstd };
    int pinfo[5] = { static_cast<int>(sizeof(row_size)), sizeof(eps), iris_r, iris_w, iris_w };
    task.kernel("RowwiseMomentsKernel", 1, nullptr, gws, lws, 5, params, pinfo);
    if (i == repeat - 1) {
      task.flush_out(mem_mean);
      task.flush_out(mem_rstd);
    }
    task.submit(iris_gpu, nullptr, 1);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of RowwiseMoments kernel: %f (us)\n",
              (time * 1e-3f) / repeat);

  double avg_rstd = 0.0;
  double avg_mean = 0.0;
  for (size_t i = 0; i < output_size; i++) {
    avg_mean += h_mean[i];
    avg_rstd += h_rstd[i];
  }
  avg_rstd /= output_size;
  avg_mean /= output_size;

  std::printf("Checksum: mean = %lf and rstd = %lf\n", avg_mean, avg_rstd);

  std::free(h_X);
  std::free(h_mean);
  std::free(h_rstd);
  platform.finalize();
  return 0;
}

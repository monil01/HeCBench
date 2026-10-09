#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void reference(
    const float* subject,
    const float* avgs,
    const float* stds,
    float* lb_keogh,
    const float* zlower,
    const float* zupper,
    const int M,
    const int N)
{
  for (int indx = 0; indx < N - M + 1; indx++) {
    float residues = 0.0f;
    float avg = avgs[indx];
    float std = stds[indx];

    for (int i = 0; i < M; ++i) {
      float value = (subject[indx + i] - avg) / std;
      float lower = value - zlower[i];
      float upper = value - zupper[i];
      residues += upper * upper * (upper > 0) + lower * lower * (lower < 0);
    }

    lb_keogh[indx] = residues;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: ./%s <query length> <subject length> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int M = std::atoi(argv[1]);
  int N = std::atoi(argv[2]);
  int repeat = std::atoi(argv[3]);
  int out_count = N - M + 1;

  std::printf("Query length = %d\n", M);
  std::printf("Subject length = %d\n", N);

  std::vector<float> subject(N);
  std::vector<float> lower(N);
  std::vector<float> upper(N);
  std::vector<float> lb(out_count);
  std::vector<float> lb_h(out_count);
  std::vector<float> avgs(out_count);
  std::vector<float> stds(out_count);

  std::srand(123);
  for (int i = 0; i < N; ++i) subject[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  for (int i = 0; i < out_count; ++i) avgs[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  for (int i = 0; i < out_count; ++i) stds[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  for (int i = 0; i < M; ++i) upper[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  for (int i = 0; i < M; ++i) lower[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

  iris::DMem d_subject(subject.data(), sizeof(float) * static_cast<size_t>(N));
  iris::DMem d_avgs(avgs.data(), sizeof(float) * static_cast<size_t>(out_count));
  iris::DMem d_stds(stds.data(), sizeof(float) * static_cast<size_t>(out_count));
  iris::DMem d_lb(lb.data(), sizeof(float) * static_cast<size_t>(out_count));
  iris::DMem d_lower(lower.data(), sizeof(float) * static_cast<size_t>(M));
  iris::DMem d_upper(upper.data(), sizeof(float) * static_cast<size_t>(M));

  const int blocks = 256;
  const int grids = (out_count + blocks - 1) / blocks;
  const size_t gws[1] = {static_cast<size_t>(grids * blocks)};
  const size_t lws[1] = {static_cast<size_t>(blocks)};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[8] = {
        &d_subject,
        &d_avgs,
        &d_stds,
        &d_lb,
        &d_lower,
        &d_upper,
        &M,
        &N,
    };
    int pinfo[8] = {
        iris_r,
        iris_r,
        iris_r,
        iris_w,
        iris_r,
        iris_r,
        sizeof(M),
        sizeof(N),
    };
    task.kernel("lb_keogh", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 8, params, pinfo);
    task.flush_out(d_lb);
    task.submit(iris_gpu, nullptr, 1);
  }

  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n",
              (time * 1e-9f) / repeat);

  d_lb.fetch(lb.data());

  reference(subject.data(), avgs.data(), stds.data(), lb_h.data(),
            lower.data(), upper.data(), M, N);
  bool ok = true;
  for (int i = 0; i < out_count; i++) {
    if (std::fabs(lb[i] - lb_h[i]) > 1e-2f) {
      std::printf("%d %f %f\n", i, lb[i], lb_h[i]);
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  platform.finalize();
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

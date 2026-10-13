#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "reference.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = std::atoi(argv[1]);

  int beamsize = 4;
  int nhead = 16;
  int dim_feature = nhead * 256;
  int n_steps = 9;
  float scaler = std::sqrt(nhead * 1.f / dim_feature);
  int qk_col = dim_feature;
  int v_col = dim_feature;
  int threshold = 64;

  const int q_size = beamsize * dim_feature;
  const int k_size = beamsize * dim_feature * n_steps;
  const int v_size = beamsize * dim_feature * n_steps;
  std::vector<float> hq(q_size), hk(k_size), hv(v_size), h_dst(q_size), r_dst(q_size);

  std::srand(123);
  for (float& value : hq) value = std::rand() / static_cast<float>(RAND_MAX);
  for (float& value : hk) value = std::rand() / static_cast<float>(RAND_MAX);
  for (float& value : hv) value = std::rand() / static_cast<float>(RAND_MAX);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  iris::DMem q_mem(hq.data(), hq.size() * sizeof(float));
  iris::DMem k_mem(hk.data(), hk.size() * sizeof(float));
  iris::DMem v_mem(hv.data(), hv.size() * sizeof(float));
  iris::DMem dst_mem(h_dst.data(), h_dst.size() * sizeof(float));

  size_t gws[] = {static_cast<size_t>(q_size)};
  size_t lws[] = {256};
  void* params[] = {&q_mem, &k_mem, &v_mem, &beamsize, &n_steps, &qk_col,
                    &v_col, &nhead, &scaler, &threshold, &dst_mem};
  int pinfo[] = {iris_r, iris_r, iris_r, sizeof(beamsize), sizeof(n_steps),
                 sizeof(qk_col), sizeof(v_col), sizeof(nhead), sizeof(scaler),
                 sizeof(threshold), iris_w};

  iris::Task check_task;
  check_task.kernel("mha", 1, nullptr, gws, lws, 11, params, pinfo);
  check_task.flush_out(dst_mem);
  check_task.submit(iris_gpu, nullptr, 1);

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    iris::Task task;
    task.kernel("mha", 1, nullptr, gws, lws, 11, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }
  const auto end = std::chrono::steady_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (us)\n",
              (elapsed * 1e-3f) / repeat);

  mha_reference(hq.data(), hk.data(), hv.data(), beamsize, n_steps, qk_col,
                v_col, nhead, scaler, threshold, r_dst.data());
  bool ok = true;
  for (int i = 0; i < q_size; ++i) {
    if (std::fabs(h_dst[i] - r_dst[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  platform.finalize();
  return ok ? 0 : 1;
}

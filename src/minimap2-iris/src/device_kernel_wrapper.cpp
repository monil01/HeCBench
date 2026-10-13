#include <iris/iris.hpp>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include "device_kernel_wrapper.h"
#include "kernel_common.h"

void device_chain_kernel_wrapper(std::vector<control_dt>& cont,
    std::vector<anchor_dt>& arg, std::vector<return_dt>& ret,
    int max_dist_x, int max_dist_y, int bw) {
  iris::Platform platform;
  int argc = 0;
  char** argv = nullptr;
  if (platform.init(&argc, &argv, 1)) throw std::runtime_error("IRIS initialization failed");
  int count = 0, device = -1;
  iris_device_count(&count);
  for (int i = 0; i < count; ++i) {
    int backend = 0, type = 0;
    char name[256] = {};
    iris_device_info(i, iris_backend, &backend, nullptr);
    iris_device_info(i, iris_type, &type, nullptr);
    iris_device_info(i, iris_name, name, nullptr);
    fprintf(stderr, "IRIS device %d backend=%d type=%d name=%s\n", i, backend, type, name);
    if (backend == 3 && type == 256) device = i;
  }
  if (device < 0) throw std::runtime_error("HIP AMD GPU required");
  const auto batches = cont.size() / PE_NUM;
  ret.resize(batches * TILE_SIZE * PE_NUM);
  if (batches) {
    std::vector<score_dt> scores(PE_NUM * BACK_SEARCH_COUNT_GPU);
    std::vector<parent_dt> parents(scores.size());
    iris::DMem controls(cont.data(), cont.size() * sizeof(control_dt));
    iris::DMem anchors(arg.data(), arg.size() * sizeof(anchor_dt));
    iris::DMem results(ret.data(), ret.size() * sizeof(return_dt));
    iris::DMem score_mem(scores.data(), scores.size() * sizeof(score_dt));
    iris::DMem parent_mem(parents.data(), parents.size() * sizeof(parent_dt));
    size_t gws[] = {BLOCK_NUM * BACK_SEARCH_COUNT_GPU};
    size_t lws[] = {BACK_SEARCH_COUNT_GPU};
    const auto start = std::chrono::steady_clock::now();
    for (unsigned long long batch = 0; batch < batches; ++batch) {
      iris::Task task;
      void* params[] = {&results, &anchors, &controls, &score_mem, &parent_mem,
                       &max_dist_x, &max_dist_y, &bw, &batch};
      int info[] = {iris_rw, iris_r, iris_r, iris_rw, iris_rw,
                    sizeof(max_dist_x), sizeof(max_dist_y), sizeof(bw), sizeof(batch)};
      if (task.kernel("device_chain_tiled", 1, nullptr, gws, lws, 9, params, info))
        throw std::runtime_error("IRIS kernel setup failed");
      if (batch + 1 == batches) task.flush_out(results);
      if (task.submit(device, nullptr, 1)) throw std::runtime_error("IRIS task failed");
    }
    printf("Total kernel execution time: %f (s)\n",
        std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
  }
  platform.finalize();
}

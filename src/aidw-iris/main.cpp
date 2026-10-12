#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "reference.h"

static void launch_kernel(const char* name,
                          iris::DMem& dx_mem,
                          iris::DMem& dy_mem,
                          iris::DMem& dz_mem,
                          int dnum,
                          iris::DMem& ix_mem,
                          iris::DMem& iy_mem,
                          iris::DMem& iz_mem,
                          int inum,
                          float area,
                          iris::DMem& avg_dist_mem,
                          bool flush_output) {
  iris::Task task;
  void* params[] = {
      &dx_mem, &dy_mem, &dz_mem, &dnum, &ix_mem, &iy_mem, &iz_mem, &inum, &area, &avg_dist_mem};
  int pinfo[] = {
      iris_r, iris_r, iris_r, sizeof(dnum), iris_r, iris_r, iris_w, sizeof(inum), sizeof(area), iris_r};
  size_t gws[] = {static_cast<size_t>(((inum + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE)};
  size_t lws[] = {BLOCK_SIZE};
  task.kernel(name, 1, nullptr, gws, lws, 10, params, pinfo);
  if (flush_output) task.flush_out(iz_mem);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <pts> <check> <iterations>\n", argv[0]);
    printf("pts: number of points (unit: 1K)\n");
    printf("check: enable verification when the value is 1\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int numk = atoi(argv[1]);
  const int check = atoi(argv[2]);
  const int iterations = atoi(argv[3]);

  const int dnum = numk * 1024;
  const int inum = dnum;
  const size_t dnum_size = dnum * sizeof(float);
  const size_t inum_size = inum * sizeof(float);

  const float width = 2000, height = 2000;
  const float area = width * height;

  std::vector<float> dx(dnum), dy(dnum), dz(dnum);
  std::vector<float> avg_dist(dnum);
  std::vector<float> ix(inum), iy(inum), iz(inum);
  std::vector<float> h_iz(inum);

  srand(123);
  for (int i = 0; i < dnum; i++) {
    dx[i] = rand() / (float)RAND_MAX * 1000;
    dy[i] = rand() / (float)RAND_MAX * 1000;
    dz[i] = rand() / (float)RAND_MAX * 1000;
  }

  for (int i = 0; i < inum; i++) {
    ix[i] = rand() / (float)RAND_MAX * 1000;
    iy[i] = rand() / (float)RAND_MAX * 1000;
    iz[i] = 0.f;
  }

  for (int i = 0; i < dnum; i++) avg_dist[i] = rand() / (float)RAND_MAX * 3;

  printf("Size = : %d K \n", numk);
  printf("dnum = : %d\ninum = : %d\n", dnum, inum);

  if (check) {
    printf("Verification enabled\n");
    reference(dx.data(), dy.data(), dz.data(), dnum, ix.data(), iy.data(), h_iz.data(), inum, area, avg_dist.data());
  } else {
    printf("Verification disabled\n");
  }

  iris::DMem dx_mem(dx.data(), dnum_size);
  iris::DMem dy_mem(dy.data(), dnum_size);
  iris::DMem dz_mem(dz.data(), dnum_size);
  iris::DMem avg_dist_mem(avg_dist.data(), dnum_size);
  iris::DMem ix_mem(ix.data(), inum_size);
  iris::DMem iy_mem(iy.data(), inum_size);
  iris::DMem iz_mem(iz.data(), inum_size);

  launch_kernel("AIDW_Kernel", dx_mem, dy_mem, dz_mem, dnum, ix_mem, iy_mem, iz_mem, inum, area, avg_dist_mem, true);
  if (check) {
    bool ok = verify(iz.data(), h_iz.data(), inum, EPS);
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  launch_kernel("AIDW_Kernel_Tiled", dx_mem, dy_mem, dz_mem, dnum, ix_mem, iy_mem, iz_mem, inum, area, avg_dist_mem, true);
  if (check) {
    bool ok = verify(iz.data(), h_iz.data(), inum, EPS);
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; i++) {
    launch_kernel("AIDW_Kernel", dx_mem, dy_mem, dz_mem, dnum, ix_mem, iy_mem, iz_mem, inum, area, avg_dist_mem, false);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of AIDW_Kernel       %f (s)\n", (time * 1e-9f) / iterations);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; i++) {
    launch_kernel("AIDW_Kernel_Tiled", dx_mem, dy_mem, dz_mem, dnum, ix_mem, iy_mem, iz_mem, inum, area, avg_dist_mem, false);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of AIDW_Kernel_Tiled %f (s)\n", (time * 1e-9f) / iterations);

  platform.finalize();
  return 0;
}

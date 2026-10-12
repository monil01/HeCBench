#include <iris/iris.hpp>

#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "kernel.h"

constexpr int THREADS = 256;

static void gpu_pso_iris(int p, int r, float* positions, float* velocities,
                         float* pBests, float* gBest)
{
  int size = p * DIM;
  size_t size_byte = sizeof(float) * size;
  size_t res_size_byte = sizeof(float) * DIM;

  iris::DMem mem_pos(positions, size_byte);
  iris::DMem mem_vel(velocities, size_byte);
  iris::DMem mem_pbest(pBests, size_byte);
  iris::DMem mem_gbest(gBest, res_size_byte);

  size_t gws1[1] = {static_cast<size_t>(((size + THREADS - 1) / THREADS) * THREADS)};
  size_t gws2[1] = {static_cast<size_t>(((p + THREADS - 1) / THREADS) * THREADS)};
  size_t lws[1] = {THREADS};

  auto start = std::chrono::steady_clock::now();
  for (int iter = 0; iter < r; iter++) {
    float rp = getRandomClamped(iter);
    float rg = getRandomClamped(r - iter);

    iris::Task update_particle;
    void* particle_params[7] = {&mem_pos, &mem_vel, &mem_pbest, &mem_gbest, &p, &rp, &rg};
    int particle_pinfo[7] = {iris_rw, iris_rw, iris_r, iris_r,
                             sizeof(p), sizeof(rp), sizeof(rg)};
    update_particle.kernel("kernelUpdateParticle", 1, nullptr, gws1, lws, 7,
                           particle_params, particle_pinfo);
    update_particle.submit(iris_gpu, nullptr, 1);

    iris::Task update_pbest;
    void* pbest_params[4] = {&mem_pos, &mem_pbest, &mem_gbest, &p};
    int pbest_pinfo[4] = {iris_r, iris_rw, iris_rw, sizeof(p)};
    update_pbest.kernel("kernelUpdatePBest", 1, nullptr, gws2, lws, 4,
                        pbest_params, pbest_pinfo);
    update_pbest.submit(iris_gpu, nullptr, 1);
  }
  iris::Task flush;
  flush.flush_out(mem_gbest);
  flush.flush_out(mem_pbest);
  flush.submit(iris_gpu, nullptr, 1);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", time * 1e-3f / r);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of particles> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int p = atoi(argv[1]);
  const int r = atoi(argv[2]);

  printf("Number of particles is %d\n", p);
  printf("Number of dimensions is %d\n", DIM);

  size_t size = p * DIM;
  size_t size_byte = sizeof(float) * size;

  float* positions = static_cast<float*>(malloc(size_byte));
  float* velocities = static_cast<float*>(malloc(size_byte));
  float* pBests = static_cast<float*>(malloc(size_byte));
  float* pBests_ref = static_cast<float*>(malloc(size_byte));
  float* gBest = static_cast<float*>(malloc(sizeof(float) * DIM));

  srand(123);
  for (size_t i = 0; i < size; i++) {
    positions[i] = getRandom(START_RANGE_MIN, START_RANGE_MAX);
    pBests[i] = pBests_ref[i] = positions[i];
    velocities[i] = 0;
  }
  for (int i = 0; i < DIM; i++) gBest[i] = pBests[i];

  printf("\nExecute PSO on a device\n");
  gpu_pso_iris(p, r, positions, velocities, pBests, gBest);
  printf("Result=%f\n", host_fitness_function(gBest));

  printf("\nExecute PSO on a host. This may take a while for large problem size..\n");
  for (int i = 0; i < DIM; i++) gBest[i] = pBests_ref[i];
  pso(p, r, positions, velocities, pBests_ref, gBest);
  printf("Result=%f\n", host_fitness_function(gBest));

  bool ok = true;
  for (int i = 0; i < DIM; i++) {
    if (fabsf(pBests_ref[i] - pBests[i]) > 1e-3f) {
      printf("@%d %f %f\n", i, pBests_ref[i], pBests[i]);
      ok = false;
      break;
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");

  free(positions);
  free(velocities);
  free(pBests);
  free(pBests_ref);
  free(gBest);
  platform.finalize();
  return ok ? 0 : 1;
}

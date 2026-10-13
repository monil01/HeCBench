/*
   ECL-BH v4.5: Simulation of the gravitational forces in a star cluster using
   the Barnes-Hut n-body algorithm.

   Copyright (c) 2010-2020 Texas State University. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 * Neither the name of Texas State University nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Martin Burtscher and Sahar Azimi

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/ECL-BH/.

Publication: This work is described in detail in the following paper.
Martin Burtscher and Keshav Pingali. An Efficient CUDA Implementation of the
Tree-based Barnes Hut n-Body Algorithm. Chapter 6 in GPU Computing Gems
Emerald Edition, pp. 75-92. January 2011.
 */


#include <iris/iris.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <initializer_list>

// Layout matches CUDA/HIP vector types without native runtime host allocation.
struct alignas(16) float4 { float x, y, z, w; };
struct alignas(8) float2 { float x, y; };
struct float3 { float x, y, z; };
#define WARPSIZE 32

static void check(int rc, const char* operation) {
  if (rc != IRIS_SUCCESS) {
    fprintf(stderr, "IRIS failure: %s (%d)\n", operation, rc);
    exit(2);
  }
}
static int gpu_device;
static void launch(const char* name, size_t blocks, size_t threads,
                   std::initializer_list<const void*> args,
                   std::initializer_list<int> access) {
  iris::Task task;
  std::vector<void*> params;
  for (const void* arg : args) params.push_back(const_cast<void*>(arg));
  std::vector<int> modes(access);
  size_t gws = blocks * threads, lws = threads;
  check(task.kernel(name, 1, nullptr, &gws, &lws, params.size(),
                    params.data(), modes.data()), name);
  check(task.submit(gpu_device, nullptr, true), name);
}

// random number generator (https://github.com/staceyson/splash2/blob/master/codes/apps/barnes/util.C)

static int randx = 7;
static double drnd()
{
  const int lastrand = randx;
  randx = (1103515245L * randx + 12345) & 0x7FFFFFFF;
  return (double)lastrand / 2147483648.0;
}


int main(int argc, char* argv[])
{
  // perform some checks

  printf("ECL-BH v4.5\n");
  printf("Copyright (c) 2010-2020 Texas State University\n");

  if (argc != 3) {
    fprintf(stderr, "\n");
    fprintf(stderr, "arguments: number_of_bodies number_of_timesteps\n");
    exit(-1);
  }

  const int nbodies = atoi(argv[1]);
  if (nbodies < 1) {
    fprintf(stderr, "nbodies is too small: %d\n", nbodies);
    exit(-1);
  }

  if (nbodies > (1 << 30)) {
    fprintf(stderr, "nbodies is too large: %d\n", nbodies);
    exit(-1);
  }

  int timesteps = atoi(argv[2]);
  if (timesteps < 0) {
    fprintf(stderr, "the number of steps should be positive: %d\n", timesteps);
    exit(-1);
  }

  int i;
  int nnodes, step;
  float dtime, dthf, epssq, itolsq;

  float4 *accVel;
  float2 *vel;
  float4 *posMass;
  double rsc, vsc, r, v, x, y, z, sq, scale;

  // the number of thread blocks may be adjusted for higher performance
  const int blocks = 24;

  nnodes = nbodies * 2;
  if (nnodes < 1024*blocks) nnodes = 1024*blocks;
  while ((nnodes & (WARPSIZE-1)) != 0) nnodes++;
  nnodes--;

  dtime = 0.025f; 
  dthf = dtime * 0.5f;
  epssq = 0.05 * 0.05;
  itolsq = 1.0f / (0.5 * 0.5);

  printf("configuration: %d bodies, %d time steps\n", nbodies, timesteps);

  // allocate host memory

  std::vector<float4> acc_storage(nnodes+1);
  accVel = acc_storage.data();
  if (accVel == NULL) fprintf(stderr, "cannot allocate accVel\n");

  std::vector<float2> vel_storage(nnodes+1);
  vel = vel_storage.data();
  if (vel == NULL) fprintf(stderr, "cannot allocate vel\n");

  std::vector<float4> pos_storage(nnodes+1);
  posMass = pos_storage.data();
  if (posMass == NULL) fprintf(stderr, "cannot allocate posMass\n");

  // initialize host memory (https://github.com/staceyson/splash2/blob/master/codes/apps/barnes/code.C)

  rsc = (3 * 3.1415926535897932384626433832795) / 16;
  vsc = sqrt(1.0 / rsc);
  for (i = 0; i < nbodies; i++) {
    float4 p;
    p.w = 1.f / nbodies;
    r = 1.0 / sqrt(pow(drnd()*0.999, -2.0/3.0) - 1.0);
    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = rsc * r / sqrt(sq);
    p.x = x * scale;
    p.y = y * scale;
    p.z = z * scale;
    posMass[i] = p;

    do {
      x = drnd();
      y = drnd() * 0.1;
    } while (y > x*x * pow(1 - x*x, 3.5));
    v = x * sqrt(2.0 / sqrt(1 + r*r));
    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = vsc * v / sqrt(sq);
    float2 v;
    v.x = x * scale;
    v.y = y * scale;
    accVel[i].w = z * scale;
    vel[i] = v;
  }


  iris::Platform platform;
  check(platform.init(&argc, &argv, true), "initialize");
  int ndev = 0;
  check(iris_device_count(&ndev), "device count");
  gpu_device = -1;
  for (int dev = 0; dev < ndev; ++dev) {
    int type = 0, backend = -1;
    char name[256] = {};
    check(iris_device_info(dev, iris_type, &type, nullptr), "device type");
    check(iris_device_info(dev, iris_backend, &backend, nullptr), "backend");
    check(iris_device_info(dev, iris_name, name, nullptr), "device name");
    fprintf(stderr, "IRIS device %d: %s type=%d backend=%d\n", dev, name, type, backend);
    if ((type & iris_amd) && gpu_device < 0) gpu_device = dev;
  }
  if (gpu_device < 0) { fprintf(stderr, "No AMD GPU available\n"); return 2; }
  platform.overview();
  {
  std::vector<int> child((nnodes+1)*size_t(8)), count(nnodes+1),
                   start(nnodes+1), sort(nnodes+1);
  std::vector<float3> max_storage(blocks*2), min_storage(blocks*2);
  int step_value = -1, bottom = 0;
  unsigned int blkcnt = 0;
  float radius = 0;
  iris::DMem d_child(child.data(), child.size()*sizeof(int));
  iris::DMem d_count(count.data(), count.size()*sizeof(int));
  iris::DMem d_start(start.data(), start.size()*sizeof(int));
  iris::DMem d_sort(sort.data(), sort.size()*sizeof(int));
  iris::DMem d_posMass(posMass, pos_storage.size()*sizeof(float4));
  iris::DMem d_vel(vel, vel_storage.size()*sizeof(float2));
  iris::DMem d_accVel(accVel, acc_storage.size()*sizeof(float4));
  iris::DMem d_max(max_storage.data(), max_storage.size()*sizeof(float3));
  iris::DMem d_min(min_storage.data(), min_storage.size()*sizeof(float3));
  iris::DMem d_step(&step_value, sizeof(step_value));
  iris::DMem d_bottom(&bottom, sizeof(bottom));
  iris::DMem d_blkcnt(&blkcnt, sizeof(blkcnt));
  iris::DMem d_radius(&radius, sizeof(radius));
  auto starttime = std::chrono::steady_clock::now();
  launch("InitializationKernel", 1, 1, {&d_step, &d_blkcnt},
         {iris_w, iris_w});
  for (step = 0; step < timesteps; step++) {
  launch("BoundingBoxKernel", blocks*2, 256, {&nnodes, &nbodies, &d_start, &d_child, &d_posMass, &d_max, &d_min, &d_radius, &d_bottom, &d_step, &d_blkcnt},
         {sizeof(int), sizeof(int), iris_rw, iris_rw, iris_rw, iris_rw, iris_rw, iris_w, iris_w, iris_rw, iris_rw});
  launch("ClearKernel1", blocks, 256, {&nnodes, &nbodies, &d_child},
         {sizeof(int), sizeof(int), iris_rw});
  launch("TreeBuildingKernel", blocks*2, 256, {&nnodes, &nbodies, &d_child, &d_posMass, &d_radius, &d_bottom},
         {sizeof(int), sizeof(int), iris_rw, iris_r, iris_r, iris_rw});
  launch("ClearKernel2", blocks, 256, {&nnodes, &d_start, &d_posMass, &d_bottom},
         {sizeof(int), iris_rw, iris_rw, iris_r});
  launch("SummarizationKernel", blocks, 256, {&nnodes, &nbodies, &d_count, &d_child, &d_posMass, &d_bottom},
         {sizeof(int), sizeof(int), iris_rw, iris_r, iris_rw, iris_r});
  launch("SortKernel", blocks, 256, {&nnodes, &nbodies, &d_sort, &d_count, &d_start, &d_child, &d_bottom},
         {sizeof(int), sizeof(int), iris_w, iris_r, iris_rw, iris_rw, iris_r});
  launch("ForceCalculationKernel", blocks*2, 256, {&nnodes, &nbodies, &dthf, &itolsq, &epssq, &d_sort, &d_child, &d_posMass, &d_vel, &d_accVel, &d_radius, &d_step},
         {sizeof(int), sizeof(int), sizeof(float), sizeof(float), sizeof(float), iris_r, iris_r, iris_r, iris_rw, iris_rw, iris_r, iris_r});
  launch("IntegrationKernel", blocks*2, 256, {&nbodies, &dtime, &dthf, &d_posMass, &d_vel, &d_accVel},
         {sizeof(int), sizeof(float), sizeof(float), iris_rw, iris_rw, iris_rw});
  }
  auto endtime = std::chrono::steady_clock::now();
  auto runtime = std::chrono::duration_cast<std::chrono::nanoseconds>(endtime - starttime).count();
  printf("Total kernel execution time: %.4lf s\n", runtime * 1e-9);
  iris::Task flush;
  check(flush.flush_out(d_accVel), "flush accVel");
  check(flush.flush_out(d_vel), "flush vel");
  check(flush.flush_out(d_posMass), "flush posMass");
  check(flush.submit(gpu_device, nullptr, true), "flush outputs");
  }
#ifdef DEBUG
  // print output for verification
  for (i = 0; i < nbodies; i++) {
    printf("%d: %.2e %.2e %.2e\n", i, posMass[i].x, posMass[i].y, posMass[i].z);
    printf("%d: %.2e %.2e %.2e %.2e\n", i, accVel[i].x, accVel[i].y, accVel[i].z, accVel[i].w);
    printf("%d: %.2e %.2e\n", i, vel[i].x, vel[i].y);
  }
#endif

  check(platform.finalize(), "finalize");
  return 0;
}

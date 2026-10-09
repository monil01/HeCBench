#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static void reference(
    const int* d_tisspoints,
    const float* d_gtt,
    const float* d_gbartt,
    float* d_ct,
    const float* d_ctprev,
    const float* d_qt,
    int nnt,
    int nntDev,
    int step,
    int isp)
{
  for (int i = 0; i < step * nnt; i++) {
    int jtp, ixyz, ix, iy, iz, jx, jy, jz, istep;
    int nnt2 = 2 * nnt;
    float p = 0.f;

    int itp = i / step;
    int itp1 = i % step;
    if (itp < nnt) {
      ix = d_tisspoints[itp];
      iy = d_tisspoints[itp + nnt];
      iz = d_tisspoints[itp + nnt2];
      for (jtp = itp1; jtp < nnt; jtp += step) {
        jx = d_tisspoints[jtp];
        jy = d_tisspoints[jtp + nnt];
        jz = d_tisspoints[jtp + nnt2];
        ixyz = std::abs(jx - ix) + std::abs(jy - iy) + std::abs(jz - iz) + (isp - 1) * nntDev;
        p += d_gtt[ixyz] * d_ctprev[jtp] + d_gbartt[ixyz] * d_qt[jtp];
      }
      if (itp1 == 0) d_ct[itp] = p;
    }

    for (istep = 1; istep < step; istep++) {
      if (itp1 == istep && itp < nnt) d_ct[itp] += p;
    }
  }
}

static void run_tissue(
    iris::DMem& mem_tisspoints,
    iris::DMem& mem_gtt,
    iris::DMem& mem_gbartt,
    iris::DMem& mem_ct,
    iris::DMem& mem_ctprev,
    iris::DMem& mem_qt,
    int nnt,
    int nntDev,
    int step,
    int isp)
{
  const size_t threadsPerBlock = 256;
  const size_t blocksPerGrid = (static_cast<size_t>(step) * static_cast<size_t>(nnt) + threadsPerBlock - 1) / threadsPerBlock;
  size_t gws[1] = { blocksPerGrid * threadsPerBlock };
  size_t lws[1] = { threadsPerBlock };

  iris::Task task;
  void* params[10] = {
    &mem_tisspoints,
    &mem_gtt,
    &mem_gbartt,
    &mem_ct,
    &mem_ctprev,
    &mem_qt,
    &nnt,
    &nntDev,
    &step,
    &isp
  };
  int pinfo[10] = {
    iris_r,
    iris_r,
    iris_r,
    iris_rw,
    iris_r,
    iris_r,
    sizeof(nnt),
    sizeof(nntDev),
    sizeof(step),
    sizeof(isp)
  };

  task.kernel("tissue", 1, nullptr, gws, lws, 10, params, pinfo);
  task.flush_out(mem_ct);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::printf("Usage: %s <dimension of a 3D grid> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int dim = std::atoi(argv[1]);
  if (dim > 32) {
    std::printf("Maximum dimension is 32\n");
    platform.finalize();
    return 1;
  }
  const int repeat = std::atoi(argv[2]);

  const int nnt = dim * dim * dim;
  const int nntDev = 32 * 32 * 32;
  const int nsp = 2;
  const int step = 4;
  const int isp = 2;

  int* h_tisspoints = static_cast<int*>(std::malloc(3 * nntDev * sizeof(int)));
  float* h_gtt = static_cast<float*>(std::malloc(nsp * nntDev * sizeof(float)));
  float* h_gbartt = static_cast<float*>(std::malloc(nsp * nntDev * sizeof(float)));
  float* h_ct = static_cast<float*>(std::malloc(nntDev * sizeof(float)));
  float* h_ctprev = static_cast<float*>(std::malloc(nntDev * sizeof(float)));
  float* h_qt = static_cast<float*>(std::malloc(nntDev * sizeof(float)));
  float* h_ct_gold = static_cast<float*>(std::malloc(nntDev * sizeof(float)));

  if (!h_tisspoints || !h_gtt || !h_gbartt || !h_ct || !h_ctprev || !h_qt || !h_ct_gold) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(h_tisspoints);
    std::free(h_gtt);
    std::free(h_gbartt);
    std::free(h_ct);
    std::free(h_ctprev);
    std::free(h_qt);
    std::free(h_ct_gold);
    platform.finalize();
    return 1;
  }

  for (int i = 0; i < 3 * nntDev; i++) {
    h_tisspoints[i] = std::rand() % (nntDev / 3);
  }
  for (int i = 0; i < nsp * nntDev; i++) {
    h_gtt[i] = std::rand() / static_cast<float>(RAND_MAX);
    h_gbartt[i] = std::rand() / static_cast<float>(RAND_MAX);
  }
  for (int i = 0; i < nntDev; i++) {
    h_ct[i] = h_ct_gold[i] = 0.0f;
    h_ctprev[i] = std::rand() / static_cast<float>(RAND_MAX);
    h_qt[i] = std::rand() / static_cast<float>(RAND_MAX);
  }

  iris::DMem mem_tisspoints(h_tisspoints, 3 * nntDev * sizeof(int));
  iris::DMem mem_gtt(h_gtt, nsp * nntDev * sizeof(float));
  iris::DMem mem_gbartt(h_gbartt, nsp * nntDev * sizeof(float));
  iris::DMem mem_ct(h_ct, nntDev * sizeof(float));
  iris::DMem mem_ctprev(h_ctprev, nntDev * sizeof(float));
  iris::DMem mem_qt(h_qt, nntDev * sizeof(float));

  for (int i = 0; i < 2; i++) {
    run_tissue(mem_tisspoints, mem_gtt, mem_gbartt, mem_ct, mem_ctprev, mem_qt, nnt, nntDev, step, isp);
  }

  for (int i = 0; i < 2; i++) {
    reference(h_tisspoints, h_gtt, h_gbartt, h_ct_gold, h_ctprev, h_qt, nnt, nntDev, step, isp);
  }

  mem_ct.fetch(h_ct);
  bool ok = true;
  for (int i = 0; i < nntDev; i++) {
    if (std::fabs(h_ct[i] - h_ct_gold[i]) > 1e-2f) {
      std::printf("@%d: %f %f\n", i, h_ct[i], h_ct_gold[i]);
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    run_tissue(mem_tisspoints, mem_gtt, mem_gbartt, mem_ct, mem_ctprev, mem_qt, nnt, nntDev, step, isp);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  std::free(h_tisspoints);
  std::free(h_gtt);
  std::free(h_gbartt);
  std::free(h_ct);
  std::free(h_ctprev);
  std::free(h_qt);
  std::free(h_ct_gold);
  platform.finalize();
  return 0;
}

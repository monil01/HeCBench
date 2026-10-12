#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define dfloat double
#define dlong int

#define p_IJWID 6
#define p_JID   4
#define p_JWID  5
#define p_Np    512
#define p_Nq    8
#define p_Nvgeo 12
#define p_RXID  0
#define p_RYID  1
#define p_RZID  7
#define p_SXID  2
#define p_SYID  3
#define p_SZID  8
#define p_TXID  9
#define p_TYID  10
#define p_TZID  11
#define p_cubNp 4096
#define p_cubNq 16

#include "../adv-cuda/reference.h"

static dfloat* drandAlloc(int n) {
  dfloat* v = static_cast<dfloat*>(std::malloc(static_cast<size_t>(n) * sizeof(dfloat)));
  for (int i = 0; i < n; ++i) v[i] = drand48();
  return v;
}

int main(int argc, char** argv) {
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  if (argc < 4) {
    std::printf("Usage: ./main N cubN numElements [nRepetitions]\n");
    platform.finalize();
    return 255;
  }

  const int N = std::atoi(argv[1]);
  const int cubN = std::atoi(argv[2]);
  const dlong Nelements = std::atoi(argv[3]);
  int Ntests = 1;
  if (argc >= 5) Ntests = std::atoi(argv[4]);

  const int Nq = N + 1;
  const int cubNq = cubN + 1;
  const int Np = Nq * Nq * Nq;
  const int cubNp = cubNq * cubNq * cubNq;
  const dlong offset = Nelements * Np;

  std::printf("Data type in bytes: %zu\n", sizeof(dfloat));

  srand48(123);
  dfloat* vgeo           = drandAlloc(Np * Nelements * p_Nvgeo);
  dfloat* cubvgeo        = drandAlloc(cubNp * Nelements * p_Nvgeo);
  dfloat* cubDiffInterpT = drandAlloc(3 * cubNp * Nelements);
  dfloat* cubInterpT     = drandAlloc(Np * cubNp);
  dfloat* u              = drandAlloc(3 * Np * Nelements);
  dfloat* adv            = drandAlloc(3 * Np * Nelements);
  dfloat* adv_ref        = drandAlloc(3 * Np * Nelements);

  iris::DMem mem_vgeo(vgeo, static_cast<size_t>(Np) * Nelements * p_Nvgeo * sizeof(dfloat));
  iris::DMem mem_cubvgeo(cubvgeo, static_cast<size_t>(cubNp) * Nelements * p_Nvgeo * sizeof(dfloat));
  iris::DMem mem_cubDiffInterpT(cubDiffInterpT, static_cast<size_t>(3) * cubNp * Nelements * sizeof(dfloat));
  iris::DMem mem_cubInterpT(cubInterpT, static_cast<size_t>(Np) * cubNp * sizeof(dfloat));
  iris::DMem mem_u(u, static_cast<size_t>(3) * Np * Nelements * sizeof(dfloat));
  iris::DMem mem_adv(adv, static_cast<size_t>(3) * Np * Nelements * sizeof(dfloat));

  size_t gws[2] = {static_cast<size_t>(Nelements) * 16, 16};
  size_t lws[2] = {16, 16};
  int kernel_Nelements = Nelements;
  int kernel_offset = offset;

  auto start = std::chrono::high_resolution_clock::now();
  for (int test = 0; test < Ntests; ++test) {
    iris::Task task;
    void* params[] = {
        &kernel_Nelements,
        &mem_vgeo,
        &mem_cubvgeo,
        &mem_cubDiffInterpT,
        &mem_cubInterpT,
        &kernel_offset,
        &mem_u,
        &mem_adv};
    int pinfo[] = {
        sizeof(Nelements),
        iris_r,
        iris_r,
        iris_r,
        iris_r,
        sizeof(offset),
        iris_r,
        iris_w};
    task.kernel("advCubatureHex3D", 2, nullptr, gws, lws, 8, params, pinfo);
    if (test == Ntests - 1) task.flush_out(mem_adv);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::high_resolution_clock::now();
  const double elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() /
      static_cast<double>(Ntests);

  reference(Nelements, vgeo, cubvgeo, cubDiffInterpT, cubInterpT, offset, u, adv_ref);

  bool ok = true;
  for (int i = 0; i < 3 * Np * Nelements; i++) {
    if (std::fabs(adv[i] - adv_ref[i]) > 1e-4) {
      std::cout << i << " : " << adv[i] << " != " << adv_ref[i] << std::endl;
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  const dfloat GDOFPerSecond = (N * N * N) * Nelements / elapsed;
  std::cout << " NRepetitions=" << Ntests
            << " N=" << N
            << " cubN=" << cubN
            << " Nelements=" << Nelements
            << " elapsed time=" << elapsed
            << " GDOF/s=" << GDOFPerSecond
            << "\n";

  std::free(vgeo);
  std::free(cubvgeo);
  std::free(cubDiffInterpT);
  std::free(cubInterpT);
  std::free(u);
  std::free(adv);
  std::free(adv_ref);

  platform.finalize();
  return ok ? 0 : 1;
}

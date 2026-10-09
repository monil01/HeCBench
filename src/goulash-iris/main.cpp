#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "utils.h"

#define BLOCK_SIZE 256

static void run_gate(iris::DMem& mem_m_gate, iris::DMem& mem_vm, long nCells,
                     bool flush)
{
  const size_t gws[1] = {
      static_cast<size_t>(((nCells + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE)};
  const size_t lws[1] = {BLOCK_SIZE};

  iris::Task task;
  void* params[3] = {&mem_m_gate, &nCells, &mem_vm};
  int pinfo[3] = {iris_rw, sizeof(nCells), iris_r};
  task.kernel("gate", 1, nullptr, const_cast<size_t*>(gws),
              const_cast<size_t*>(lws), 3, params, pinfo);
  if (flush) task.flush_out(mem_m_gate);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <Iterations> <Kernel_GBs_used>\n\n", argv[0]);
    return 1;
  }

  long iterations = std::atol(argv[1]);
  double kernel_mem_used = std::atof(argv[2]);

  long nCells = static_cast<long>((kernel_mem_used * 1024.0 * 1024.0 * 1024.0) /
                                  (sizeof(double) * 2));
  std::printf("Number of cells: %ld\n", nCells);

  double* m_gate = static_cast<double*>(std::calloc(nCells, sizeof(double)));
  if (m_gate == nullptr) std::printf("failed calloc m_gate\n");

  double* m_gate_h = static_cast<double*>(std::calloc(nCells, sizeof(double)));
  if (m_gate_h == nullptr) std::printf("failed calloc m_gate_h\n");

  double* Vm = static_cast<double*>(std::calloc(nCells, sizeof(double)));
  if (Vm == nullptr) std::printf("failed calloc Vm\n");

  if (m_gate == nullptr || m_gate_h == nullptr || Vm == nullptr) {
    std::free(m_gate);
    std::free(m_gate_h);
    std::free(Vm);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  iris::DMem mem_m_gate(m_gate, sizeof(double) * nCells);
  iris::DMem mem_vm(Vm, sizeof(double) * nCells);
  mem_m_gate.update(m_gate);
  mem_vm.update(Vm);

  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;

  for (long itime = 0; itime <= iterations; itime++) {
    const bool flush_after_launch = (itime == 0);
    if (itime == 1) {
      start = std::chrono::steady_clock::now();
    }
    run_gate(mem_m_gate, mem_vm, nCells, flush_after_launch);
  }

  iris_synchronize();
  end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("total kernel time %lf(s) for %ld iterations\n", time * 1e-9,
              iterations - 1);

  reference(m_gate_h, nCells, Vm);

  bool ok = true;
  for (long i = 0; i < nCells; i++) {
    if (std::fabs(m_gate[i] - m_gate_h[i]) > 1e-6) {
      ok = false;
      break;
    }
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");

  platform.finalize();
  std::free(m_gate);
  std::free(m_gate_h);
  std::free(Vm);
  return ok ? 0 : 1;
}

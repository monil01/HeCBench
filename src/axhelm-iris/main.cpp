#include <iris/iris.hpp>
#include "hip_backend.hpp"
#include <vector>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define POLYNOMIAL_DEGREE 7
#define p_Nggeo 7
#define p_G00ID 1
#define p_G01ID 2
#define p_G02ID 3
#define p_G11ID 4
#define p_G12ID 5
#define p_G22ID 6
#define p_GWJID 0

#include "meshBasis.hpp"
#include "axhelmReference.cpp"

static dfloat* drandAlloc(int count) {
  dfloat* values = static_cast<dfloat*>(std::calloc(count, sizeof(dfloat)));
  for (int i = 0; i < count; ++i) values[i] = drand48();
  return values;
}

static int run_dimension(iris::Platform& platform, int ndim, int elements, int repeats,
                         dfloat* ggeo, dfloat* q, dfloat* aq, dfloat* drv,
                         dfloat* lambda, int offset, int np, int nq) {
  const size_t ggeo_bytes = static_cast<size_t>(np) * elements * p_Nggeo * sizeof(dfloat);
  const size_t q_bytes = static_cast<size_t>(ndim) * np * elements * sizeof(dfloat);
  const size_t drv_bytes = static_cast<size_t>(nq) * nq * sizeof(dfloat);
  const size_t lambda_bytes = static_cast<size_t>(2) * offset * sizeof(dfloat);
  const std::vector<dfloat> expected(aq, aq + ndim * offset);
  iris::DMem mem_ggeo(ggeo, ggeo_bytes);
  iris::DMem mem_q(q, q_bytes);
  iris::DMem mem_aq(aq, q_bytes);
  iris::DMem mem_drv(drv, drv_bytes);
  iris::DMem mem_lambda(lambda, lambda_bytes);

  const size_t gws[1] = {static_cast<size_t>(elements) * 64};
  const size_t lws[1] = {64};
  const char* kernel = ndim > 1 ? "axhelm_n3" : "axhelm";
  const auto start = std::chrono::steady_clock::now();
  for (int test = 0; test < repeats; ++test) {
    iris::Task task;
    // The output is a separate DMem object so the kernel's final argument is writable.
    void* launch_params[7] = {&elements, &offset, &mem_ggeo, &mem_drv,
                               &mem_lambda, &mem_q, &mem_aq};
    int launch_pinfo[7] = {sizeof(elements), sizeof(offset), iris_r, iris_r,
                           iris_r, iris_r, iris_w};
    task.kernel(kernel, 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 7, launch_params, launch_pinfo);
    task.flush_out(mem_aq);
    iris_check(task.submit(hip_device, nullptr, 1), kernel);
  }
  const auto end = std::chrono::steady_clock::now();
  // Each submitted task explicitly flushed the output buffer.
  const double elapsed = std::chrono::duration<double>(end - start).count() / repeats;

  dfloat max_diff = 0;
  for (int i = 0; i < ndim * offset; ++i) {
    if (!std::isfinite(aq[i])) return 1;
    max_diff = std::max(max_diff, static_cast<dfloat>(std::fabs(aq[i] - expected[i])));
  }
  std::cout << "Correctness check: maxError = " << max_diff << "\n";
  std::cout << (max_diff < 1e-3 ? "PASS" : "FAIL") << "\n";
  std::cout << "NRepetitions=" << repeats << " Ndim=" << ndim
            << " N=" << POLYNOMIAL_DEGREE << " Nelements=" << elements
            << " elapsed_seconds=" << elapsed << "\n";
  return max_diff < 1e-3 ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::printf("Usage: %s Ndim numElements nRepetitions\n", argv[0]);
    return 1;
  }
  const int ndim = std::atoi(argv[1]);
  const int elements = std::atoi(argv[2]);
  const int repeats = std::atoi(argv[3]);
  if (ndim != 1 && ndim != 3) {
    std::fprintf(stderr, "Ndim must be 1 or 3\n");
    return 1;
  }

  iris::Platform platform;
  iris_check(platform.init(&argc, &argv, 1), "initialize");
  select_hip_device();
  if (elements <= 0 || repeats <= 0) return 1;
  const int nq = POLYNOMIAL_DEGREE + 1;
  const int np = nq * nq * nq;
  const int offset = elements * np;
  dfloat *rv, *wv, *drv;
  meshJacobiGQ(0, 0, POLYNOMIAL_DEGREE, &rv, &wv);
  meshDmatrix1D(POLYNOMIAL_DEGREE, nq, rv, &drv);
  dfloat* ggeo = drandAlloc(np * elements * p_Nggeo);
  dfloat* q = drandAlloc(ndim * offset);
  dfloat* aq = drandAlloc(ndim * offset);
  dfloat* lambda = static_cast<dfloat*>(std::calloc(2 * offset, sizeof(dfloat)));
  for (int i = 0; i < offset; ++i) {
    lambda[i] = 1.0f;
    lambda[i + offset] = 1.1f;
  }
  for (int n = 0; n < ndim; ++n)
    axhelmReference(nq, elements, 1.1f, ggeo, drv, q + n * offset, aq + n * offset);

  const int status = run_dimension(platform, ndim, elements, repeats, ggeo, q, aq,
                                   drv, lambda, offset, np, nq);
  std::free(ggeo);
  std::free(q);
  std::free(aq);
  std::free(lambda);
  std::free(rv);
  std::free(wv);
  std::free(drv);
  platform.finalize();
  return status;
}

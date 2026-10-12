#include <iris/iris.hpp>

#include "binomialOptions.h"
#include "reference.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

typedef struct
{
  real S;
  real X;
  real vDt;
  real puByDf;
  real pdByDf;
} __TOptionData;

static real randData(real low, real high)
{
  real t = (real)rand() / (real)RAND_MAX;
  return ((real)1.0 - t) * low + t * high;
}

static void preprocess(__TOptionData* dst, TOptionData* optionData, int optN)
{
  for (int i = 0; i < optN; i++) {
    const real T = optionData[i].T;
    const real R = optionData[i].R;
    const real V = optionData[i].V;

    const real dt = T / (real)NUM_STEPS;
    const real vDt = V * std::sqrt(dt);
    const real rDt = R * dt;
    const real If = std::exp(rDt);
    const real Df = std::exp(-rDt);
    const real u = std::exp(vDt);
    const real d = std::exp(-vDt);
    const real pu = (If - d) / (u - d);
    const real pd = (real)1.0 - pu;

    dst[i].S = optionData[i].S;
    dst[i].X = optionData[i].X;
    dst[i].vDt = vDt;
    dst[i].puByDf = pu * Df;
    dst[i].pdByDf = pd * Df;
  }
}

static void run_binomial_gpu(real* callValue, TOptionData* optionData, int optN, int numIterations)
{
  __TOptionData h_OptionData[MAX_OPTIONS];
  preprocess(h_OptionData, optionData, optN);

  iris::DMem mem_options(h_OptionData, sizeof(__TOptionData) * MAX_OPTIONS);
  iris::DMem mem_call(callValue, sizeof(real) * MAX_OPTIONS);

  size_t gws[1] = {static_cast<size_t>(optN) * 128};
  size_t lws[1] = {128};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < numIterations; i++) {
    iris::Task task;
    void* params[2] = {&mem_options, &mem_call};
    int pinfo[2] = {iris_r, iris_w};
    task.kernel("binomialOptionsKernel", 1, nullptr, gws, lws, 2, params, pinfo);
    if (i == numIterations - 1) task.flush_out(mem_call);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time : %f (us)\n", time * 1e-3f / numIterations);
  mem_call.fetch(callValue);
}

int main(int argc, char **argv)
{
  (void)argc;
  std::printf("[%s] - Starting...\n", argv[0]);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int OPT_N = MAX_OPTIONS;
  TOptionData optionData[MAX_OPTIONS];
  real callValueBS[MAX_OPTIONS], callValueGPU[MAX_OPTIONS], callValueCPU[MAX_OPTIONS];
  real sumDelta, sumRef, gpuTime, errorVal;

  std::printf("Generating input data...\n");
  srand(123);

  for (int i = 0; i < OPT_N; i++) {
    optionData[i].S = randData(5.0f, 30.0f);
    optionData[i].X = randData(1.0f, 100.0f);
    optionData[i].T = randData(0.25f, 10.0f);
    optionData[i].R = 0.06f;
    optionData[i].V = 0.10f;
    BlackScholesCall(callValueBS[i], optionData[i]);
  }

  std::printf("Running GPU binomial tree...\n");
  auto start = std::chrono::high_resolution_clock::now();
  run_binomial_gpu(callValueGPU, optionData, OPT_N, NUM_ITERATIONS);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<real> elapsed_seconds = end - start;
  gpuTime = (real)elapsed_seconds.count();

  std::printf("Options count            : %i     \n", OPT_N);
  std::printf("Time steps               : %i     \n", NUM_STEPS);
  std::printf("Total binomialOptionsGPU() time: %f msec\n", gpuTime * 1000);
  std::printf("Options per second       : %f     \n", OPT_N / gpuTime);

  std::printf("Running CPU binomial tree...\n");
  for (int i = 0; i < OPT_N; i++)
    binomialOptionsCPU(callValueCPU[i], optionData[i]);

  std::printf("Comparing the results...\n");
  sumDelta = 0;
  sumRef = 0;
  std::printf("GPU binomial vs. Black-Scholes\n");
  for (int i = 0; i < OPT_N; i++) {
    sumDelta += std::fabs(callValueBS[i] - callValueGPU[i]);
    sumRef += std::fabs(callValueBS[i]);
  }
  if (sumRef > 1E-5)
    std::printf("L1 norm: %E\n", (double)(sumDelta / sumRef));
  else
    std::printf("Avg. diff: %E\n", (double)(sumDelta / (real)OPT_N));

  std::printf("CPU binomial vs. Black-Scholes\n");
  sumDelta = 0;
  sumRef = 0;
  for (int i = 0; i < OPT_N; i++) {
    sumDelta += std::fabs(callValueBS[i] - callValueCPU[i]);
    sumRef += std::fabs(callValueBS[i]);
  }
  if (sumRef > 1E-5)
    std::printf("L1 norm: %E\n", sumDelta / sumRef);
  else
    std::printf("Avg. diff: %E\n", (double)(sumDelta / (real)OPT_N));

  std::printf("CPU binomial vs. GPU binomial\n");
  sumDelta = 0;
  sumRef = 0;
  for (int i = 0; i < OPT_N; i++) {
    sumDelta += std::fabs(callValueGPU[i] - callValueCPU[i]);
    sumRef += callValueCPU[i];
  }

  std::printf("Avg. diff: %E\n", (double)(sumDelta / (real)OPT_N));
  std::printf("L1 norm: %E\n", errorVal = sumDelta / sumRef);

  if (errorVal > 5e-4) {
    std::printf("Test failed!\n");
    platform.finalize();
    return EXIT_FAILURE;
  }

  std::printf("Test passed\n");
  platform.finalize();
  return EXIT_SUCCESS;
}

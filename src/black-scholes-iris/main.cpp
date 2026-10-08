#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "black_scholes.hpp"

#define NUM_DIFF_SETTINGS 37

static const optionInputStruct kOptions[NUM_DIFF_SETTINGS] = {
  { CALL, 40.00f, 42.00f, 0.08f, 0.04f, 0.75f, 0.35f, 5.0975f, 1.0e-4f },
  { CALL, 100.00f, 90.00f, 0.10f, 0.10f, 0.10f, 0.15f, 0.0205f, 1.0e-4f },
  { CALL, 100.00f, 100.00f, 0.10f, 0.10f, 0.10f, 0.15f, 1.8734f, 1.0e-4f },
  { CALL, 100.00f, 110.00f, 0.10f, 0.10f, 0.10f, 0.15f, 9.9413f, 1.0e-4f },
  { CALL, 100.00f, 90.00f, 0.10f, 0.10f, 0.10f, 0.25f, 0.3150f, 1.0e-4f },
  { CALL, 100.00f, 100.00f, 0.10f, 0.10f, 0.10f, 0.25f, 3.1217f, 1.0e-4f },
  { CALL, 100.00f, 110.00f, 0.10f, 0.10f, 0.10f, 0.25f, 10.3556f, 1.0e-4f },
  { CALL, 100.00f, 90.00f, 0.10f, 0.10f, 0.10f, 0.35f, 0.9474f, 1.0e-4f },
  { CALL, 100.00f, 100.00f, 0.10f, 0.10f, 0.10f, 0.35f, 4.3693f, 1.0e-4f },
  { CALL, 100.00f, 110.00f, 0.10f, 0.10f, 0.10f, 0.35f, 11.1381f, 1.0e-4f },
  { CALL, 100.00f, 90.00f, 0.10f, 0.10f, 0.50f, 0.15f, 0.8069f, 1.0e-4f },
  { CALL, 100.00f, 100.00f, 0.10f, 0.10f, 0.50f, 0.15f, 4.0232f, 1.0e-4f },
  { CALL, 100.00f, 110.00f, 0.10f, 0.10f, 0.50f, 0.15f, 10.5769f, 1.0e-4f },
  { CALL, 100.00f, 90.00f, 0.10f, 0.10f, 0.50f, 0.25f, 2.7026f, 1.0e-4f },
  { CALL, 100.00f, 100.00f, 0.10f, 0.10f, 0.50f, 0.25f, 6.6997f, 1.0e-4f },
  { CALL, 100.00f, 110.00f, 0.10f, 0.10f, 0.50f, 0.25f, 12.7857f, 1.0e-4f },
  { CALL, 100.00f, 90.00f, 0.10f, 0.10f, 0.50f, 0.35f, 4.9329f, 1.0e-4f },
  { CALL, 100.00f, 100.00f, 0.10f, 0.10f, 0.50f, 0.35f, 9.3679f, 1.0e-4f },
  { CALL, 100.00f, 110.00f, 0.10f, 0.10f, 0.50f, 0.35f, 15.3086f, 1.0e-4f },
  { PUT, 100.00f, 90.00f, 0.10f, 0.10f, 0.10f, 0.15f, 9.9210f, 1.0e-4f },
  { PUT, 100.00f, 100.00f, 0.10f, 0.10f, 0.10f, 0.15f, 1.8734f, 1.0e-4f },
  { PUT, 100.00f, 110.00f, 0.10f, 0.10f, 0.10f, 0.15f, 0.0408f, 1.0e-4f },
  { PUT, 100.00f, 90.00f, 0.10f, 0.10f, 0.10f, 0.25f, 10.2155f, 1.0e-4f },
  { PUT, 100.00f, 100.00f, 0.10f, 0.10f, 0.10f, 0.25f, 3.1217f, 1.0e-4f },
  { PUT, 100.00f, 110.00f, 0.10f, 0.10f, 0.10f, 0.25f, 0.4551f, 1.0e-4f },
  { PUT, 100.00f, 90.00f, 0.10f, 0.10f, 0.10f, 0.35f, 10.8479f, 1.0e-4f },
  { PUT, 100.00f, 100.00f, 0.10f, 0.10f, 0.10f, 0.35f, 4.3693f, 1.0e-4f },
  { PUT, 100.00f, 110.00f, 0.10f, 0.10f, 0.10f, 0.35f, 1.2376f, 1.0e-4f },
  { PUT, 100.00f, 90.00f, 0.10f, 0.10f, 0.50f, 0.15f, 10.3192f, 1.0e-4f },
  { PUT, 100.00f, 100.00f, 0.10f, 0.10f, 0.50f, 0.15f, 4.0232f, 1.0e-4f },
  { PUT, 100.00f, 110.00f, 0.10f, 0.10f, 0.50f, 0.15f, 1.0646f, 1.0e-4f },
  { PUT, 100.00f, 90.00f, 0.10f, 0.10f, 0.50f, 0.25f, 12.2149f, 1.0e-4f },
  { PUT, 100.00f, 100.00f, 0.10f, 0.10f, 0.50f, 0.25f, 6.6997f, 1.0e-4f },
  { PUT, 100.00f, 110.00f, 0.10f, 0.10f, 0.50f, 0.25f, 3.2734f, 1.0e-4f },
  { PUT, 100.00f, 90.00f, 0.10f, 0.10f, 0.50f, 0.35f, 14.4452f, 1.0e-4f },
  { PUT, 100.00f, 100.00f, 0.10f, 0.10f, 0.50f, 0.35f, 9.3679f, 1.0e-4f },
  { PUT, 100.00f, 110.00f, 0.10f, 0.10f, 0.50f, 0.35f, 5.7963f, 1.0e-4f }
};

static float price_option_cpu(const optionInputStruct& opt)
{
  const float variance = opt.vol * opt.vol * opt.t;
  const float dividendDiscount = std::exp(-opt.q * opt.t);
  const float riskFreeDiscount = std::exp(-opt.r * opt.t);
  const float forwardPrice = opt.spot * dividendDiscount / riskFreeDiscount;
  const float stdDev = std::sqrt(variance);
  const float d1 = std::log(forwardPrice / opt.strike) / stdDev + 0.5f * stdDev;
  const float d2 = d1 - stdDev;
  const float cum_d1 = 0.5f * (1.0f + std::erf(d1 * M_SQRT_2));
  const float cum_d2 = 0.5f * (1.0f + std::erf(d2 * M_SQRT_2));

  if (opt.type == CALL) {
    return riskFreeDiscount * (forwardPrice * cum_d1 - opt.strike * cum_d2);
  }
  return riskFreeDiscount * (forwardPrice * (-1.0f + cum_d1) + opt.strike * (1.0f - cum_d2));
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  const int numVals = 50000000;
  std::vector<optionInputStruct> values(numVals);
  std::vector<float> outputVals(numVals, 0.0f);
  std::vector<float> cpuVals(numVals, 0.0f);

  for (int i = 0; i < numVals; i++) {
    values[i] = kOptions[i % NUM_DIFF_SETTINGS];
  }

  std::printf("Number of options: %d\n\n", numVals);

  iris::DMem optionsMem(values.data(), values.size() * sizeof(optionInputStruct));
  iris::DMem outputMem(outputVals.data(), outputVals.size() * sizeof(float));

  size_t gws[1] = { static_cast<size_t>(((numVals + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE) * THREAD_BLOCK_SIZE) };
  size_t lws[1] = { THREAD_BLOCK_SIZE };

  auto start = std::chrono::steady_clock::now();
  auto kstart = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[3] = { &optionsMem, &outputMem, const_cast<int*>(&numVals) };
    int pinfo[3] = { iris_r, iris_w, sizeof(numVals) };
    task.kernel("getOutValOption", 1, nullptr, gws, lws, 3, params, pinfo);
    task.flush_out(outputMem);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto kend = std::chrono::steady_clock::now();
  auto end = std::chrono::steady_clock::now();

  double ktimeGpu = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
  double avg_ktimeGpu = ktimeGpu * 1e-6 / repeat;
  double mtimeGpu = (std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() - ktimeGpu) * 1e-6 + avg_ktimeGpu;

  std::printf("Run on GPU\n");
  std::printf("Average kernel execution time on GPU: %f (ms)\n", avg_ktimeGpu);
  std::printf("Processing time using GPU %f (ms)\n", mtimeGpu);

  float gpuTotal = 0.0f;
  for (int i = 0; i < numVals; i++) gpuTotal += outputVals[i];
  std::printf("Summation of output prices on GPU: %f\n", gpuTotal);
  std::printf("Output price at index %d on GPU: %f\n\n", numVals / 2, outputVals[numVals / 2]);

  start = std::chrono::steady_clock::now();
  int errors = 0;
  float cpuTotal = 0.0f;
  float maxAbsDiff = 0.0f;
  for (int i = 0; i < numVals; i++) {
    cpuVals[i] = price_option_cpu(values[i]);
    cpuTotal += cpuVals[i];
    float diff = std::fabs(cpuVals[i] - outputVals[i]);
    maxAbsDiff = diff > maxAbsDiff ? diff : maxAbsDiff;
    errors += diff <= values[i].tol ? 0 : 1;
  }
  end = std::chrono::steady_clock::now();
  double mtimeCpu = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-6;

  std::printf("Run on CPU\n");
  std::printf("Processing time on CPU: %f (ms)\n", mtimeCpu);
  std::printf("Summation of output prices on CPU: %f\n", cpuTotal);
  std::printf("Output price at index %d on CPU: %f\n\n", numVals / 2, cpuVals[numVals / 2]);
  std::printf("Speedup on GPU: %f\n", mtimeCpu / mtimeGpu);
  std::printf("Max absolute difference: %g\n", maxAbsDiff);
  std::printf("%s\n", errors == 0 ? "PASS" : "FAIL");

  platform.finalize();
  return errors == 0 ? 0 : 1;
}

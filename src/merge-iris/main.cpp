#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CSV 0
#if (CSV)
#define PS(X, S)            \
  std::cout << X << ", " << S << ", "; \
  fflush(stdout);
#define PV(X)            \
  std::cout << X << ", "; \
  fflush(stdout);
#else
#define PS(X, S)                       \
  std::cout << X << " " << S << " :\n"; \
  fflush(stdout);
#define PV(X)                                  \
  std::cout << "\t" << #X << " \t: " << X << "\n"; \
  fflush(stdout);
#endif

static float positiveInfinity(float) { return FLT_MAX; }
static double positiveInfinity(double) { return DBL_MAX; }
static uint32_t positiveInfinity(uint32_t) { return 0xFFFFFFFFUL; }
static uint64_t positiveInfinity(uint64_t) { return 0xFFFFFFFFFFFFFFFFUL; }

static float negativeInfinity(float) { return FLT_MIN; }
static double negativeInfinity(double) { return DBL_MIN; }
static uint32_t negativeInfinity(uint32_t) { return 0; }
static uint64_t negativeInfinity(uint64_t) { return 0; }

template <typename vec_t>
static vec_t getPositiveInfinity() {
  vec_t tmp = 0;
  return positiveInfinity(tmp);
}

template <typename vec_t>
static vec_t getNegativeInfinity() {
  vec_t tmp = 0;
  return negativeInfinity(tmp);
}

template <typename vec_t>
static vec_t rand64() {
  vec_t rtn;
  do {
    uint32_t* rtn32 = reinterpret_cast<uint32_t*>(&rtn);
    rtn32[0] = rand();
    if (sizeof(vec_t) > 4) rtn32[1] = rand();
  } while (!(rtn < getPositiveInfinity<vec_t>() &&
             rtn > getNegativeInfinity<vec_t>()));
  return rtn;
}

#define PADDING 1024

template <typename vec_t>
struct KernelNames;

template <>
struct KernelNames<uint32_t> {
  static constexpr const char* diagonals = "workloadDiagonals_u32";
  static constexpr const char* merge = "mergeSinglePath_u32";
};

template <>
struct KernelNames<float> {
  static constexpr const char* diagonals = "workloadDiagonals_f32";
  static constexpr const char* merge = "mergeSinglePath_f32";
};

template <>
struct KernelNames<uint64_t> {
  static constexpr const char* diagonals = "workloadDiagonals_u64";
  static constexpr const char* merge = "mergeSinglePath_u64";
};

template <>
struct KernelNames<double> {
  static constexpr const char* diagonals = "workloadDiagonals_f64";
  static constexpr const char* merge = "mergeSinglePath_f64";
};

template <typename vec_t, uint32_t blocks, uint32_t threads, bool timing>
static uint32_t mergeType(const uint64_t size, const uint32_t runs) {
  std::vector<vec_t> hA(size + PADDING);
  std::vector<vec_t> hB(size + PADDING);
  std::vector<vec_t> hC(2 * size + PADDING);
  std::vector<uint32_t> hDpi(2 * (blocks + 1));

  iris::DMem dA(hA.data(), (size + PADDING) * sizeof(vec_t));
  iris::DMem dB(hB.data(), (size + PADDING) * sizeof(vec_t));
  iris::DMem dC(hC.data(), (2 * size + PADDING) * sizeof(vec_t));
  iris::DMem dpi(hDpi.data(), (2 * (blocks + 1)) * sizeof(uint32_t));

  uint32_t errors = 0;
  double total_time = 0.0;

  for (uint32_t r = 0; r < runs; r++) {
    for (uint64_t n = 0; n < size; n++) {
      hA[n] = rand64<vec_t>();
      hB[n] = rand64<vec_t>();
    }

    for (uint64_t n = size; n < size + PADDING; n++) {
      hA[n] = getPositiveInfinity<vec_t>();
      hB[n] = getPositiveInfinity<vec_t>();
    }

    std::sort(hA.begin(), hA.end());
    std::sort(hB.begin(), hB.end());
    dA.update(hA.data());
    dB.update(hB.data());

    auto start = std::chrono::steady_clock::now();

    {
      uint32_t a_length = static_cast<uint32_t>(size);
      uint32_t b_length = static_cast<uint32_t>(size);
      size_t gws[1] = {blocks * 32};
      size_t lws[1] = {32};
      void* params[5] = {&dA, &a_length, &dB, &b_length, &dpi};
      int pinfo[5] = {iris_r, sizeof(a_length), iris_r, sizeof(b_length),
                      iris_w};
      iris::Task task;
      task.kernel(KernelNames<vec_t>::diagonals, 1, nullptr, gws, lws, 5,
                  params, pinfo);
      task.submit(iris_gpu, nullptr, 1);
    }

    {
      uint32_t a_length = static_cast<uint32_t>(size);
      uint32_t b_length = static_cast<uint32_t>(size);
      uint32_t c_length = static_cast<uint32_t>(size * 2);
      size_t gws[1] = {blocks * threads};
      size_t lws[1] = {threads};
      void* params[7] = {&dA, &a_length, &dB, &b_length, &dpi, &dC, &c_length};
      int pinfo[7] = {iris_r, sizeof(a_length), iris_r, sizeof(b_length),
                      iris_r, iris_w, sizeof(c_length)};
      iris::Task task;
      task.kernel(KernelNames<vec_t>::merge, 1, nullptr, gws, lws, 7, params,
                  pinfo);
      task.flush_out(dC);
      task.submit(iris_gpu, nullptr, 1);
    }

    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    total_time += time;

    dC.fetch(hC.data());
    for (uint32_t i = 1; i < size; i++) {
      errors += hC[i] < hC[i - 1];
    }
  }

  PV(errors);
  std::printf("%s. ", errors ? "FAIL" : "PASS");

  if (timing)
    std::printf("Average kernel execution time: %f (us).\n",
                (total_time * 1e-3f) / runs);
  else
    std::printf("Warmup run\n");
  return errors;
}

template <uint32_t blocks, uint32_t threads>
static uint32_t mergeAllTypes(const uint64_t size, const uint32_t runs) {
  uint32_t errors = 0;
  PS("uint32_t", size)
  errors += mergeType<uint32_t, blocks, threads, false>(size, runs);
  std::printf("\n");
  PS("uint32_t", size)
  errors += mergeType<uint32_t, blocks, threads, true>(size, runs);
  std::printf("\n");

  PS("float", size)
  errors += mergeType<float, blocks, threads, false>(size, runs);
  std::printf("\n");
  PS("float", size)
  errors += mergeType<float, blocks, threads, true>(size, runs);
  std::printf("\n");

  PS("uint64_t", size)
  errors += mergeType<uint64_t, blocks, threads, false>(size, runs);
  std::printf("\n");
  PS("uint64_t", size)
  errors += mergeType<uint64_t, blocks, threads, true>(size, runs);
  std::printf("\n");

  PS("double", size)
  errors += mergeType<double, blocks, threads, false>(size, runs);
  std::printf("\n");
  PS("double", size)
  errors += mergeType<double, blocks, threads, true>(size, runs);
  std::printf("\n");
  return errors;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::printf("Usage: %s <length of the arrays> <runs>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const uint64_t length = std::atol(argv[1]);
  const uint32_t runs = std::atoi(argv[2]);

  const int blocks = 112;
  const int threads = 128;

  uint32_t errors = mergeAllTypes<blocks, threads>(length, runs);

  platform.finalize();
  return errors == 0 ? 0 : 1;
}

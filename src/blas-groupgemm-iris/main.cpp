#include <iris/iris.hpp>

#include <hip/hip_fp16.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "reference.h"

constexpr int THREADS = 256;

template <typename T>
static inline T fromFloat(float x)
{
  return (T)x;
}

template <>
inline __half fromFloat<__half>(float x)
{
  return __float2half(x);
}

template <typename T>
static inline float toFloat(T x)
{
  return (float)x;
}

template <>
inline float toFloat<__half>(__half x)
{
  return __half2float(x);
}

static bool groupedGemmHalf(const char* name,
                            const std::vector<int>& m,
                            const std::vector<int>& n,
                            const std::vector<int>& k,
                            int repeat)
{
  printf(">>>>>>>>>>>>>>> %s grouped GEMM >>>>>>>>>>>>>>>\n", name);
  const int groups = (int)m.size();

  std::vector<std::vector<float>> hAf(groups), hBf(groups), hReff(groups), hCf(groups);
  std::vector<std::vector<__half>> hA(groups), hB(groups), hC(groups);
  std::vector<size_t> a_offsets(groups), b_offsets(groups), c_offsets(groups), c_prefix(groups + 1);
  size_t total_a = 0, total_b = 0, total_c = 0;

  srand48(123);
  for (int g = 0; g < groups; g++) {
    hAf[g].resize((size_t)m[g] * k[g]);
    hBf[g].resize((size_t)k[g] * n[g]);
    hReff[g].assign((size_t)m[g] * n[g], 0.f);
    hCf[g].resize((size_t)m[g] * n[g]);
    hA[g].resize(hAf[g].size());
    hB[g].resize(hBf[g].size());
    hC[g].resize(hCf[g].size());
    for (size_t i = 0; i < hAf[g].size(); i++) {
      hAf[g][i] = (float)drand48();
      hA[g][i] = fromFloat<__half>(hAf[g][i]);
    }
    for (size_t i = 0; i < hBf[g].size(); i++) {
      hBf[g][i] = (float)drand48();
      hB[g][i] = fromFloat<__half>(hBf[g][i]);
    }
    a_offsets[g] = total_a;
    b_offsets[g] = total_b;
    c_offsets[g] = total_c;
    c_prefix[g] = total_c;
    total_a += hA[g].size();
    total_b += hB[g].size();
    total_c += hC[g].size();
  }
  c_prefix[groups] = total_c;

  std::vector<__half> flat_a(total_a), flat_b(total_b), flat_c(total_c);
  for (int g = 0; g < groups; g++) {
    std::copy(hA[g].begin(), hA[g].end(), flat_a.begin() + a_offsets[g]);
    std::copy(hB[g].begin(), hB[g].end(), flat_b.begin() + b_offsets[g]);
  }

  iris::DMem mem_a(flat_a.data(), flat_a.size() * sizeof(__half));
  iris::DMem mem_b(flat_b.data(), flat_b.size() * sizeof(__half));
  iris::DMem mem_c(flat_c.data(), flat_c.size() * sizeof(__half));
  iris::DMem mem_m(const_cast<int*>(m.data()), m.size() * sizeof(int));
  iris::DMem mem_n(const_cast<int*>(n.data()), n.size() * sizeof(int));
  iris::DMem mem_k(const_cast<int*>(k.data()), k.size() * sizeof(int));
  iris::DMem mem_a_offsets(a_offsets.data(), a_offsets.size() * sizeof(size_t));
  iris::DMem mem_b_offsets(b_offsets.data(), b_offsets.size() * sizeof(size_t));
  iris::DMem mem_c_offsets(c_offsets.data(), c_offsets.size() * sizeof(size_t));
  iris::DMem mem_c_prefix(c_prefix.data(), c_prefix.size() * sizeof(size_t));

  const size_t gws[1] = {((total_c + THREADS - 1) / THREADS) * THREADS};
  const size_t lws[1] = {THREADS};

  auto launch = [&]() {
    iris::Task task;
    void* params[12] = {&mem_a, &mem_b, &mem_c, &mem_m, &mem_n, &mem_k,
                        &mem_a_offsets, &mem_b_offsets, &mem_c_offsets, &mem_c_prefix,
                        const_cast<int*>(&groups), &total_c};
    int pinfo[12] = {iris_r, iris_r, iris_w, iris_r, iris_r, iris_r,
                     iris_r, iris_r, iris_r, iris_r, sizeof(groups), sizeof(total_c)};
    task.kernel("grouped_gemm_half", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 12, params, pinfo);
    task.flush_out(mem_c);
    task.submit(iris_gpu, nullptr, 1);
  };

  for (int r = 0; r < 30; r++) launch();

  for (int g = 0; g < groups; g++) {
    std::copy(flat_c.begin() + c_offsets[g],
              flat_c.begin() + c_offsets[g] + hC[g].size(),
              hC[g].begin());
    for (size_t i = 0; i < hC[g].size(); i++) hCf[g][i] = toFloat<__half>(hC[g][i]);
  }

  groupedGemm_ref(m, n, k, 1.f, 0.f, hAf, hBf, hReff);
  bool ok = verify(hCf, hReff, 1.0);

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) launch();
  auto end = std::chrono::steady_clock::now();
  double avg_us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
                  * 1e-3 / repeat;
  performance(m, n, k, avg_us);
  return ok;
}

int main(int argc, char* argv[])
{
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = (argc > 1) ? atoi(argv[1]) : 100;
  const int num_experts = (argc > 2) ? atoi(argv[2]) : 64;
  const int hidden = (argc > 3) ? atoi(argv[3]) : 2048;
  const int inter = (argc > 4) ? atoi(argv[4]) : 2048;
  const int avg_tokens = (argc > 5) ? atoi(argv[5]) : 16;

  const long num_tokens = (long)num_experts * avg_tokens;
  srand48(123);
  std::vector<double> cdf(num_experts);
  double wsum = 0;
  for (int e = 0; e < num_experts; e++) {
    wsum += 0.2 + drand48();
    cdf[e] = wsum;
  }

  std::vector<int> m(num_experts, 0), n(num_experts, inter), k(num_experts, hidden);
  for (long t = 0; t < num_tokens; t++) {
    double r = drand48() * wsum;
    int e = 0;
    while (e < num_experts - 1 && r > cdf[e]) e++;
    m[e]++;
  }
  for (int e = 0; e < num_experts; e++) {
    if (m[e] == 0) m[e] = 1;
  }

  printf("MoE FFN grouped GEMM: %d experts, hidden(K)=%d, intermediate(N)=%d, repeat=%d\n",
         num_experts, hidden, inter, repeat);

  bool ok = groupedGemmHalf("Half precision", m, n, k, repeat);
  platform.finalize();
  return ok ? 0 : 1;
}

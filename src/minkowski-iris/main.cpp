#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

using namespace std;

#define BLOCK_SIZE 16

constexpr int m_size = 512 * 8;
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int K = m_size / 2;

#include "verify.cpp"

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = atoi(argv[1]);
  float(*a_host)[N] = new float[M][N];
  float(*b_host)[K] = new float[N][K];
  float(*c_host)[K] = new float[M][K];
  float(*c_back)[K] = new float[M][K];

  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      a_host[i][j] = 1.f / N;

  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < K; j++)
      b_host[i][j] = rand() % 256;

  for (int j = 0; j < K; j++) {
    float sum = 0;
    for (int i = 0; i < N; i++) sum += b_host[i][j];
    for (int i = 0; i < N; i++) b_host[i][j] /= sum;
  }

  iris::DMem mem_a(a_host, sizeof(float) * M * N);
  iris::DMem mem_b(b_host, sizeof(float) * N * K);
  iris::DMem mem_c(c_back, sizeof(float) * M * K);

  unsigned int grid_cols = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
  unsigned int grid_rows = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const size_t gws[2] = { grid_cols * BLOCK_SIZE, grid_rows * BLOCK_SIZE };
  const size_t lws[2] = { BLOCK_SIZE, BLOCK_SIZE };

  cout << "Problem size: c(" << M << "," << K << ") = a(" << M << "," << N
       << ") * b(" << N << "," << K << ")\n";

  for (int k = 1; k <= 4; k++) {
    printf("Minkowski distance with p = %d\n", k);
    float p = static_cast<float>(k);
    float one_over_p = 1.f / p;

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      iris::Task task;
      void* params[5] = { &mem_a, &mem_b, &mem_c, &p, &one_over_p };
      int pinfo[5] = { iris_r, iris_r, iris_w, sizeof(p), sizeof(one_over_p) };
      task.kernel("minkowski", 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 5, params, pinfo);
      if (i == repeat - 1) {
        task.flush_out(mem_c);
      }
      task.submit(iris_gpu, nullptr, 1);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    mem_c.fetch(c_back);
#ifdef VERIFY
    VerifyResult(a_host, b_host, c_host, c_back, p, one_over_p);
#endif
  }

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;
  delete[] c_back;
  platform.finalize();
  return 0;
}

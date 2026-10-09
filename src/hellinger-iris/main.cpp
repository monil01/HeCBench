#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <new>

#define BLOCK_SIZE 16

#ifdef DOUBLE_PRECISION
  #define SQRT sqrt
  #define FABS fabs
  #define FP double
#else
  #define SQRT sqrtf
  #define FABS fabsf
  #define FP float
#endif

constexpr int m_size = 768 * 8;
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int P = m_size / 2;

#ifdef VERIFY
bool ValueSame(FP a, FP b) {
  return FABS(a - b) < 1e-5;
}

void VerifyResult(FP (*a_host)[N], FP (*b_host)[P], FP (*c_host)[P],
                  FP (*c_back)[P]) {
  int i, j, k;

  for (i = 0; i < M; i++)
    for (j = 0; j < P; j++) c_host[i][j] = (FP)0.0;

  for (i = 0; i < M; i++) {
    for (k = 0; k < N; k++) {
      for (j = 0; j < P; j++) {
        c_host[i][j] += SQRT(a_host[i][k] * b_host[k][j]);
      }
    }
  }
  for (i = 0; i < M; i++) {
    for (j = 0; j < P; j++) {
      FP value = ((FP)1.0 - c_host[i][j]);
      FP gate = (!std::signbit(value));
      c_host[i][j] = SQRT(gate * value);
    }
  }

  bool mismatch_found = false;
  int print_count = 0;

  for (i = 0; i < M; i++) {
    for (j = 0; j < P; j++) {
      if (!ValueSame(c_back[i][j], c_host[i][j])) {
        std::cout << "Fail - The result is incorrect for element: [" << i
                  << ", " << j << "], expected: " << c_host[i][j]
                  << ", but found: " << c_back[i][j] << "\n";
        mismatch_found = true;
        print_count++;
        if (print_count == 5) break;
      }
    }

    if (print_count == 5) break;
  }

  if (!mismatch_found) {
    std::cout << "PASS\n";
  } else {
    std::cout << "FAIL\n";
  }
}
#endif

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <repeat>\n";
    return 1;
  }
  const int repeat = std::atoi(argv[1]);

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  FP(*a_host)[N] = new FP[M][N];
  FP(*b_host)[P] = new FP[N][P];
  FP(*c_host)[P] = new FP[M][P];
  FP(*c_back)[P] = new FP[M][P];

  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      a_host[i][j] = (FP)1.0 / N;

  std::srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < P; j++)
      b_host[i][j] = std::rand() % 256;

  for (int j = 0; j < P; j++) {
    FP sum = 0;
    for (int i = 0; i < N; i++)
      sum += b_host[i][j];
    for (int i = 0; i < N; i++)
      b_host[i][j] /= sum;
  }

  iris::DMem mem_a(a_host, sizeof(FP) * M * N);
  iris::DMem mem_b(b_host, sizeof(FP) * N * P);
  iris::DMem mem_c(c_back, sizeof(FP) * M * P);

  int m = M;
  int n = N;
  int p = P;
  size_t gws[2] = {
      static_cast<size_t>((P + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE),
      static_cast<size_t>((M + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE)};
  size_t lws[2] = {BLOCK_SIZE, BLOCK_SIZE};

  auto start = std::chrono::steady_clock::now();

  for (int iter = 0; iter < repeat; iter++) {
    iris::Task task;
    void* params[6] = {&mem_a, &mem_b, &mem_c, &m, &n, &p};
    int pinfo[6] = {iris_r, iris_r, iris_w, sizeof(m), sizeof(n), sizeof(p)};
    task.kernel("hellinger", 2, nullptr, gws, lws, 6, params, pinfo);
    if (iter == repeat - 1) task.flush_out(mem_c);
    task.submit(iris_gpu, nullptr, 1);
  }

  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time " << (time * 1e-9f) / repeat
            << " (s)\n";

  std::cout << "Problem size: c(" << M << "," << P << ") = a(" << M << ","
            << N << ") * b(" << N << "," << P << ")\n";

#ifdef VERIFY
  VerifyResult(a_host, b_host, c_host, c_back);
#endif

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;
  delete[] c_back;

  platform.finalize();
  return 0;
}

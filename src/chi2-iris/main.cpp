#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

static void cpu_kernel(const unsigned int rows,
                       const unsigned int cols,
                       const int cRows,
                       const int contRows,
                       const unsigned char* snpdata,
                       float* results)
{
  for (unsigned int tid = 0; tid < cols; tid++) {
    int cases[3] = {1, 1, 1};
    int controls[3] = {1, 1, 1};
    for (int m = 0; m < cRows; m++) {
      unsigned char y = snpdata[static_cast<size_t>(m) * cols + tid];
      if (y == '0') cases[0]++;
      else if (y == '1') cases[1]++;
      else if (y == '2') cases[2]++;
    }
    for (int n = cRows; n < cRows + contRows; n++) {
      unsigned char y = snpdata[static_cast<size_t>(n) * cols + tid];
      if (y == '0') controls[0]++;
      else if (y == '1') controls[1]++;
      else if (y == '2') controls[2]++;
    }

    int tot_cases = cases[0] + cases[1] + cases[2];
    int tot_controls = controls[0] + controls[1] + controls[2];
    int total = tot_cases + tot_controls;
    float chisquare = 0.0f;
    for (unsigned int p = 0; p < 3; p++) {
      float exp = static_cast<float>(cases[p] + controls[p]);
      float Cexpected = tot_cases * exp / total;
      float Conexpected = tot_controls * exp / total;
      float numerator1 = static_cast<float>(cases[p]) - Cexpected;
      float numerator2 = static_cast<float>(controls[p]) - Conexpected;
      chisquare += numerator1 * numerator1 / Cexpected +
                   numerator2 * numerator2 / Conexpected;
    }
    results[tid] = chisquare;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 7) {
    std::printf("Usage: %s <rows> <cols> <cases> <controls> <threads> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  unsigned int rows = std::atoi(argv[1]);
  unsigned int cols = std::atoi(argv[2]);
  int ncases = std::atoi(argv[3]);
  int ncontrols = std::atoi(argv[4]);
  int nthreads = std::atoi(argv[5]);
  int repeat = std::atoi(argv[6]);

  std::printf("Individuals=%d SNPs=%d cases=%d controls=%d nthreads=%d\n",
              rows, cols, ncases, ncontrols, nthreads);

  size_t size = static_cast<size_t>(rows) * cols;
  std::printf("Size of the data = %lu\n", size);

  size_t snpdata_size = sizeof(unsigned char) * size;
  size_t result_size = sizeof(float) * cols;

  unsigned char* dataT = static_cast<unsigned char*>(std::malloc(snpdata_size));
  float* h_results = static_cast<float*>(std::malloc(result_size));
  float* cpu_results = static_cast<float*>(std::malloc(result_size));
  if (dataT == nullptr || h_results == nullptr || cpu_results == nullptr) {
    std::printf("ERROR: Memory for data not allocated.\n");
    std::free(dataT);
    std::free(h_results);
    std::free(cpu_results);
    platform.finalize();
    return 1;
  }

  std::mt19937 gen(19937);
  std::uniform_int_distribution<> distrib(0, 2);
  for (size_t i = 0; i < snpdata_size; i++) {
    dataT[i] = static_cast<unsigned char>(distrib(gen) + '0');
  }

  iris::DMem d_data(dataT, snpdata_size);
  iris::DMem d_results(h_results, result_size);

  unsigned int jobs = cols;
  size_t gws[1] = {static_cast<size_t>((jobs + nthreads - 1) / nthreads) *
                   static_cast<size_t>(nthreads)};
  size_t lws[1] = {static_cast<size_t>(nthreads)};

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[6] = {&rows, &cols, &ncases, &ncontrols, &d_data, &d_results};
    int pinfo[6] = {sizeof(rows), sizeof(cols), sizeof(ncases), sizeof(ncontrols), iris_r, iris_w};
    task.kernel("chi_kernel", 1, nullptr, gws, lws, 6, params, pinfo);
    task.flush_out(d_results);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average chi_kernel execution time = %f (s)\n", time * 1e-9f / repeat);

  d_results.fetch(h_results);

  start = std::chrono::high_resolution_clock::now();
  cpu_kernel(rows, cols, ncases, ncontrols, dataT, cpu_results);
  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Host execution time = %f (s)\n", time * 1e-9f);

  int error = 0;
  for (unsigned int k = 0; k < jobs; k++) {
    if (std::fabs(cpu_results[k] - h_results[k]) > 1e-4f) error++;
  }
  std::printf("%s\n", error ? "FAIL" : "PASS");

  std::free(dataT);
  std::free(h_results);
  std::free(cpu_results);
  platform.finalize();
  return error ? 1 : 0;
}

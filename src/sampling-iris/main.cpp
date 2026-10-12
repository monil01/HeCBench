#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Dataset {
  int nrows_exact;
  int nrows_sampled;
  int ncols;
  int nrows_background;
  int max_samples;
  uint64_t seed;
};

static bool verify_dataset(const Dataset& params,
                           const std::vector<int>& n,
                           const std::vector<float>& x,
                           const std::vector<float>& d,
                           float sent_value)
{
  const int nrows_x = params.nrows_exact + params.nrows_sampled;
  bool test_sampled_x = true;
  int j = 0;
  for (int i = params.nrows_exact * params.ncols; i < nrows_x * params.ncols / 2;
       i += 2 * params.ncols) {
    int counter = 0;
    for (int k = i; k < i + params.ncols; k++) {
      if (x[k] == 1.0f) counter++;
    }
    test_sampled_x = test_sampled_x && (counter == n[j]);

    counter = 0;
    for (int k = i + params.ncols; k < i + 2 * params.ncols; k++) {
      if (x[k] == 1.0f) counter++;
    }
    test_sampled_x = test_sampled_x && (counter == (params.ncols - n[j]));
    j++;
  }

  bool test_scatter_exact = true;
  for (int i = 0; i < params.nrows_exact; i++) {
    for (int row = i * params.nrows_background * params.ncols;
         row < (i + 1) * params.nrows_background * params.ncols;
         row += params.ncols) {
      int counter = 0;
      for (int k = row; k < row + params.ncols; k++) {
        if (d[k] == sent_value) counter++;
      }
      test_scatter_exact = test_scatter_exact && (counter == 2);
      if (!test_scatter_exact) {
        std::printf("test_scatter_exact counter failed with: %d, expected value was 2.\n", counter);
        break;
      }
    }
    if (!test_scatter_exact) break;
  }

  bool test_scatter_sampled = true;
  int compliment_ctr = 0;
  for (int i = params.nrows_exact; i < params.nrows_exact + params.nrows_sampled / 2; i++) {
    for (int row = (i + compliment_ctr) * params.nrows_background * params.ncols;
         row < (i + compliment_ctr + 1) * params.nrows_background * params.ncols;
         row += params.ncols) {
      int counter = 0;
      for (int k = row; k < row + params.ncols; k++) {
        if (d[k] == sent_value) counter++;
      }
      test_scatter_sampled = test_scatter_sampled && (counter == n[i - params.nrows_exact]);
      if (!test_scatter_sampled) {
        std::printf("test_scatter_sampled counter failed with: %d, expected value was %d.\n",
                    counter, n[i - params.nrows_exact]);
        break;
      }
    }

    compliment_ctr++;
    for (int row = (i + compliment_ctr) * params.nrows_background * params.ncols;
         row < (i + compliment_ctr + 1) * params.nrows_background * params.ncols;
         row += params.ncols) {
      int counter = 0;
      for (int k = row; k < row + params.ncols; k++) {
        if (d[k] == sent_value) counter++;
      }
      test_scatter_sampled = test_scatter_sampled &&
                             (counter == params.ncols - n[i - params.nrows_exact]);
      if (!test_scatter_sampled) {
        std::printf("test_scatter_sampled counter failed with: %d, expected value was %d.\n",
                    counter, params.ncols - n[i - params.nrows_exact]);
        break;
      }
    }
  }

  return test_sampled_x && test_scatter_exact && test_scatter_sampled;
}

static void launch_exact(iris::DMem& mem_x,
                         iris::DMem& mem_background,
                         iris::DMem& mem_dataset,
                         iris::DMem& mem_observation,
                         int nrows_x,
                         int ncols,
                         int nrows_background)
{
  const int nblocks = nrows_x;
  const int nthreads = std::min(256, ncols);
  size_t gws[1] = {static_cast<size_t>(nblocks * nthreads)};
  size_t lws[1] = {static_cast<size_t>(nthreads)};
  int x_base = 0;
  int dataset_base = 0;
  void* params[] = {
      &mem_x, &x_base, &nrows_x, &ncols, &mem_background, &nrows_background,
      &mem_dataset, &dataset_base, &mem_observation};
  int pinfo[] = {iris_rw, sizeof(x_base), sizeof(nrows_x), sizeof(ncols), iris_r,
                 sizeof(nrows_background), iris_w, sizeof(dataset_base), iris_r};
  iris::Task task;
  task.kernel("exact_rows_kernel", 1, nullptr, gws, lws, 9, params, pinfo);
  task.flush_out(mem_dataset);
  task.submit(iris_gpu, nullptr, 1);
}

static void launch_sampled(iris::DMem& mem_nsamples,
                           iris::DMem& mem_x,
                           iris::DMem& mem_background,
                           iris::DMem& mem_dataset,
                           iris::DMem& mem_observation,
                           int nrows_x,
                           int ncols,
                           int nrows_background,
                           int len_samples,
                           uint64_t seed)
{
  const int nblocks = len_samples / 2;
  const int nthreads = std::min(256, ncols);
  size_t gws[1] = {static_cast<size_t>(nblocks * nthreads)};
  size_t lws[1] = {static_cast<size_t>(nthreads)};
  int x_base = (nrows_x - len_samples) * ncols;
  int dataset_base = (nrows_x - len_samples) * nrows_background * ncols;
  void* params[] = {&mem_nsamples, &mem_x, &x_base, &len_samples, &ncols,
                    &mem_background, &nrows_background, &mem_dataset,
                    &dataset_base, &mem_observation, &seed};
  int pinfo[] = {iris_r, iris_rw, sizeof(x_base), sizeof(len_samples), sizeof(ncols),
                 iris_r, sizeof(nrows_background), iris_rw, sizeof(dataset_base),
                 iris_r, sizeof(seed)};
  iris::Task task;
  task.kernel("sampled_rows_kernel", 1, nullptr, gws, lws, 11, params, pinfo);
  task.flush_out(mem_x);
  task.flush_out(mem_dataset);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = std::atoi(argv[1]);
  const std::vector<Dataset> inputs = {
      {1000, 0, 2000, 10, 11, 1234ULL},
      {0, 1000, 2000, 10, 11, 1234ULL},
      {1000, 1000, 2000, 10, 11, 1234ULL},
  };

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  bool all_ok = true;
  for (const Dataset& params : inputs) {
    double time_ns = 0.0;
    bool params_ok = true;

    for (int r = 0; r < repeat; r++) {
      const int nrows_x = params.nrows_exact + params.nrows_sampled;
      const size_t background_count = static_cast<size_t>(params.nrows_background) * params.ncols;
      const size_t x_count = static_cast<size_t>(nrows_x) * params.ncols;
      const size_t dataset_count = x_count * params.nrows_background;

      std::vector<float> background(background_count);
      std::vector<float> observation(params.ncols);
      std::vector<int> nsamples(std::max(1, params.nrows_sampled / 2), 0);
      std::vector<float> x(x_count, 0.0f);
      std::vector<float> dataset(dataset_count, 0.0f);

      const float sent_value = static_cast<float>(dataset_count * 100);
      std::fill(observation.begin(), observation.end(), sent_value);
      for (int i = 0; i < params.nrows_background; i++) {
        for (int j = 0; j < params.ncols; j++) {
          background[static_cast<size_t>(i) * params.ncols + j] = static_cast<float>((i * 2) + 1);
        }
      }
      for (int i = 0; i < params.nrows_exact; i++) {
        for (int j = i; j < i + 2; j++) {
          x[static_cast<size_t>(i) * params.ncols + j] = 1.0f;
        }
      }
      for (int i = 0; i < params.nrows_sampled / 2; i++) {
        nsamples[i] = params.max_samples - i % 2;
      }

      iris::DMem mem_background(background.data(), background.size() * sizeof(float));
      iris::DMem mem_observation(observation.data(), observation.size() * sizeof(float));
      iris::DMem mem_nsamples(nsamples.data(), nsamples.size() * sizeof(int));
      iris::DMem mem_x(x.data(), x.size() * sizeof(float));
      iris::DMem mem_dataset(dataset.data(), dataset.size() * sizeof(float));

      auto start = std::chrono::steady_clock::now();
      if (nrows_x - params.nrows_sampled > 0) {
        launch_exact(mem_x, mem_background, mem_dataset, mem_observation,
                     nrows_x - params.nrows_sampled, params.ncols, params.nrows_background);
      }
      if (params.nrows_sampled > 0) {
        launch_sampled(mem_nsamples, mem_x, mem_background, mem_dataset, mem_observation,
                       nrows_x, params.ncols, params.nrows_background,
                       params.nrows_sampled, params.seed);
      }
      auto end = std::chrono::steady_clock::now();
      time_ns += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

      mem_x.fetch(x.data());
      mem_dataset.fetch(dataset.data());
      params_ok = params_ok && verify_dataset(params, nsamples, x, dataset, sent_value);
    }

    std::printf("Average execution time of kernels: %f (us)\n", (time_ns * 1e-3) / repeat);
    std::printf("%s\n", params_ok ? "PASS" : "FAIL");
    all_ok = all_ok && params_ok;
  }

  platform.finalize();
  return all_ok ? 0 : 2;
}

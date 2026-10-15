#include <iris/iris.hpp>
#include <hip/hip_runtime_api.h>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

static void check(int rc) {
  if (rc != IRIS_SUCCESS) { std::fprintf(stderr, "IRIS error %d\n", rc); std::exit(2); }
}

int main(int argc, char** argv) {
  if (argc != 2) { std::printf("Usage: %s <number of concurrent kernels>\n", argv[0]); return 1; }
  char* end = nullptr;
  const long count = std::strtol(argv[1], &end, 10);
  if (*end || count < 1 || count > 4096) return 1;
  int n = static_cast<int>(count), rate = 0;
  // A device-property query only: all memory and execution are managed by IRIS.
  if (hipDeviceGetAttribute(&rate, hipDeviceAttributeClockRate, 0) != hipSuccess) return 2;
  long clocks = static_cast<long>(20.0f * rate);
  if (clocks > INT_MAX) return 2; // Native loop induction variable is int.
  setenv("IRIS_ASYNC", "1", 1);
  setenv("IRIS_ARCHS", "HIP", 1);
  std::string streams = std::to_string(n + 1);
  setenv("IRIS_NSTREAMS", streams.c_str(), 1);
  setenv("IRIS_NCOPY_STREAMS", "1", 1);
  iris::Platform platform;
  check(platform.init(&argc, &argv, 1));
  check(iris_set_stream_policy(STREAM_POLICY_SAME_FOR_TASK));
  bool pass = false;
  {
    std::vector<long> values(n, -1);
    const bool trace = std::getenv("HECBENCH_CONCURRENT_TRACE") != nullptr;
    std::vector<unsigned long long> intervals(2 * n, 0);
    size_t off[] = {0}, shape[] = {static_cast<size_t>(n)};
    iris::DMem data(values.data(), off, shape, shape, sizeof(long), 1);
    iris::DMem timing(intervals.data(), off, shape, shape, 2 * sizeof(unsigned long long), 1);
    check(data.enable_outer_dim_regions());
    check(timing.enable_outer_dim_regions());
    // Materialize the shared allocation before launching independent writers.
    iris::Task init;
    check(init.h2d(&data));
    check(init.submit(0, nullptr, true));
    std::vector<std::unique_ptr<iris::DMemRegion>> regions;
    std::vector<std::unique_ptr<iris::DMemRegion>> timing_regions;
    std::vector<std::unique_ptr<iris::Task>> producers;
    size_t one[] = {1}, warp[] = {32};
    for (int i = 0; i < n; ++i) {
      regions.emplace_back(new iris::DMemRegion(&data, i));
      timing_regions.emplace_back(new iris::DMemRegion(&timing, i));
      producers.emplace_back(new iris::Task("clock_block", false, true));
      check(iris_task_set_stream_policy(producers.back()->task(), STREAM_POLICY_SAME_FOR_TASK));
      void* params[] = {regions.back().get(), &clocks, timing_regions.back().get()};
      int info[] = {iris_w, sizeof(clocks), iris_w};
      check(producers.back()->kernel(trace ? "clock_block_trace" : "clock_block", 1,
                                    nullptr, one, one, trace ? 3 : 2, params, info));
    }
    iris::Task reduction("sum");
    for (auto& p : producers) reduction.depends_on(*p);
    void* params[] = {&data, &n};
    int info[] = {iris_rw, sizeof(n)};
    check(reduction.kernel("sum", 1, nullptr, warp, warp, 2, params, info));
    check(reduction.flush_out(data));
    if (trace) {
      // Join the independently-written timing regions before fetching their
      // parent, just as the reduction joins the arithmetic output regions.
      check(reduction.add_hidden_dmem(timing, iris_rw));
      check(reduction.flush_out(timing));
    }
    std::printf("time clocks = %ld\n", clocks);
    const auto start = std::chrono::steady_clock::now();
    for (auto& p : producers) check(p->submit(0, nullptr, false));
    check(reduction.submit(0, nullptr, true));
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (trace) check(timing.fetch(intervals.data()));
    const long expected_one = (clocks / 3) * 3 + ((clocks % 3) == 2 ? 1 : 0);
    pass = values[0] == n * expected_one;
    for (int i = 1; i < n; ++i) pass = pass && values[i] == expected_one;
    if (trace) {
      int overlapping_pairs = 0;
      for (int i = 0; i < n; ++i) {
        std::printf("interval[%d]=%llu,%llu\n", i, intervals[2*i], intervals[2*i+1]);
        pass = pass && intervals[2*i] < intervals[2*i+1];
        for (int j = 0; j < i; ++j)
          if (intervals[2*i] < intervals[2*j+1] && intervals[2*j] < intervals[2*i+1]) ++overlapping_pairs;
      }
      std::printf("GPU wall-clock overlapping producer pairs: %d\n", overlapping_pairs);
      if (n > 1) pass = pass && overlapping_pairs > 0;
    }
    std::printf("Measured time for sample = %.3fs\n", seconds);
    std::printf("sum=%ld expected=%ld; all %d producer/reduction outputs checked\n", values[0], n * expected_one, n);
    std::printf("%s\n", pass ? "PASS" : "FAIL");
  }
  check(platform.finalize());
  return pass && iris_error_count() == 0 ? 0 : 1;
}

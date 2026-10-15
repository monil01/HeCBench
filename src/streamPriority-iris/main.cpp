#include <iris/iris.hpp>
#include <hip/hip_runtime_api.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include "reference.h"

#ifndef HECBENCH_STREAM_PRIORITY_N
#define HECBENCH_STREAM_PRIORITY_N (1 << 24)
#endif
static void check(int rc) {
  if (rc != IRIS_SUCCESS || iris_error_count()) {
    std::fprintf(stderr, "IRIS stream error %d\n", rc); std::exit(2);
  }
}
struct Outputs {
  std::vector<float> act, grad, weights, prefetch;
  explicit Outputs(int n) : act(n), grad(n), weights(n, 1), prefetch(n, 0) {}
};
static bool compare(const char* name, const std::vector<float>& actual,
                    const std::vector<float>& expected) {
  double max_error = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(static_cast<double>(actual[i]) - expected[i]);
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || error > 1e-3) {
      std::fprintf(stderr, "FAIL %s[%zu]: %.9g expected %.9g\n", name, i, actual[i], expected[i]);
      return false;
    }
    if (error > max_error) max_error = error;
  }
  std::printf("checked %s: %zu values, maxabs %.9g\n", name, actual.size(), max_error);
  return true;
}
static void dump(const char* mode, const Outputs& output) {
  const char* prefix = std::getenv("HECBENCH_STREAM_PRIORITY_DUMP");
  if (!prefix) return;
  const std::string path = std::string(prefix) + "." + mode + ".bin";
  FILE* file = std::fopen(path.c_str(), "wb");
  if (!file) std::exit(2);
  for (const auto* array : {&output.act, &output.grad, &output.weights, &output.prefetch})
    if (std::fwrite(array->data(), sizeof(float), array->size(), file) != array->size()) std::exit(2);
  if (std::fclose(file)) std::exit(2);
}
static bool run(int n, int batches, bool priority, const Outputs& reference) {
  Outputs output(n);
  std::vector<float> grad_in(n, 1);
  iris::DMem prefetch(output.prefetch.data(), n*sizeof(float));
  iris::DMem weights(output.weights.data(), n*sizeof(float));
  iris::DMem grad_input(grad_in.data(), n*sizeof(float));
  iris::DMem act(output.act.data(), n*sizeof(float));
  iris::DMem grad(output.grad.data(), n*sizeof(float));
  const int train_stream = priority ? 1 : 3;
  const int prefetch_stream = priority ? 2 : 3;
  size_t local[] = {256}, global[] = {(static_cast<size_t>(n)+255)/256*256};
  std::vector<std::unique_ptr<iris::Task>> tasks;
  iris::Task* previous_update = nullptr;
  iris::Task* previous_prefetch = nullptr;
  auto create = [&](const char* name, int stream) {
    tasks.emplace_back(new iris::Task(name, false, true));
    check(iris_task_set_stream_index(tasks.back()->task(), stream));
    return tasks.back().get();
  };
  const auto start = std::chrono::steady_clock::now();
  for (int b = 0; b < batches; ++b) {
    iris::Task* forward = create("forward", train_stream);
    if (previous_prefetch) forward->depends_on(*previous_prefetch);
    if (previous_update) forward->depends_on(*previous_update);
    float scale = 1.0f;
    void* fargs[] = {&prefetch, &act, &n, &scale};
    int finfo[] = {iris_r, iris_w, sizeof(n), sizeof(scale)};
    check(forward->kernel("forward_pass_kernel", 1, nullptr, global, local, 4, fargs, finfo));
    check(forward->submit(0, nullptr, false));

    iris::Task* backward = create("backward", train_stream);
    backward->depends_on(*forward);
    void* bargs[] = {&act, &grad_input, &grad, &n};
    int binfo[] = {iris_r, iris_r, iris_w, sizeof(n)};
    check(backward->kernel("backward_pass_kernel", 1, nullptr, global, local, 4, bargs, binfo));
    check(backward->submit(0, nullptr, false));

    iris::Task* update = create("update", train_stream);
    update->depends_on(*backward);
    float rate = 0.01f;
    void* uargs[] = {&weights, &grad, &n, &rate};
    int uinfo[] = {iris_rw, iris_r, sizeof(n), sizeof(rate)};
    check(update->kernel("sgd_update_kernel", 1, nullptr, global, local, 4, uargs, uinfo));
    check(update->submit(0, nullptr, false));

    iris::Task* fetch = create("prefetch", prefetch_stream);
    // Native events permit background prefetch after forward, in parallel
    // with backward/update. The baseline is intentionally one stream.
    fetch->depends_on(priority ? *forward : *update);
    int batch_id = b + 1;
    void* pargs[] = {&prefetch, &n, &batch_id};
    int pinfo[] = {iris_w, sizeof(n), sizeof(batch_id)};
    check(fetch->kernel("data_prefetch_kernel", 1, nullptr, global, local, 3, pargs, pinfo));
    check(fetch->submit(0, nullptr, false));
    previous_update = update;
    previous_prefetch = fetch;
  }
  check(previous_update->wait());
  check(previous_prefetch->wait());
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
  check(act.fetch(output.act.data()));
  check(grad.fetch(output.grad.data()));
  check(weights.fetch(output.weights.data()));
  check(prefetch.fetch(output.prefetch.data()));
  const char* mode = priority ? "priority" : "baseline";
  bool ok = compare("act", output.act, reference.act);
  ok = compare("grad_out", output.grad, reference.grad) && ok;
  ok = compare("weights", output.weights, reference.weights) && ok;
  ok = compare("prefetch", output.prefetch, reference.prefetch) && ok;
  dump(mode, output);
  std::printf("%s streams: %.3f ms; %s; all %d output floats checked\n", mode, ms, ok ? "PASS" : "FAIL", 4*n);
  for (auto& task : tasks) check(iris_task_release(task->task()));
  return ok;
}
int main(int argc, char** argv) {
  if (argc != 2 || std::atoi(argv[1]) <= 0) { std::printf("Usage: %s <repeat>\n", argv[0]); return 1; }
  const int batches = std::atoi(argv[1]), n = HECBENCH_STREAM_PRIORITY_N;
  if (n <= 0) return 1;
  Outputs reference(n);
  std::vector<float> grad_in(n, 1);
  for (int b = 0; b < batches; ++b) {
    forward_pass(reference.prefetch.data(), reference.act.data(), n, 1.0f);
    backward_pass(reference.act.data(), grad_in.data(), reference.grad.data(), n);
    sgd_update(reference.weights.data(), reference.grad.data(), n, 0.01f);
    data_prefetch(reference.prefetch.data(), n, b+1);
  }
  setenv("IRIS_ARCHS", "HIP", 1);
  setenv("IRIS_ASYNC", "1", 1);
  setenv("IRIS_NSTREAMS", "4", 1);
  setenv("IRIS_NCOPY_STREAMS", "1", 1);
  iris::Platform platform;
  check(platform.init(&argc, &argv, 1));
  // Validate bounds before any submitted work can retain a stream handle.
  {
    iris::Task guard("stream_api_bounds", false, true);
    if (iris_task_set_stream_index(guard.task(), -1) != IRIS_ERROR ||
        iris_task_set_stream_index(guard.task(), 4) != IRIS_ERROR ||
        iris_hip_set_stream_priority(-1, 1, 0) != IRIS_ERROR ||
        iris_hip_set_stream_priority(0, -1, 0) != IRIS_ERROR) return 2;
    check(iris_task_release(guard.task()));
  }
  int least = 0, greatest = 0;
  if (hipDeviceGetStreamPriorityRange(&least, &greatest) != hipSuccess) return 2;
  check(iris_hip_set_stream_priority(0, 1, greatest));
  check(iris_hip_set_stream_priority(0, 2, least));
  int actual_high = 0, actual_low = 0, actual_baseline = 0;
  if (hipStreamGetPriority(static_cast<hipStream_t>(iris_dev_get_stream(0, 1)), &actual_high) != hipSuccess ||
      hipStreamGetPriority(static_cast<hipStream_t>(iris_dev_get_stream(0, 2)), &actual_low) != hipSuccess ||
      hipStreamGetPriority(static_cast<hipStream_t>(iris_dev_get_stream(0, 3)), &actual_baseline) != hipSuccess ||
      actual_high != greatest || actual_low != least) return 2;
  std::printf("IRIS HIP stream priorities: baseline=%d train=%d prefetch=%d; supported range=%d..%d\n", actual_baseline, actual_high, actual_low, greatest, least);
  const bool baseline = run(n, batches, false, reference);
  const bool priority = run(n, batches, true, reference);
  check(platform.finalize());
  return baseline && priority && !iris_error_count() ? 0 : 1;
}

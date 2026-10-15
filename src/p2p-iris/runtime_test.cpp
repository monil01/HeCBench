// Regression: bindings belong to a Command, not a task's first kernel.
#include <iris/iris.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
static void ok(int rc) { if (rc != IRIS_SUCCESS || iris_error_count()) std::exit(2); }
static void reject(int rc) { if (rc != IRIS_ERROR || iris_error_count()) std::exit(2); }
int main(int argc, char** argv) {
  setenv("IRIS_ARCHS", "HIP", 1);
  setenv("HSA_FORCE_FINE_GRAIN_PCIE", "1", 0);
  iris::Platform p; ok(p.init(&argc, &argv, 1)); ok(iris_init_devices(1));
  ok(iris_hip_enable_peer_access(1, 0));
  const size_t n = 263;
  std::vector<float> a(n), b(n, -1), c(n, -2);
  for (size_t i = 0; i < n; ++i) a[i] = float(i + 1);
  {
    iris::DMem A(a.data(), n * sizeof(float)), B(b.data(), n * sizeof(float)), C(c.data(), n * sizeof(float));
    iris::Task ua, ub;
    ok(ua.h2d(&A)); ok(ua.submit(0, nullptr, true));
    ok(ub.h2d(&B)); ok(ub.h2d(&C)); ok(ub.submit(1, nullptr, true));
    iris::Task chain("peer_invalid_composition", false, true);
    size_t global[] = {512}, local_size[] = {256};
    void* first[] = {&A, &B, const_cast<size_t*>(&n)};
    int info[] = {iris_r, iris_w, sizeof(n)};
    ok(chain.kernel("SimpleKernel", 1, nullptr, global, local_size, 3, first, info));
    ok(iris_task_bind_hip_remote_dmem(chain.task(), 0, 0));
    // Added AFTER binding: rejection must happen at submission as well as
    // binding, otherwise task composition can bypass the single-kernel rule.
    void* second[] = {&B, &C, const_cast<size_t*>(&n)};
    ok(chain.kernel("SimpleKernel", 1, nullptr, global, local_size, 3, second, info));
    reject(iris_task_bind_hip_remote_dmem(chain.task(), 0, 0));
    std::fprintf(stderr, "PASS binding rejects multi-kernel task\n");
    reject(chain.submit(1, nullptr, true));
    std::fprintf(stderr, "PASS task submission rejects appended kernel\n");
    iris_graph absent;
    ok(iris_graph_create_null(&absent));
    reject(iris_graph_task(absent, chain.task(), 1, nullptr));
    std::fprintf(stderr, "PASS null-graph immediate submission propagates rejection\n");
    iris_graph graph;
    ok(iris_graph_create(&graph));
    ok(iris_graph_task(graph, chain.task(), 1, nullptr));
    reject(iris_graph_submit(graph, 1, 1));
    std::fprintf(stderr, "PASS graph submission rejects appended kernel\n");
    int order[] = {1, 0};
    reject(iris_graph_submit_with_order(graph, order, 1, 1));
    std::fprintf(stderr, "PASS ordered graph submission rejects appended kernel\n");
    // Rejected graphs were never submitted. Free the container directly;
    // upstream GraphRelease's object-track callback deadlocks on this case.
    int count = iris_graph_tasks_count(graph);
    if (count != 2) return 1;
    iris_task tasks[2]; if (iris_graph_get_tasks(graph, tasks) != count) return 1;
    ok(iris_graph_free(graph));
    for (int i = count - 1; i >= 0; --i) ok(iris_task_release(tasks[i]));
    if (iris_dmem_is_allocated_on_device(A.mem(), 1) != 0 ||
        iris_dmem_is_allocated_on_device(B.mem(), 0) != 0 ||
        iris_dmem_is_allocated_on_device(C.mem(), 0) != 0) return 1;
  }
  ok(p.finalize());
  std::puts("PASS remote single-kernel contract: binding and task/graph/ordered-graph submission rejected before execution; no wrong-device allocations; iris_errors0");
  return 0;
}

// Shared-view ABI/ownership regression. HIP is used for metadata queries only.
#include <iris/iris_runtime.h>
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdint>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "failed: %s\n", #x); return 1; } } while (0)
int main(int argc, char** argv) {
  REQUIRE(iris_init(&argc, &argv, 1) == IRIS_SUCCESS);
  void* view = reinterpret_cast<void*>(uintptr_t(1));
  iris_mem ordinary;
  REQUIRE(iris_data_mem_create(&ordinary, nullptr, 10 * sizeof(int)) == IRIS_SUCCESS);
  REQUIRE(iris_dmem_bind_hip_shared_view(ordinary, 0, &view) == IRIS_ERROR);
  REQUIRE(view == nullptr);
  REQUIRE(iris_dmem_set_hip_allocation_kind(ordinary, 1) == IRIS_SUCCESS);
  REQUIRE(iris_dmem_bind_hip_shared_view(ordinary, 0, &view) == IRIS_ERROR);
  REQUIRE(iris_mem_release(ordinary) == IRIS_SUCCESS);
  int source[10] = {};
  iris_mem backed;
  REQUIRE(iris_data_mem_create(&backed, source, sizeof(source)) == IRIS_SUCCESS);
  REQUIRE(iris_dmem_set_hip_allocation_kind(backed, 2) == IRIS_SUCCESS);
  REQUIRE(iris_dmem_bind_hip_shared_view(backed, 0, &view) == IRIS_ERROR);
  REQUIRE(iris_mem_release(backed) == IRIS_SUCCESS);
  iris_mem usm;
  REQUIRE(iris_data_mem_create(&usm, nullptr, sizeof(source)) == IRIS_SUCCESS);
  REQUIRE(iris_dmem_set_hip_allocation_kind(usm, 2) == IRIS_SUCCESS);
  REQUIRE(iris_mem_enable_usm_all(usm) == IRIS_SUCCESS);
  REQUIRE(iris_dmem_bind_hip_shared_view(usm, 0, &view) == IRIS_ERROR);
  REQUIRE(iris_mem_release(usm) == IRIS_SUCCESS);
  iris_mem materialized;
  REQUIRE(iris_data_mem_create(&materialized, nullptr, sizeof(source)) == IRIS_SUCCESS);
  REQUIRE(iris_dmem_set_hip_allocation_kind(materialized, 2) == IRIS_SUCCESS);
  REQUIRE(iris_mem_arch(materialized, 0, &view) == IRIS_SUCCESS);
  REQUIRE(iris_dmem_bind_hip_shared_view(materialized, 0, &view) == IRIS_ERROR);
  REQUIRE(iris_mem_release(materialized) == IRIS_SUCCESS);
  iris_mem regions;
  REQUIRE(iris_data_mem_create(&regions, nullptr, sizeof(source)) == IRIS_SUCCESS);
  REQUIRE(iris_dmem_set_hip_allocation_kind(regions, 2) == IRIS_SUCCESS);
  REQUIRE(iris_data_mem_enable_outer_dim_regions(regions) == IRIS_SUCCESS);
  REQUIRE(iris_dmem_bind_hip_shared_view(regions, 0, &view) == IRIS_ERROR);
  REQUIRE(iris_mem_release(regions) == IRIS_SUCCESS);
  iris_mem multidimensional;
  size_t shape[] = {2, 5};
  REQUIRE(iris_data_mem_create_nd(&multidimensional, nullptr, shape, 2, sizeof(int), iris_int) == IRIS_SUCCESS);
  REQUIRE(iris_dmem_set_hip_allocation_kind(multidimensional, 2) == IRIS_SUCCESS);
  REQUIRE(iris_dmem_bind_hip_shared_view(multidimensional, 0, &view) == IRIS_ERROR);
  REQUIRE(iris_mem_release(multidimensional) == IRIS_SUCCESS);
  for (int repetition = 0; repetition < 8; ++repetition) {
    iris_mem mem;
    REQUIRE(iris_data_mem_create(&mem, nullptr, 10 * sizeof(int)) == IRIS_SUCCESS);
    REQUIRE(iris_dmem_set_hip_allocation_kind(mem, 2) == IRIS_SUCCESS);
    REQUIRE(iris_dmem_bind_hip_shared_view(mem, -1, &view) == IRIS_ERROR);
    REQUIRE(iris_dmem_bind_hip_shared_view(mem, 0, nullptr) == IRIS_ERROR);
    REQUIRE(iris_dmem_bind_hip_shared_view(mem, 0, &view) == IRIS_SUCCESS);
    auto* p = static_cast<int*>(view);
    hipPointerAttribute_t attr{};
    REQUIRE(hipPointerGetAttributes(&attr, p) == hipSuccess);
    REQUIRE(attr.type == hipMemoryTypeHost && !attr.isManaged);
    REQUIRE((attr.allocationFlags & hipHostMallocCoherent) != 0);
    void* physical = nullptr;
    REQUIRE(iris_mem_arch(mem, 0, &physical) == IRIS_SUCCESS && physical == p);
    REQUIRE(iris_dmem_bind_hip_shared_view(mem, 0, &view) == IRIS_ERROR);
    REQUIRE(iris_dmem_set_hip_allocation_kind(mem, 0) == IRIS_ERROR);
    for (int k = 0; k < 10; ++k) p[k] = 0;
    p[7] = p[9] = 255;
    p[3] = 256; // Prove GPU minimum changes the value, not just a no-op.
    int unused[80] = {};
    iris_mem progress;
    REQUIRE(iris_data_mem_create(&progress, unused, sizeof(unused)) == IRIS_SUCCESS);
    int loops = 1;
    size_t global[] = {256}, local[] = {256};
    void* args[] = {&mem, &loops, &progress};
    int info[] = {iris_rw, sizeof(loops), iris_r};
    iris_task task;
    REQUIRE(iris_task_create(&task) == IRIS_SUCCESS);
    REQUIRE(iris_task_kernel(task, "atomicKernelUntraced", 1, nullptr, global, local, 3, args, info) == IRIS_SUCCESS);
    REQUIRE(iris_task_submit(task, 0, nullptr, 1) == IRIS_SUCCESS);
    REQUIRE(p[0] == 2560 && p[1] >= 0 && p[1] < 256 && p[2] == 255 && p[3] == 0);
    REQUIRE(p[4] == 0 && p[5] == 0 && p[6] >= 0 && p[6] < 256);
    REQUIRE(p[7] == 1 && p[8] == -1 && p[9] == 255);
    REQUIRE(iris_task_release(task) == IRIS_SUCCESS);
    REQUIRE(iris_mem_release(progress) == IRIS_SUCCESS);
    REQUIRE(iris_mem_release(mem) == IRIS_SUCCESS);
    // This is not a free by the application: prove DMem performed hipHostFree.
    const hipError_t query = hipPointerGetAttributes(&attr, p);
    printf("after DMem release query=%d type=%d flags=%u\n", int(query), int(attr.type), attr.allocationFlags);
    // HIP may successfully classify an address as unregistered after free.
    REQUIRE(query != hipSuccess || attr.type == hipMemoryTypeUnregistered);
    const hipError_t cleared = hipGetLastError();
    (void)cleared;
  }
  const int errors = iris_error_count();
  REQUIRE(iris_finalize() == IRIS_SUCCESS);
  REQUIRE(errors == 0);
  printf("shared-view ABI/ownership PASS: 80 counters, 8 DMem-owned frees, invalid controls rejected, iris_errors=0\n");
}

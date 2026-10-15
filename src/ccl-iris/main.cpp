#include <iris/iris.hpp>
#include "../../tools/iris_vendor_bridge.hpp"
#include <hip/hip_runtime_api.h>
#include <rccl/rccl.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool ok, const char* operation) {
  if (!ok) { std::fprintf(stderr, "%s failed\n", operation); MPI_Abort(MPI_COMM_WORLD, 1); }
}
static int allreduce(void* stream, unsigned, void* const* args) {
  const auto comm = hecbench_vendor_arg<ncclComm_t>(args, 1);
  const auto count = hecbench_vendor_arg<size_t>(args, 2);
  const auto repeat = hecbench_vendor_arg<int>(args, 3);
  const auto input = hecbench_vendor_arg<const float*>(args, 4);
  const auto output = hecbench_vendor_arg<float*>(args, 5);
  for (int i = 0; i < repeat; ++i) {
    const auto status = ncclAllReduce(input, output, count, ncclFloat, ncclSum, comm,
                                     static_cast<hipStream_t>(stream));
    if (status != ncclSuccess) return static_cast<int>(status);
  }
  return static_cast<int>(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
}
int main(int argc, char** argv) {
  if (argc != 2 || std::atoi(argv[1]) <= 0) { std::printf("Usage: %s <repeat>\n", argv[0]); return 1; }
  int repeat = std::atoi(argv[1]);
  MPI_Init(&argc, &argv);
  int rank, world, local_rank, ngpus;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm local;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
  MPI_Comm_rank(local, &local_rank); MPI_Comm_free(&local);
  require(hipGetDeviceCount(&ngpus) == hipSuccess && local_rank < ngpus, "GPU selection");
  setenv("IRIS_ARCHS", "HIP", 1);
  setenv("IRIS_KERNEL_HOST2HIP", "./libhecbench_vendor.so", 1);
  iris::Platform platform;
  require(platform.init(&argc, &argv, 1) == IRIS_SUCCESS, "IRIS init");
  require(hipSetDevice(local_rank) == hipSuccess, "HIP context selection");
  ncclUniqueId id;
  if (rank == 0) require(ncclGetUniqueId(&id) == ncclSuccess, "RCCL unique ID");
  require(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD) == MPI_SUCCESS, "ID broadcast");
  ncclComm_t comm;
  require(ncclCommInitRank(&comm, world, id, rank) == ncclSuccess, "RCCL communicator");
  const char* max_env = std::getenv("HECBENCH_CCL_MAX_ELEMENTS");
  const long limit = max_env ? std::atol(max_env) : 1000L * 1024 * 1024;
  require(limit >= 1024*1024 && limit <= 1000L*1024*1024, "validation limit");
  bool all_ok = true;
  const bool patterned = std::getenv("HECBENCH_CCL_PATTERNED") != nullptr;
  for (size_t count = 1024*1024; count <= static_cast<size_t>(limit); count *= 10) {
    std::vector<float> input(count, 1.0f), output(count, -1.0f);
    if (patterned) for (size_t i = 0; i < count; ++i) input[i] = float(static_cast<int>(i % 97) - 48 + rank);
    iris::DMem send(input.data(), count*sizeof(float)), recv(output.data(), count*sizeof(float));
    iris::Task task;
    hecbench_vendor_callback callback = allreduce;
    void* args[] = {&callback, &comm, &count, &repeat, &send, &recv};
    int info[] = {sizeof(callback), sizeof(comm), sizeof(count), sizeof(repeat), iris_r, iris_w};
    size_t gws[] = {1}, lws[] = {1};
    require(task.kernel("hecbench_vendor", 1, nullptr, gws, lws, 6, args, info) == IRIS_SUCCESS, "RCCL task");
    require(task.flush_out(recv) == IRIS_SUCCESS, "output fetch");
    const double start = MPI_Wtime();
    require(task.submit(local_rank, nullptr, true) == IRIS_SUCCESS, "RCCL submit");
    const double seconds = MPI_Wtime() - start;
    bool ok = true;
    for (size_t i = 0; i < count; ++i) {
      const float expected = patterned ? float(world * (static_cast<int>(i % 97) - 48) + world * (world - 1) / 2) : float(world);
      if (output[i] != expected) { ok = false; break; }
    }
    all_ok = all_ok && ok;
    std::printf("MPI Rank %d: %s; all %zu float outputs checked; %.9f s/iteration\n",
                rank, ok ? "PASS" : "FAIL", count, seconds/repeat);
  }
  require(ncclCommFinalize(comm) == ncclSuccess, "RCCL finalize");
  require(ncclCommDestroy(comm) == ncclSuccess, "RCCL destroy");
  all_ok = all_ok && iris_error_count() == 0;
  require(platform.finalize() == IRIS_SUCCESS, "IRIS finalize");
  MPI_Finalize();
  return all_ok ? 0 : 1;
}

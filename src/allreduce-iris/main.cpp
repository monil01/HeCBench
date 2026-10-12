#include <iris/iris.hpp>

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_mpi(int status, const char* operation) {
  if (status != MPI_SUCCESS) throw std::runtime_error(operation);
}

void reduce_segment(iris::DMem& output_mem, iris::DMem& buffer_mem,
                    float* output, float* buffer, size_t offset, size_t length) {
  buffer_mem.update(buffer);
  output_mem.update(output);

  size_t lws[1] = {256};
  size_t gws[1] = {std::max<size_t>(lws[0],
                                    ((length + lws[0] - 1) / lws[0]) * lws[0])};
  void* params[4] = {&buffer_mem, &output_mem, &offset, &length};
  int pinfo[4] = {iris_r, iris_rw, static_cast<int>(sizeof(offset)),
                  static_cast<int>(sizeof(length))};

  iris::Task task;
  task.kernel("kernel_add_segment", 1, nullptr, gws, lws, 4, params, pinfo);
  task.flush_out(output_mem);
  if (task.submit(iris_gpu, nullptr, 1) != IRIS_SUCCESS)
    throw std::runtime_error("IRIS reduction task submission failed");
}

void ring_allreduce(iris::Platform& platform, float* input, size_t length,
                    float* output, double* elapsed_seconds) {
  int rank = 0;
  int world = 0;
  check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank failed");
  check_mpi(MPI_Comm_size(MPI_COMM_WORLD, &world), "MPI_Comm_size failed");

  unsigned long long length_u64 = static_cast<unsigned long long>(length);
  std::vector<unsigned long long> lengths(static_cast<size_t>(world));
  check_mpi(MPI_Allgather(&length_u64, 1, MPI_UNSIGNED_LONG_LONG,
                          lengths.data(), 1, MPI_UNSIGNED_LONG_LONG,
                          MPI_COMM_WORLD),
            "MPI_Allgather failed");
  for (auto other : lengths) {
    if (other != length_u64)
      throw std::runtime_error("RingAllreduce received different lengths");
  }

  const size_t base = length / static_cast<size_t>(world);
  std::vector<size_t> segment_sizes(static_cast<size_t>(world), base);
  for (size_t i = 0; i < length % static_cast<size_t>(world); ++i)
    ++segment_sizes[i];
  std::vector<size_t> segment_ends(static_cast<size_t>(world));
  for (int i = 0; i < world; ++i)
    segment_ends[static_cast<size_t>(i)] = segment_sizes[static_cast<size_t>(i)] +
        (i == 0 ? 0 : segment_ends[static_cast<size_t>(i - 1)]);

  if (length == 0) {
    *elapsed_seconds = 0.0;
    return;
  }

  std::memcpy(output, input, length * sizeof(float));
  iris::DMem output_mem(output, length * sizeof(float));
  const size_t buffer_size = segment_sizes.empty() ? 0 : segment_sizes[0];
  std::vector<float> buffer(buffer_size);
  iris::DMem buffer_mem(buffer.data(), buffer.size() * sizeof(float));

  const int recv_from = (rank - 1 + world) % world;
  const int send_to = (rank + 1) % world;
  MPI_Status status;
  const auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < world - 1; ++i) {
    const int recv_chunk = (rank - i - 1 + world) % world;
    const int send_chunk = (rank - i + world) % world;
    const size_t recv_size = segment_sizes[static_cast<size_t>(recv_chunk)];
    const size_t send_size = segment_sizes[static_cast<size_t>(send_chunk)];
    const size_t recv_offset = recv_chunk == 0 ? 0 : segment_ends[recv_chunk - 1];
    const size_t send_offset = send_chunk == 0 ? 0 : segment_ends[send_chunk - 1];

    check_mpi(MPI_Sendrecv(output + send_offset, static_cast<int>(send_size), MPI_FLOAT,
                           send_to, 0, buffer.data(), static_cast<int>(recv_size),
                           MPI_FLOAT, recv_from, 0, MPI_COMM_WORLD, &status),
              "MPI_Sendrecv scatter-reduce failed");
    reduce_segment(output_mem, buffer_mem, output, buffer.data(), recv_offset, recv_size);
  }

  for (int i = 0; i < world - 1; ++i) {
    const int send_chunk = (rank - i + 1 + world) % world;
    const int recv_chunk = (rank - i + world) % world;
    const size_t send_offset = send_chunk == 0 ? 0 : segment_ends[send_chunk - 1];
    const size_t recv_offset = recv_chunk == 0 ? 0 : segment_ends[recv_chunk - 1];
    check_mpi(MPI_Sendrecv(output + send_offset,
                           static_cast<int>(segment_sizes[send_chunk]), MPI_FLOAT,
                           send_to, 0, output + recv_offset,
                           static_cast<int>(segment_sizes[recv_chunk]), MPI_FLOAT,
                           recv_from, 0, MPI_COMM_WORLD, &status),
              "MPI_Sendrecv allgather failed");
    output_mem.update(output);
  }

  const auto end = std::chrono::steady_clock::now();
  *elapsed_seconds = std::chrono::duration<double>(end - start).count();
  (void)platform;
}

}  // namespace

int main(int argc, char** argv) {
  iris::Platform platform;
  if (platform.init(&argc, &argv, 1) != IRIS_SUCCESS) return 1;
  check_mpi(MPI_Init(nullptr, nullptr), "MPI_Init failed");

  const std::vector<size_t> buffer_sizes = {
      0, 32, 256, 1024, 4096, 16384, 65536, 262144, 1048576,
      8388608, 67108864, 536870912};
  const std::vector<size_t> iterations = {
      100000, 100000, 100000, 100000, 1000, 1000, 1000, 1000,
      100, 50, 10, 1};

  int rank = 0;
  int world = 0;
  check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank failed");
  check_mpi(MPI_Comm_size(MPI_COMM_WORLD, &world), "MPI_Comm_size failed");
  for (size_t i = 0; i < buffer_sizes.size(); ++i) {
    const size_t size = buffer_sizes[i];
    std::vector<float> input(size, 1.0f);
    std::vector<float> output(size);
    double seconds = 0.0;
    for (size_t iter = 0; iter < iterations[i]; ++iter) {
      double iteration_seconds = 0.0;
      ring_allreduce(platform, input.data(), size, output.data(), &iteration_seconds);
      seconds += iteration_seconds;
      for (float value : output) {
        if (value != static_cast<float>(world)) {
          std::cerr << "Unexpected result from allreduce: " << value << std::endl;
          MPI_Finalize();
          platform.finalize();
          return 1;
        }
      }
    }
    if (rank == 0) {
      std::cout << "Verified allreduce for size " << size << " ("
                << seconds * 1e6 / iterations[i]
                << " us per iteration)" << std::endl;
    }
  }

  check_mpi(MPI_Finalize(), "MPI_Finalize failed");
  platform.finalize();
  return 0;
}

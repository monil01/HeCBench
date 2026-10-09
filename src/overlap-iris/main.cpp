#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define STREAM_COUNT 4
#define BLOCK_SIZE 256

static int N = 1 << 22;
static int nreps = 10;
static int inner_reps = 5;
static int memsize = 0;

static bool check(const std::vector<std::vector<int>>& outputs)
{
  bool passed = true;
  for (int j = 0; j < STREAM_COUNT; ++j) {
    for (int i = 0; i < N; ++i) {
      passed &= (outputs[j][i] == inner_reps);
    }
  }
  return passed;
}

static float processWithStreams(
    int streams_used,
    std::vector<iris::DMem>& mem_in,
    std::vector<iris::DMem>& mem_out)
{
  int current_stream = 0;
  const size_t gws[1] = {
      ((static_cast<size_t>(N) + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE};
  const size_t lws[1] = {BLOCK_SIZE};

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < nreps; ++i) {
    int next_stream = (current_stream + 1) % streams_used;

    iris::Task task;
    void* params[4] = {
        &mem_out[current_stream],
        &mem_in[current_stream],
        &N,
        &inner_reps,
    };
    int pinfo[4] = {
        iris_w,
        iris_r,
        sizeof(N),
        sizeof(inner_reps),
    };
    task.kernel("incKernel", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 4, params, pinfo);
    task.flush_out(mem_out[current_stream]);
    task.submit(iris_gpu, nullptr, 1);

    (void)next_stream;
    current_stream = next_stream;
  }

  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return time * 1e-6f;
}

int main(int argc, char* argv[])
{
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  std::printf("Length of the array = %d\n", N);
  memsize = N * static_cast<int>(sizeof(int));

  std::vector<std::vector<int>> h_data_in(STREAM_COUNT);
  std::vector<std::vector<int>> h_data_out(STREAM_COUNT);
  std::vector<iris::DMem> dmem_in;
  std::vector<iris::DMem> dmem_out;
  dmem_in.reserve(STREAM_COUNT);
  dmem_out.reserve(STREAM_COUNT);

  for (int i = 0; i < STREAM_COUNT; ++i) {
    h_data_in[i].assign(N, 0);
    h_data_out[i].assign(N, 0);
    dmem_in.emplace_back(h_data_in[i].data(), static_cast<size_t>(memsize));
    dmem_out.emplace_back(h_data_out[i].data(), static_cast<size_t>(memsize));
  }

  float serial_time = processWithStreams(1, dmem_in, dmem_out);
  float overlap_time = processWithStreams(STREAM_COUNT, dmem_in, dmem_out);

  for (int i = 0; i < STREAM_COUNT; ++i) {
    dmem_out[i].fetch(h_data_out[i].data());
  }

  std::printf("\nAverage measured timings over %d repetitions:\n", nreps);
  std::printf(" Avg. time when execution fully serialized\t: %f ms\n",
              serial_time / nreps);
  std::printf(" Avg. time when overlapped using %d streams\t: %f ms\n",
              STREAM_COUNT, overlap_time / nreps);
  std::printf(" Avg. speedup gained (serialized - overlapped)\t: %f\n",
              (serial_time - overlap_time) / nreps);

  std::printf("\nMeasured throughput:\n");
  std::printf(" Fully serialized execution\t\t: %f GB/s\n",
              (nreps * (memsize * 2e-6f)) / serial_time);
  std::printf(" Overlapped using %d streams\t\t: %f GB/s\n", STREAM_COUNT,
              (nreps * (memsize * 2e-6f)) / overlap_time);

  bool bResults = check(h_data_out);
  std::printf("\n%s\n", bResults ? "PASS" : "FAIL");

  platform.finalize();
  return bResults ? EXIT_SUCCESS : EXIT_FAILURE;
}

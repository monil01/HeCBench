#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define VALUE 1
#define BLOCK_SIZE 256

static int launch_segment_reduce(iris::DMem& mem_out,
                                 iris::DMem& mem_in,
                                 size_t num_segments,
                                 int segment_size)
{
  const size_t gws[1] = {
    ((num_segments + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE
  };
  const size_t lws[1] = {BLOCK_SIZE};
  int segments_arg = static_cast<int>(num_segments);

  iris::Task task;
  void* params[4] = {&mem_out, &mem_in, &segments_arg, &segment_size};
  int pinfo[4] = {iris_w, iris_r, sizeof(segments_arg), sizeof(segment_size)};
  task.kernel("segment_reduce_kernel", 1, nullptr, const_cast<size_t*>(gws),
              const_cast<size_t*>(lws), 4, params, pinfo);
  task.flush_out(mem_out);
  return task.submit(iris_gpu, nullptr, 1);
}

static int segreduce(size_t num_elements, int repeat)
{
  std::printf("num_elements = %zu\n", num_elements);

  std::vector<int> h_in(num_elements, VALUE);
  iris::DMem mem_in(h_in.data(), num_elements * sizeof(int));

  int total_errors = 0;
  for (size_t segment_size = 16; segment_size <= 16384; segment_size *= 2) {
    const size_t num_segments = num_elements / segment_size;
    std::vector<int> h_out(num_segments, 0);
    iris::DMem mem_out(h_out.data(), num_segments * sizeof(int));
    const int segment_size_arg = static_cast<int>(segment_size);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) {
      const int rc = launch_segment_reduce(mem_out, mem_in, num_segments,
                                           segment_size_arg);
      if (rc != IRIS_SUCCESS) {
        std::fprintf(stderr, "IRIS task submit failed with code %d\n", rc);
        return 1;
      }
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::printf("num_segments = %zu ", num_segments);
    std::printf("segment_size = %zu ", segment_size);
    std::printf("Throughput = %f (G/s)\n", 1.f * num_elements * repeat / time);

    const int correct_segment_sum = static_cast<int>(segment_size) * VALUE;
    int errors = 0;
    for (size_t i = 0; i < num_segments; i++) {
      if (h_out[i] != correct_segment_sum) {
        errors++;
        if (errors < 10) {
          std::printf("segment %zu has sum %d (expected %d)\n", i,
                      h_out[i], correct_segment_sum);
        }
      }
    }

    if (errors > 0) {
      std::printf("segmented reduction does not agree with the reference! %d "
                  "errors!\n", errors);
      total_errors += errors;
    }
  }

  std::printf("%s\n", total_errors == 0 ? "PASS" : "FAIL");
  return total_errors == 0 ? 0 : 1;
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::printf("Usage: %s <multiplier> <repeat>\n", argv[0]);
    std::printf("The total number of elements is 16384 x multiplier\n");
    return 1;
  }

  const int multiplier = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  if (multiplier <= 0 || repeat <= 0) {
    std::fprintf(stderr, "multiplier and repeat must be positive\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const size_t num_elements = 16384 * static_cast<size_t>(multiplier);
  const int rc = segreduce(num_elements, repeat);

  platform.finalize();
  return rc;
}

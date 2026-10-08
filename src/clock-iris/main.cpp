#include <iris/iris.hpp>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 8192
#endif

#define NUM_THREADS 256

int main(int argc, char** argv) {
  std::printf("CUDA Clock sample\n");

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  float input[NUM_THREADS * 2];
  float output[NUM_BLOCKS];
  clock_t timer[NUM_BLOCKS * 2];

  for (int i = 0; i < NUM_THREADS * 2; i++) {
    input[i] = static_cast<float>(i);
  }
  for (int i = 0; i < NUM_BLOCKS; i++) {
    output[i] = 0.0f;
  }
  for (int i = 0; i < NUM_BLOCKS * 2; i++) {
    timer[i] = 0;
  }
  constexpr float expectedReduction = 0.0f;

  iris::DMem mem_input(input, sizeof(input));
  iris::DMem mem_output(output, sizeof(output));
  iris::DMem mem_timer(timer, sizeof(timer));

  size_t gws[1] = {NUM_BLOCKS * NUM_THREADS};
  size_t lws[1] = {NUM_THREADS};

  iris::Task task;
  void* params[3] = {&mem_input, &mem_output, &mem_timer};
  int pinfo[3] = {iris_r, iris_w, iris_w};
  task.kernel("timedReduction", 1, nullptr, gws, lws, 3, params, pinfo);
  task.flush_out(mem_output);
  task.flush_out(mem_timer);
  task.submit(iris_gpu, nullptr, 1);

  int errors = 0;
  for (int i = 0; i < NUM_BLOCKS; i++) {
    if (output[i] != expectedReduction) {
      errors++;
      if (errors < 5) {
        std::fprintf(stderr, "output[%d] = %f, expected %f\n", i, output[i], expectedReduction);
      }
    }
  }

  long double totalBlockTime = 0;
  for (int i = 0; i < NUM_BLOCKS; i++) {
    totalBlockTime += timer[NUM_BLOCKS + i] - timer[i];
  }

  clock_t minStart = timer[0];
  clock_t maxEnd = timer[NUM_BLOCKS];

  for (int i = 1; i < NUM_BLOCKS; i++) {
    minStart = timer[i] < minStart ? timer[i] : minStart;
    maxEnd = timer[NUM_BLOCKS + i] > maxEnd ? timer[NUM_BLOCKS + i] : maxEnd;
  }

  const clock_t totalClocks = maxEnd - minStart;
  std::printf("Total clocks = %lu\n", static_cast<unsigned long>(totalClocks));
  if (totalClocks != 0) {
    std::printf("Execution efficiency = %Lf\n", 100 * totalBlockTime / static_cast<long double>(totalClocks));
  } else {
    std::printf("Execution efficiency = 0.000000\n");
    errors++;
  }
  std::printf("%s\n", errors == 0 ? "PASS" : "FAIL");

  platform.finalize();
  return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

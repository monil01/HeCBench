#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 256

static void softMax_cpu(const int numSlice, const int sliceSize, const float* src, float* dest)
{
  for (int i = 0; i < numSlice; i++) {
    float max_ = src[i * sliceSize];
    for (int j = 0; j < sliceSize; j++) {
      max_ = (max_ < src[i * sliceSize + j]) ? src[i * sliceSize + j] : max_;
    }
    float sum = 0;
    for (int j = 0; j < sliceSize; j++) {
      float e = std::exp(src[i * sliceSize + j] - max_);
      sum += e;
      dest[i * sliceSize + j] = e;
    }
    for (int j = 0; j < sliceSize; j++) {
      dest[i * sliceSize + j] /= sum;
    }
  }
}

static float* alloc_floats(size_t count)
{
  void* ptr = nullptr;
  if (posix_memalign(&ptr, 1024, sizeof(float) * count) != 0) return nullptr;
  return static_cast<float*>(ptr);
}

static void launch_softmax(
    const char* kernel_name,
    const size_t blocks,
    const int numSlice,
    const int sliceSize,
    iris::DMem& mem_input,
    iris::DMem& mem_output,
    const bool flush_output)
{
  size_t gws[1] = { blocks * BLOCK_SIZE };
  size_t lws[1] = { BLOCK_SIZE };
  void* params[4] = { const_cast<int*>(&numSlice), const_cast<int*>(&sliceSize), &mem_input, &mem_output };
  int pinfo[4] = { sizeof(numSlice), sizeof(sliceSize), iris_r, iris_w };

  iris::Task task;
  task.kernel(kernel_name, 1, nullptr, gws, lws, 4, params, pinfo);
  if (flush_output) task.flush_out(mem_output);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    std::printf("Usage: %s <number of slices> <slice size> <implementations> <repeat>\n", argv[0]);
    std::printf("implementation 0: naive\n");
    std::printf("implementation 1: optimized\n");
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int numSlice = std::atoi(argv[1]);
  int sliceSize = std::atoi(argv[2]);
  int kernel = std::atoi(argv[3]);
  int repeat = std::atoi(argv[4]);
  int numElem = numSlice * sliceSize;

  float* input = alloc_floats(static_cast<size_t>(numElem));
  float* output_gpu = alloc_floats(static_cast<size_t>(numElem));
  float* output_cpu = alloc_floats(static_cast<size_t>(numElem));
  if (!input || !output_gpu || !output_cpu) {
    std::fprintf(stderr, "host allocation failed for %d elements\n", numElem);
    std::free(input);
    std::free(output_gpu);
    std::free(output_cpu);
    platform.finalize();
    return 1;
  }

  std::srand(2);
  for (int i = 0; i < numSlice; i++)
    for (int j = 0; j < sliceSize; j++)
      input[i * sliceSize + j] = static_cast<float>(std::rand() % 13);

  iris::DMem mem_input(input, sizeof(float) * static_cast<size_t>(numElem));
  iris::DMem mem_output(output_gpu, sizeof(float) * static_cast<size_t>(numElem));

  const char* kernel_name = kernel == 1 ? "softMax2" : "softMax";
  const size_t slices_per_block = kernel == 1 ? (BLOCK_SIZE / 64) : BLOCK_SIZE;
  const size_t blocks = (static_cast<size_t>(numSlice) + slices_per_block - 1) / slices_per_block;

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    launch_softmax(kernel_name, blocks, numSlice, sliceSize, mem_input, mem_output, n == repeat - 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  bool ok = true;
  softMax_cpu(numSlice, sliceSize, input, output_cpu);
  for (int i = 0; i < numElem; i++) {
    if (std::fabs(output_cpu[i] - output_gpu[i]) > 1e-3f) {
      std::printf("@index %d host: %f device: %f\n", i, output_cpu[i], output_gpu[i]);
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::free(input);
  std::free(output_cpu);
  std::free(output_gpu);
  platform.finalize();
  return 0;
}

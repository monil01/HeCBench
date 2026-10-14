#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "DCT8x8.h"

static unsigned int iDivUp(unsigned int dividend, unsigned int divisor)
{
  return dividend / divisor + (dividend % divisor != 0);
}

static int Verify(const float* h_OutputGPU,
                  float* h_OutputCPU,
                  const float* h_Input,
                  const unsigned int stride,
                  const unsigned int imageH,
                  const unsigned int imageW,
                  const int dir)
{
  std::printf("Comparing against Host/C++ computation...\n");
  DCT8x8CPU(h_OutputCPU, h_Input, stride, imageH, imageW, dir);
  double sum = 0.0;
  double delta = 0.0;
  for (unsigned int i = 0; i < imageH; i++) {
    for (unsigned int j = 0; j < imageW; j++) {
      const double cpu = h_OutputCPU[i * stride + j];
      const double diff = h_OutputGPU[i * stride + j] - cpu;
      sum += cpu * cpu;
      delta += diff * diff;
    }
  }
  const double L2norm = std::sqrt(delta / sum);
  std::printf("Relative L2 norm: %.3e\n\n", L2norm);
  if (L2norm < 1E-6) {
    std::printf("PASS\n");
    return 0;
  }
  std::printf("FAIL\n");
  return 1;
}

static int run_dct8x8(const char* kernel_name,
                      iris::DMem& mem_output,
                      iris::DMem& mem_input,
                      unsigned int stride,
                      unsigned int imageH,
                      unsigned int imageW)
{
  const size_t lws[2] = { BLOCK_X, BLOCK_Y / BLOCK_SIZE };
  const size_t gws[2] = {
    static_cast<size_t>(iDivUp(imageW, BLOCK_X)) * lws[0],
    static_cast<size_t>(iDivUp(imageH, BLOCK_Y)) * lws[1]
  };

  iris::Task task;
  void* params[5] = {
    &mem_output,
    &mem_input,
    &stride,
    &imageH,
    &imageW
  };
  int pinfo[5] = {
    iris_w,
    iris_r,
    sizeof(stride),
    sizeof(imageH),
    sizeof(imageW)
  };
  task.kernel(kernel_name, 2, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 5, params, pinfo);
  task.flush_out(mem_output);
  task.submit(iris_gpu, nullptr, 1);
  return 0;
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    std::printf("Usage: %s <image width> <image height> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const unsigned int imageW = static_cast<unsigned int>(std::atoi(argv[1]));
  const unsigned int imageH = static_cast<unsigned int>(std::atoi(argv[2]));
  const int numIterations = std::atoi(argv[3]);
  const unsigned int stride = imageW;
  const size_t imageBytes = static_cast<size_t>(imageH) * stride * sizeof(float);

  std::printf("Allocating and initializing host memory...\n");
  float* h_Input = static_cast<float*>(std::malloc(imageBytes));
  float* h_OutputCPU = static_cast<float*>(std::malloc(imageBytes));
  float* h_OutputGPU = static_cast<float*>(std::malloc(imageBytes));
  if (!h_Input || !h_OutputCPU || !h_OutputGPU) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(h_OutputGPU);
    std::free(h_OutputCPU);
    std::free(h_Input);
    platform.finalize();
    return 1;
  }

  std::srand(2009);
  for (unsigned int i = 0; i < imageH; i++) {
    for (unsigned int j = 0; j < imageW; j++) {
      h_Input[i * stride + j] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    }
  }

  iris::DMem mem_input(h_Input, imageBytes);
  iris::DMem mem_output(h_OutputGPU, imageBytes);

  std::printf("Performing Forward DCT8x8 of %u x %u image on the device\n\n", imageH, imageW);

  auto start = std::chrono::steady_clock::now();
  for (int iter = 0; iter < numIterations; iter++) {
    run_dct8x8("DCT8x8_kernel", mem_output, mem_input, stride, imageH, imageW);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average DCT8x8 kernel execution time %f (s)\n", (time * 1e-9f) / numIterations);

  mem_output.fetch(h_OutputGPU);
  int errors = Verify(h_OutputGPU, h_OutputCPU, h_Input, stride, imageH, imageW, DCT_FORWARD);

  std::printf("Performing Inverse DCT8x8 of %u x %u image on the device\n\n", imageH, imageW);

  start = std::chrono::steady_clock::now();
  for (int iter = 0; iter < numIterations; iter++) {
    run_dct8x8("IDCT8x8_kernel", mem_output, mem_input, stride, imageH, imageW);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average IDCT8x8 kernel execution time %f (s)\n", (time * 1e-9f) / numIterations);

  mem_output.fetch(h_OutputGPU);
  errors += Verify(h_OutputGPU, h_OutputCPU, h_Input, stride, imageH, imageW, DCT_INVERSE);

  std::free(h_OutputGPU);
  std::free(h_OutputCPU);
  std::free(h_Input);
  platform.finalize();
  return errors == 0 ? 0 : 1;
}

#include <iris/iris.hpp>

#include <cmath>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "sobol.h"
#include "sobol_gold.h"

#define L1ERROR_TOLERANCE (1e-6)

static double sobolIRIS(int repeat,
                        int n_vectors,
                        int n_dimensions,
                        iris::DMem& mem_directions,
                        iris::DMem& mem_output)
{
  const int threadsperblock = 128;

  unsigned int dimGridX = (n_dimensions < (4 * 24)) ? (4 * 24) : 1;
  if (dimGridX > static_cast<unsigned int>(n_vectors / threadsperblock)) {
    dimGridX = (n_vectors + threadsperblock - 1) / threadsperblock;
  }

  const unsigned int targetDimGridX = dimGridX;
  for (dimGridX = 1; dimGridX < targetDimGridX; dimGridX *= 2) {
  }

  size_t gws[2] = {static_cast<size_t>(dimGridX) * threadsperblock,
                   static_cast<size_t>(n_dimensions)};
  size_t lws[2] = {threadsperblock, 1};

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    iris::Task task;
    void* params[4] = {&n_vectors, &n_dimensions, &mem_directions, &mem_output};
    int pinfo[4] = {sizeof(n_vectors), sizeof(n_dimensions), iris_r, iris_w};
    task.kernel("sobolGPU_kernel", 2, nullptr, gws, lws, 4, params, pinfo);
    if (i == repeat - 1) {
      task.flush_out(mem_output);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  const auto end = std::chrono::steady_clock::now();
  const double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return time;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <number of vectors> <number of dimensions> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int n_vectors = std::atoi(argv[1]);
  const int n_dimensions = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  std::cout << "Allocating CPU memory..." << std::endl;
  std::vector<unsigned int> h_directions;
  std::vector<float> h_outputCPU;
  std::vector<float> h_outputGPU;

  try {
    h_directions.resize(static_cast<size_t>(n_dimensions) * n_directions);
    h_outputCPU.resize(static_cast<size_t>(n_vectors) * n_dimensions);
    h_outputGPU.resize(static_cast<size_t>(n_vectors) * n_dimensions);
  } catch (const std::exception& e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    std::cerr << "Unable to allocate CPU memory (try running with fewer vectors/dimensions)" << std::endl;
    platform.finalize();
    return EXIT_FAILURE;
  }

  std::cout << "Allocating GPU memory..." << std::endl;
  const size_t direction_size = h_directions.size() * sizeof(unsigned int);
  const size_t output_size = h_outputGPU.size() * sizeof(float);

  std::cout << "Initializing direction numbers..." << std::endl;
  initSobolDirectionVectors(n_dimensions, h_directions.data());

  std::cout << "Executing QRNG on GPU..." << std::endl;

  iris::DMem mem_directions(h_directions.data(), direction_size);
  iris::DMem mem_output(h_outputGPU.data(), output_size);

  const double ktime = sobolIRIS(repeat, n_vectors, n_dimensions, mem_directions, mem_output);
  std::cout << "Average kernel execution time: " << (ktime * 1e-9f) / repeat << " (s)\n";

  mem_output.fetch(h_outputGPU.data());

  std::cout << std::endl;
  std::cout << "Executing QRNG on CPU..." << std::endl;
  sobolCPU(n_vectors, n_dimensions, h_directions.data(), h_outputCPU.data());

  std::cout << "Checking results..." << std::endl;
  float l1norm_diff = 0.0F;
  float l1norm_ref = 0.0F;
  float l1error;

  if (n_vectors == 1) {
    for (int d = 0, v = 0; d < n_dimensions; ++d) {
      const float ref = h_outputCPU[d * n_vectors + v];
      l1norm_diff += std::fabs(h_outputGPU[d * n_vectors + v] - ref);
      l1norm_ref += std::fabs(ref);
    }

    l1error = l1norm_diff;

    if (l1norm_ref != 0) {
      std::cerr << "Error: L1-Norm of the reference is not zero (for single vector), golden generator appears broken\n";
    } else {
      std::cout << "L1-Error: " << l1error << std::endl;
    }
  } else {
    for (int d = 0; d < n_dimensions; ++d) {
      for (int v = 0; v < n_vectors; ++v) {
        const float ref = h_outputCPU[d * n_vectors + v];
        l1norm_diff += std::fabs(h_outputGPU[d * n_vectors + v] - ref);
        l1norm_ref += std::fabs(ref);
      }
    }

    l1error = l1norm_diff / l1norm_ref;

    if (l1norm_ref == 0) {
      std::cerr << "Error: L1-Norm of the reference is zero, golden generator appears broken\n";
    } else {
      std::cout << "L1-Error: " << l1error << std::endl;
    }
  }

  std::cout << "Shutting down..." << std::endl;
  platform.finalize();

  if (l1error < L1ERROR_TOLERANCE) {
    std::cout << "PASS" << std::endl;
  } else {
    std::cout << "FAIL" << std::endl;
  }

  return 0;
}

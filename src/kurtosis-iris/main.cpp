#include <iris/iris.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "StoreElement.h"
#include "kurtosis_common.h"

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::cout << "Usage ./main <elemCount> <repeat>\n";
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int elemCount = std::atoi(argv[1]);
  const int repeat = std::atoi(argv[2]);
  storeElement* elem = new storeElement[elemCount];

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dis(1.0f, 2.0f);
  for (int i = 0; i < elemCount; i++) {
    elem[i] = {i, 0, static_cast<unsigned long long int>(i), dis(gen)};
  }

  const int block_size = 256;
  const int partial_count = (elemCount + block_size - 1) / block_size;
  std::vector<kurtosisResult> partials(static_cast<size_t>(partial_count), {0, 0.0f, 0.0f, 0.0f, 0.0f});
  std::vector<kurtosisResult> scratch(static_cast<size_t>(partial_count), {0, 0.0f, 0.0f, 0.0f, 0.0f});

  iris::DMem mem_elem(elem, sizeof(storeElement) * static_cast<size_t>(elemCount));
  iris::DMem mem_partials(partials.data(), sizeof(kurtosisResult) * partials.size());
  iris::DMem mem_scratch(scratch.data(), sizeof(kurtosisResult) * scratch.size());

  const size_t lws[1] = {static_cast<size_t>(block_size)};
  auto start = std::chrono::steady_clock::now();
  iris::DMem* final_mem = &mem_partials;
  for (int i = 0; i < repeat; i++) {
    int elem_arg = elemCount;
    const size_t gws[1] = {static_cast<size_t>(partial_count * block_size)};
    iris::Task task;
    void* params[3] = {&mem_elem, &mem_partials, &elem_arg};
    int pinfo[3] = {iris_r, iris_w, sizeof(elem_arg)};
    task.kernel("kurtosis_partials", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 3, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);

    int current_count = partial_count;
    iris::DMem* in = &mem_partials;
    iris::DMem* out = &mem_scratch;
    while (current_count > 1) {
      int current_arg = current_count;
      int out_count = (current_count + block_size - 1) / block_size;
      const size_t reduce_gws[1] = {static_cast<size_t>(out_count * block_size)};
      iris::Task reduce;
      void* reduce_params[3] = {in, out, &current_arg};
      int reduce_pinfo[3] = {iris_r, iris_w, sizeof(current_arg)};
      reduce.kernel("kurtosis_reduce", 1, nullptr, const_cast<size_t*>(reduce_gws),
                    const_cast<size_t*>(lws), 3, reduce_params, reduce_pinfo);
      reduce.submit(iris_gpu, nullptr, 1);
      current_count = out_count;
      iris::DMem* tmp = in;
      in = out;
      out = tmp;
    }
    final_mem = in;
  }
  iris::Task flush;
  flush.flush_out(*final_mem);
  flush.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();

  kurtosisResult result{0, 0.0f, 0.0f, 0.0f, 0.0f};
  if (final_mem == &mem_partials) {
    mem_partials.fetch(partials.data());
    result = partials[0];
  } else {
    mem_scratch.fetch(scratch.data());
    result = scratch[0];
  }

  const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Total device compute time: " << time * 1e-9 << " (s)\n";
  std::cout << "Results:" << std::endl;
  std::cout << sizeof(kurtosisResult) << " "
            << result.count << " "
            << result.m2 << " "
            << result.m3 << " "
            << result.m4 << std::endl;

  delete[] elem;
  platform.finalize();
  return 0;
}

#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static bool check_output(int nres, int* output_data, int h_nres, const std::vector<int>& h_output)
{
  std::vector<int> output(output_data, output_data + nres);
  std::sort(output.begin(), output.end());
  return (h_nres == nres) &&
         std::equal(h_output.begin(), h_output.begin() + h_nres, output.begin());
}

static bool run_filter_kernel(
    const char* kernel_name,
    const char* label,
    int num_elems,
    int block_size,
    int repeat,
    iris::DMem& mem_input,
    iris::DMem& mem_output,
    iris::DMem& mem_nres,
    int* output,
    int& nres,
    int h_flt_count,
    const std::vector<int>& h_output)
{
  const size_t blocks = (static_cast<size_t>(num_elems) + block_size - 1) / block_size;
  const size_t gws[1] = { blocks * static_cast<size_t>(block_size) };
  const size_t lws[1] = { static_cast<size_t>(block_size) };
  int zero = 0;

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    mem_nres.update(&zero);
    iris::Task task;
    void* params[4] = { &mem_output, &mem_nres, &mem_input, &num_elems };
    int pinfo[4] = { iris_w, iris_rw, iris_r, sizeof(num_elems) };
    task.kernel(kernel_name, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 4, params, pinfo);
    if (i == repeat - 1) {
      task.flush_out(mem_output);
      task.flush_out(mem_nres);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of filter (%s) %lf (ms)\n", label, (time * 1e-6) / repeat);

  mem_nres.fetch(&nres);
  mem_output.fetch(output);
  const bool match = check_output(nres, output, h_flt_count, h_output);
  std::printf("%s\n", match ? "PASS" : "FAIL");
  return match;
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    std::printf("Usage: %s <number of elements> <block size> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int num_elems = std::atoi(argv[1]);
  const int block_size = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  std::vector<int> input(num_elems);
  for (int i = 0; i < num_elems; i++) {
    input[i] = i - num_elems / 2;
  }
  std::mt19937 g;
  g.seed(19937);
  std::shuffle(input.begin(), input.end(), g);

  std::vector<int> h_output(num_elems);
  int h_flt_count = 0;
  for (int i = 0; i < num_elems; i++) {
    if (input[i] > 0) {
      h_output[h_flt_count++] = input[i];
    }
  }
  std::sort(h_output.begin(), h_output.begin() + h_flt_count);

  std::vector<int> output(num_elems, 0);
  int nres = 0;
  iris::DMem mem_input(input.data(), sizeof(int) * static_cast<size_t>(num_elems));
  iris::DMem mem_output(output.data(), sizeof(int) * static_cast<size_t>(num_elems));
  iris::DMem mem_nres(&nres, sizeof(int));

  bool ok = run_filter_kernel("filter", "shared memory", num_elems, block_size, repeat,
                              mem_input, mem_output, mem_nres, output.data(), nres,
                              h_flt_count, h_output);
  ok = run_filter_kernel("filter2", "global aggregate", num_elems, block_size, repeat,
                         mem_input, mem_output, mem_nres, output.data(), nres,
                         h_flt_count, h_output) && ok;

  platform.finalize();
  return ok ? 0 : 1;
}

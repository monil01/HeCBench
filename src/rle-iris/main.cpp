#include <iris/iris.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <typeinfo>
#include <vector>

static bool g_verbose = false;
static int g_timing_iterations = 0;

static bool has_flag(int argc, char** argv, const char* flag)
{
  const std::string needle = std::string("--") + flag;
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == needle) {
      return true;
    }
  }
  return false;
}

static bool get_int_arg(int argc, char** argv, const char* name, int& value)
{
  const std::string prefix = std::string("--") + name + "=";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind(prefix, 0) == 0) {
      value = std::atoi(arg.c_str() + prefix.size());
      return true;
    }
  }
  return false;
}

static void initialize(int entropy_reduction, int* h_in, int num_items, int max_segment)
{
  const unsigned int max_int = static_cast<unsigned int>(-1);

  int key = 0;
  int i = 0;
  while (i < num_items) {
    int repeat;
    if (max_segment < 0) {
      repeat = num_items;
    } else if (max_segment < 2) {
      repeat = 1;
    } else {
      std::mt19937 mt(19937 + i);
      std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());
      repeat = dist(mt);
      for (int n = 1; n <= entropy_reduction; ++n) {
        repeat &= dist(mt);
      }
      repeat = static_cast<int>((static_cast<double>(repeat) * static_cast<double>(max_segment)) /
                                static_cast<double>(max_int));
      repeat = std::max(1, repeat);
    }

    int j = i;
    while (j < std::min(i + repeat, num_items)) {
      h_in[j] = key;
      ++j;
    }

    i = j;
    ++key;
  }

  if (g_verbose) {
    std::printf("Input:\n");
    for (int j = 0; j < num_items; ++j) {
      std::printf("%d ", h_in[j]);
    }
    std::printf("\n\n");
  }
}

static int solve(const int* h_in,
                 int* h_unique_reference,
                 int* h_lengths_reference,
                 int num_items)
{
  if (num_items == 0) {
    return 0;
  }

  int previous = h_in[0];
  int length = 1;
  int num_runs = 0;

  for (int i = 1; i < num_items; ++i) {
    if (previous != h_in[i]) {
      h_unique_reference[num_runs] = previous;
      h_lengths_reference[num_runs] = length;
      ++num_runs;
      length = 1;
    } else {
      ++length;
    }
    previous = h_in[i];
  }

  h_unique_reference[num_runs] = previous;
  h_lengths_reference[num_runs] = length;
  ++num_runs;
  return num_runs;
}

static int compare_ints(const int* reference, const int* output, int count)
{
  for (int i = 0; i < count; ++i) {
    if (reference[i] != output[i]) {
      if (g_verbose) {
        std::printf("Mismatch at %d: reference=%d output=%d\n", i, reference[i], output[i]);
      }
      return 1;
    }
  }
  return 0;
}

static void launch_rle(iris::DMem& mem_in,
                       iris::DMem& mem_unique,
                       iris::DMem& mem_lengths,
                       iris::DMem& mem_num_runs,
                       int num_items)
{
  size_t gws[1] = {1};
  size_t lws[1] = {1};
  void* params[5] = {&mem_in, &mem_unique, &mem_lengths, &mem_num_runs, &num_items};
  int pinfo[5] = {iris_r, iris_w, iris_w, iris_w, sizeof(num_items)};

  iris::Task task;
  task.kernel("rle_encode_kernel", 1, nullptr, gws, lws, 5, params, pinfo);
  task.flush_out(mem_unique);
  task.flush_out(mem_lengths);
  task.flush_out(mem_num_runs);
  task.submit(iris_gpu, nullptr, 1);
}

static int test_case(const char* label,
                     const std::vector<int>& input,
                     int max_segment,
                     int entropy_reduction)
{
  const int num_items = static_cast<int>(input.size());
  std::vector<int> unique_reference(num_items);
  std::vector<int> lengths_reference(num_items);
  const int num_runs = solve(input.data(),
                             unique_reference.data(),
                             lengths_reference.data(),
                             num_items);

  if (std::strcmp(label, "iterator") == 0) {
    std::printf("\nTest iterator: on %d items, %d segments (avg run length %.3f), {i key, "
                "i offset, i length}\n",
                num_items,
                num_runs,
                static_cast<float>(num_items) / num_runs);
  } else {
    std::printf("\nTest pointer: %d items, %d segments (avg run length %.3f), {i key, "
                "i offset, i length}, max_segment %d, entropy_reduction %d\n",
                num_items,
                num_runs,
                static_cast<float>(num_items) / num_runs,
                max_segment,
                entropy_reduction);
  }
  std::fflush(stdout);

  std::vector<int> unique_out(num_items, 0);
  std::vector<int> lengths_out(num_items, 0);
  int num_runs_out = 0;

  iris::DMem mem_in(const_cast<int*>(input.data()), sizeof(int) * input.size());
  iris::DMem mem_unique(unique_out.data(), sizeof(int) * unique_out.size());
  iris::DMem mem_lengths(lengths_out.data(), sizeof(int) * lengths_out.size());
  iris::DMem mem_num_runs(&num_runs_out, sizeof(int));

  launch_rle(mem_in, mem_unique, mem_lengths, mem_num_runs, num_items);
  mem_unique.fetch(unique_out.data());
  mem_lengths.fetch(lengths_out.data());
  mem_num_runs.fetch(&num_runs_out);

  const int compare0 = compare_ints(unique_reference.data(), unique_out.data(), num_runs);
  std::printf("\t Keys %s\n", compare0 ? "FAIL" : "PASS");

  const int compare1 = compare_ints(lengths_reference.data(), lengths_out.data(), num_runs);
  std::printf("\t Lengths %s\n", compare1 ? "FAIL" : "PASS");

  const int compare2 = (num_runs == num_runs_out) ? 0 : 1;
  std::printf("\t Count %s\n", compare2 ? "FAIL" : "PASS");
  std::fflush(stdout);
  std::fflush(stderr);

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < g_timing_iterations; ++i) {
    std::fill(unique_out.begin(), unique_out.end(), 0);
    std::fill(lengths_out.begin(), lengths_out.end(), 0);
    num_runs_out = 0;
    mem_unique.update(unique_out.data());
    mem_lengths.update(lengths_out.data());
    mem_num_runs.update(&num_runs_out);
    launch_rle(mem_in, mem_unique, mem_lengths, mem_num_runs, num_items);
  }
  const auto end = std::chrono::steady_clock::now();
  const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  const float elapsed_millis = time * 1e-6f;

  if (g_timing_iterations > 0) {
    const float avg_millis = elapsed_millis / g_timing_iterations;
    const float giga_rate = static_cast<float>(num_items) / avg_millis / 1000.0f / 1000.0f;
    const int bytes_moved = (num_items * static_cast<int>(sizeof(int))) +
                            (num_runs * (static_cast<int>(sizeof(int)) + static_cast<int>(sizeof(int))));
    const float giga_bandwidth = static_cast<float>(bytes_moved) / avg_millis / 1000.0f / 1000.0f;
    std::printf(", %.3f avg ms, %.3f billion items/s, %.3f logical GB/s",
                avg_millis,
                giga_rate,
                giga_bandwidth);
  }
  std::printf("\n\n");
  std::fflush(stdout);
  std::fflush(stderr);

  return compare0 + compare1 + compare2;
}

static int test_dispatch(int num_items)
{
  int errors = 0;

  std::vector<int> iterator_input(num_items, 1);
  errors += test_case("iterator", iterator_input, 0, 0);

  const int max_seg_limit = std::min(num_items, 1 << 16);
  const int max_seg_inc = 4;
  for (int max_segment = 1, entropy_reduction = 0;
       max_segment <= max_seg_limit;
       max_segment <<= max_seg_inc, ++entropy_reduction) {
    const int max_seg = std::max(1, max_segment);
    std::vector<int> input(num_items);
    initialize(entropy_reduction, input.data(), num_items, max_seg);
    errors += test_case("pointer", input, max_seg, entropy_reduction);
  }

  return errors;
}

int main(int argc, char** argv)
{
  int num_items = -1;

  g_verbose = has_flag(argc, argv, "v");
  get_int_arg(argc, argv, "n", num_items);
  get_int_arg(argc, argv, "i", g_timing_iterations);

  if (has_flag(argc, argv, "help")) {
    std::printf("%s [--n=<input items> [--i=<timing iterations> [--v]\n", argv[0]);
    return 0;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  std::printf("\n");
  const int effective_items = (num_items < 0) ? 1000000 : num_items;
  const int errors = test_dispatch(effective_items);

  platform.finalize();
  return errors == 0 ? 0 : 1;
}

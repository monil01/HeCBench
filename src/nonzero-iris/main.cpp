#include <iris/iris.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#define MAX_DIMS 2
#define BLOCK_SIZE 256

template <typename scalar_t>
static bool run_nonzero(const char* count_kernel,
                        const char* compact_kernel,
                        int nrows,
                        int ncols,
                        int repeat)
{
  const int in_ndims = MAX_DIMS;
  const int64_t in_sizes[MAX_DIMS] = { nrows, ncols };

  int64_t num_items = 1;
  for (int i = 0; i < in_ndims; i++) {
    num_items *= in_sizes[i];
  }

  std::mt19937 gen(19937);
  std::uniform_int_distribution<> dist(-1, 1);

  std::vector<scalar_t> h_in(static_cast<size_t>(num_items));
  std::vector<int64_t> h_out(static_cast<size_t>(num_items) * in_ndims);
  uint64_t h_nzeros_u64 = 0;

  iris::DMem mem_in(h_in.data(), sizeof(scalar_t) * h_in.size());
  iris::DMem mem_out(h_out.data(), sizeof(int64_t) * h_out.size());
  iris::DMem mem_nzeros(&h_nzeros_u64, sizeof(uint64_t));

  bool ok = true;
  long sum_time = 0;
  long idx_time = 0;

  const size_t blocks = (static_cast<size_t>(num_items) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const size_t gws[1] = { blocks * BLOCK_SIZE };
  const size_t lws[1] = { BLOCK_SIZE };
  std::vector<uint64_t> offsets(blocks);
  iris::DMem mem_offsets(offsets.data(), offsets.size() * sizeof(uint64_t));

  for (int n = 0; n < repeat; n++) {
    int64_t r_nzeros = 0;

    do {
      r_nzeros = 0;
      for (int64_t i = 0; i < num_items; i++) {
        h_in[static_cast<size_t>(i)] = static_cast<scalar_t>(dist(gen));
        if (h_in[static_cast<size_t>(i)] != static_cast<scalar_t>(0)) {
          r_nzeros++;
        }
      }
    } while (r_nzeros == 0);

    mem_in.update(h_in.data());
    h_nzeros_u64 = 0;
    mem_nzeros.update(&h_nzeros_u64);

    auto start = std::chrono::steady_clock::now();
    {
      iris::Task task;
      void* params[3] = { &mem_in, &mem_nzeros, &num_items };
      int pinfo[3] = { iris_r, iris_rw, sizeof(num_items) };
      task.kernel(count_kernel, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 3, params, pinfo);
      task.flush_out(mem_nzeros);
      task.submit(iris_gpu, nullptr, 1);
    }
    mem_nzeros.fetch(&h_nzeros_u64);
    int64_t h_nzeros = static_cast<int64_t>(h_nzeros_u64);
    auto end = std::chrono::steady_clock::now();
    sum_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    if (h_nzeros != r_nzeros) {
      std::printf("Number of non-zero elements mismatch: %ld != %ld (expected)\n",
                  h_nzeros,
                  r_nzeros);
      ok = false;
    } else {
      start = std::chrono::steady_clock::now();
      {
        iris::Task task;
        void* params[] = { &mem_in, &mem_offsets, &num_items };
        int pinfo[] = { iris_r, iris_w, sizeof(num_items) };
        size_t one[] = { 1 };
        const char* suffix = count_kernel + 14;
        std::string name = std::string("offsets_") + suffix;
        task.kernel(name.c_str(), 1, nullptr, one, one, 3, params, pinfo);
        task.submit(iris_gpu, nullptr, 1);
      }
      {
        iris::Task task;
        void* params[4] = { &mem_in, &mem_out, &mem_offsets, &num_items };
        int pinfo[4] = { iris_r, iris_w, iris_r, sizeof(num_items) };
        task.kernel(compact_kernel, 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 4, params, pinfo);
        task.submit(iris_gpu, nullptr, 1);
      }
      {
        iris::Task task;
        int64_t ncols64 = ncols;
        void* params[5] = { &mem_out, &ncols64, const_cast<int*>(&in_ndims), &h_nzeros, &num_items };
        int pinfo[5] = { iris_rw, sizeof(ncols64), sizeof(in_ndims), sizeof(h_nzeros), sizeof(num_items) };
        task.kernel("write_indices", 1, nullptr, const_cast<size_t*>(gws), const_cast<size_t*>(lws), 5, params, pinfo);
        task.flush_out(mem_out);
        task.submit(iris_gpu, nullptr, 1);
      }
      mem_out.fetch(h_out.data());
      end = std::chrono::steady_clock::now();
      idx_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      int64_t cnt_nzero = 0;
      for (int64_t i = 0; i < h_nzeros; i++) {
        if (in_ndims == 1) {
          if (h_in[static_cast<size_t>(h_out[static_cast<size_t>(i)])] != 0) {
            cnt_nzero++;
          }
        }
        if (in_ndims == 2) {
          const int64_t row = h_out[static_cast<size_t>(i)];
          const int64_t col = h_out[static_cast<size_t>(h_nzeros + i)];
          if (h_in[static_cast<size_t>(in_sizes[1] * row + col)] != 0) {
            cnt_nzero++;
          }
        }
      }

      ok = cnt_nzero == h_nzeros;
      int64_t pos = 0;
      for (int64_t flat = 0; flat < num_items; ++flat) {
        if (h_in[flat] == 0) continue;
        ok = ok && h_out[pos] == flat / ncols && h_out[h_nzeros + pos] == flat % ncols;
        ++pos;
      }
      std::printf("Whole-output stable coordinate check: %ld coordinates %s\n", pos, ok ? "PASS" : "FAIL");
    }

    if (!ok) {
      break;
    }
  }

  std::printf("Average time for sum reduction: %lf (us)\n", sum_time * 1e-3 / repeat);
  std::printf("Average time for write index operations: %lf (us)\n", idx_time * 1e-3 / repeat);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    std::printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int nrows = std::atoi(argv[1]);
  int ncols = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);

  if (nrows <= 0) nrows = 1;
  if (ncols <= 0) ncols = 1;

  bool passed = repeat > 0;
  if (!passed) { platform.finalize(); return 1; }
  for (int w = 0; w < 2; w++) {
    std::printf("=========== Data type is I8 ==========\n");
    passed &= run_nonzero<int8_t>("count_nonzero_i8", "compact_nonzero_i8", nrows, ncols, repeat);

    std::printf("=========== Data type is I16 ==========\n");
    passed &= run_nonzero<int16_t>("count_nonzero_i16", "compact_nonzero_i16", nrows, ncols, repeat);

    std::printf("=========== Data type is I32 ==========\n");
    passed &= run_nonzero<int32_t>("count_nonzero_i32", "compact_nonzero_i32", nrows, ncols, repeat);

    std::printf("=========== Data type is FP32 ==========\n");
    passed &= run_nonzero<float>("count_nonzero_f32", "compact_nonzero_f32", nrows, ncols, repeat);

    std::printf("=========== Data type is FP64 ==========\n");
    passed &= run_nonzero<double>("count_nonzero_f64", "compact_nonzero_f64", nrows, ncols, repeat);
  }

  platform.finalize();
  return passed ? 0 : 1;
}

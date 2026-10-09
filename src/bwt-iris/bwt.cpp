#include <iris/iris.hpp>

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

#include "bwt.hpp"

const int blockSize = 256;

static int RoundUpPowerOfTwo(int value) {
  value--;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  value++;
  return value;
}

std::pair<std::string, int*> bwt_with_suffix_array(const std::string sequence) {
  int argc = 0;
  char** argv = nullptr;
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int n = sequence.size();
  const int table_size = RoundUpPowerOfTwo(n);
  const int table_size_bytes = table_size * sizeof(int);
  const int seq_size_bytes = n * sizeof(char);

  int* table = static_cast<int*>(std::malloc(table_size_bytes));
  char* transformed_sequence_cstr =
      static_cast<char*>(std::malloc(seq_size_bytes));

  iris::DMem mem_table(table, table_size_bytes);
  iris::DMem mem_sequence(const_cast<char*>(sequence.c_str()), seq_size_bytes);
  iris::DMem mem_transformed(transformed_sequence_cstr, seq_size_bytes);

  int table_size_arg = table_size;
  int n_arg = n;
  size_t table_gws =
      static_cast<size_t>((table_size + blockSize - 1) / blockSize) *
      blockSize;
  size_t sequence_gws =
      static_cast<size_t>((n + blockSize - 1) / blockSize) * blockSize;
  size_t lws = blockSize;

  {
    iris::Task task;
    void* params[3] = {&mem_table, &table_size_arg, &n_arg};
    int pinfo[3] = {iris_w, sizeof(table_size_arg), sizeof(n_arg)};
    task.kernel("generate_table", 1, nullptr, &table_gws, &lws, 3, params,
                pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }

  for (int k = 2; k <= table_size; k <<= 1) {
    for (int j = k >> 1; j > 0; j = j >> 1) {
      int j_arg = j;
      int k_arg = k;
      iris::Task task;
      void* params[6] = {&mem_table, &table_size_arg, &j_arg,
                         &k_arg,     &mem_sequence,   &n_arg};
      int pinfo[6] = {iris_rw, sizeof(table_size_arg), sizeof(j_arg),
                      sizeof(k_arg), iris_r,           sizeof(n_arg)};
      task.kernel("bitonic_sort_step", 1, nullptr, &table_gws, &lws, 6,
                  params, pinfo);
      task.submit(iris_gpu, nullptr, 1);
    }
  }

  {
    iris::Task task;
    void* params[4] = {&mem_table, &mem_sequence, &mem_transformed, &n_arg};
    int pinfo[4] = {iris_r, iris_r, iris_w, sizeof(n_arg)};
    task.kernel("reconstruct_sequence", 1, nullptr, &sequence_gws, &lws, 4,
                params, pinfo);
    task.flush_out(mem_transformed);
    task.flush_out(mem_table);
    task.submit(iris_gpu, nullptr, 1);
  }

  std::string transformed_sequence(transformed_sequence_cstr, n);

  std::free(transformed_sequence_cstr);
  platform.finalize();

  return std::make_pair(transformed_sequence, table);
}

std::string bwt(const std::string sequence) {
  auto data = bwt_with_suffix_array(sequence);
  std::free(data.second);
  return data.first;
}

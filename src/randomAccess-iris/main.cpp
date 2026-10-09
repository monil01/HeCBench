#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

typedef unsigned long long int u64Int;
typedef long long int s64Int;

#define POLY 0x0000000000000007UL
#define K1_BLOCKSIZE 256
#define K2_BLOCKSIZE 128

static int initialize_table(iris::DMem& mem_table, u64Int table_size)
{
  const size_t blocks = (static_cast<size_t>(table_size) + K1_BLOCKSIZE - 1) / K1_BLOCKSIZE;
  size_t gws[1] = { blocks * K1_BLOCKSIZE };
  size_t lws[1] = { K1_BLOCKSIZE };
  iris::Task task;
  void* params[2] = { &mem_table, &table_size };
  int pinfo[2] = { iris_w, sizeof(table_size) };
  task.kernel("initTable", 1, nullptr, gws, lws, 2, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
  return 0;
}

static int update_table(iris::DMem& mem_table, u64Int table_size)
{
  size_t gws[1] = { K2_BLOCKSIZE };
  size_t lws[1] = { K2_BLOCKSIZE };
  iris::Task task;
  void* params[2] = { &mem_table, &table_size };
  int pinfo[2] = { iris_rw, sizeof(table_size) };
  task.kernel("update", 1, nullptr, gws, lws, 2, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
  return 0;
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  int failure;
  u64Int i;
  u64Int temp;
  double totalMem;
  u64Int* Table = nullptr;
  u64Int logTableSize;
  u64Int TableSize;

  totalMem = 1024 * 1024 * 512;
  totalMem /= sizeof(u64Int);

  for (totalMem *= 0.5, logTableSize = 0, TableSize = 1;
       totalMem >= 1.0;
       totalMem *= 0.5, logTableSize++, TableSize <<= 1) {
  }

  std::printf("Table size = %llu\n", TableSize);

  if (posix_memalign(reinterpret_cast<void**>(&Table), 1024, TableSize * sizeof(u64Int)) != 0) {
    Table = nullptr;
  }

  if (!Table) {
    std::fprintf(stderr, "Failed to allocate memory for the update table %llu\n", TableSize);
    platform.finalize();
    return 1;
  }

  std::fprintf(stdout, "Main table size   = 2^%llu = %llu words\n", logTableSize, TableSize);
  std::fprintf(stdout, "Number of updates = %llu\n", 4 * TableSize);

  iris::DMem mem_table(Table, TableSize * sizeof(u64Int));

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    initialize_table(mem_table, TableSize);
    update_table(mem_table, TableSize);
  }

  iris::Task flush;
  flush.flush_out(mem_table);
  flush.submit(iris_gpu, nullptr, 1);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  mem_table.fetch(Table);

  temp = 0x1;
  for (i = 0; i < 4 * TableSize; i++) {
    temp = (temp << 1) ^ (((s64Int)temp < 0) ? POLY : 0);
    Table[temp & (TableSize - 1)] ^= temp;
  }

  temp = 0;
  for (i = 0; i < TableSize; i++) {
    if (Table[i] != i) {
      temp++;
    }
  }

  std::fprintf(stdout, "Found %llu errors in %llu locations (%s).\n",
               temp, TableSize, (temp <= 0.01 * TableSize) ? "PASS" : "FAIL");
  failure = (temp <= 0.01 * TableSize) ? 0 : 1;

  std::free(Table);
  platform.finalize();
  return failure;
}

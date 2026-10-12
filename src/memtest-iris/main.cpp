#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define MAX_ERR_RECORD_COUNT 10

static void launch1(const char* name, size_t gx, size_t lx, iris::DMem& dev_mem,
                    unsigned long mem_size)
{
  iris::Task task;
  void* params[2] = {&dev_mem, &mem_size};
  int pinfo[2] = {iris_rw, sizeof(mem_size)};
  size_t gws[1] = {gx};
  size_t lws[1] = {lx};
  task.kernel(name, 1, nullptr, gws, lws, 2, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void launch_check(const char* name, size_t gx, size_t lx, iris::DMem& dev_mem,
                         unsigned long mem_size, iris::DMem& err_cnt,
                         iris::DMem& err_addr, iris::DMem& err_expect,
                         iris::DMem& err_current, iris::DMem& err_second_read)
{
  iris::Task task;
  void* params[7] = {&dev_mem, &mem_size, &err_cnt, &err_addr, &err_expect, &err_current,
                     &err_second_read};
  int pinfo[7] = {iris_r, sizeof(mem_size), iris_rw, iris_w, iris_w, iris_w, iris_w};
  size_t gws[1] = {gx};
  size_t lws[1] = {lx};
  task.kernel(name, 1, nullptr, gws, lws, 7, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void launch_pattern(const char* name, size_t gx, size_t lx, iris::DMem& dev_mem,
                           unsigned long mem_size, unsigned long p1)
{
  iris::Task task;
  void* params[3] = {&dev_mem, &mem_size, &p1};
  int pinfo[3] = {iris_rw, sizeof(mem_size), sizeof(p1)};
  size_t gws[1] = {gx};
  size_t lws[1] = {lx};
  task.kernel(name, 1, nullptr, gws, lws, 3, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void launch_read_write(size_t gx, size_t lx, iris::DMem& dev_mem, unsigned long mem_size,
                              unsigned long p1, unsigned long p2, iris::DMem& err_cnt,
                              iris::DMem& err_addr, iris::DMem& err_expect,
                              iris::DMem& err_current, iris::DMem& err_second_read)
{
  iris::Task task;
  void* params[9] = {&dev_mem, &mem_size, &p1, &p2, &err_cnt, &err_addr, &err_expect,
                     &err_current, &err_second_read};
  int pinfo[9] = {iris_rw, sizeof(mem_size), sizeof(p1), sizeof(p2), iris_rw,
                  iris_w, iris_w, iris_w, iris_w};
  size_t gws[1] = {gx};
  size_t lws[1] = {lx};
  task.kernel("kernel_read_write", 1, nullptr, gws, lws, 9, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void launch_read(size_t gx, size_t lx, iris::DMem& dev_mem, unsigned long mem_size,
                        unsigned long p1, iris::DMem& err_cnt, iris::DMem& err_addr,
                        iris::DMem& err_expect, iris::DMem& err_current,
                        iris::DMem& err_second_read)
{
  iris::Task task;
  void* params[8] = {&dev_mem, &mem_size, &p1, &err_cnt, &err_addr, &err_expect,
                     &err_current, &err_second_read};
  int pinfo[8] = {iris_r, sizeof(mem_size), sizeof(p1), iris_rw,
                  iris_w, iris_w, iris_w, iris_w};
  size_t gws[1] = {gx};
  size_t lws[1] = {lx};
  task.kernel("kernel_read", 1, nullptr, gws, lws, 8, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static void reset_error(iris::DMem& err_cnt)
{
  iris::Task task;
  void* params[1] = {&err_cnt};
  int pinfo[1] = {iris_w};
  size_t gws[1] = {1};
  size_t lws[1] = {1};
  task.kernel("kernel_reset_error", 1, nullptr, gws, lws, 1, params, pinfo);
  task.submit(iris_gpu, nullptr, 1);
}

static bool check(iris::DMem& err_cnt, unsigned* err_count)
{
  iris::Task task;
  void* params[1] = {&err_cnt};
  int pinfo[1] = {iris_r};
  size_t gws[1] = {1};
  size_t lws[1] = {1};
  task.kernel("kernel_flush_error", 1, nullptr, gws, lws, 1, params, pinfo);
  task.flush_out(err_cnt);
  task.submit(iris_gpu, nullptr, 1);
  err_cnt.fetch(err_count);
  const bool ok = (*err_count == 0);
  std::printf("%s", ok ? "." : "x");
  *err_count = 0;
  reset_error(err_cnt);
  return ok;
}

static bool moving_inversion(iris::DMem& err_cnt, iris::DMem& err_addr,
                             iris::DMem& err_expect, iris::DMem& err_current,
                             iris::DMem& err_second_read, iris::DMem& dev_mem,
                             unsigned long mem_size, unsigned long p1)
{
  unsigned long p2 = ~p1;
  const size_t gx = 1024 * 64;
  const size_t lx = 64;
  launch_pattern("kernel_write", gx, lx, dev_mem, mem_size, p1);
  for (int i = 0; i < 10; i++) {
    launch_read_write(gx, lx, dev_mem, mem_size, p1, p2, err_cnt, err_addr, err_expect,
                      err_current, err_second_read);
    p1 = p2;
    p2 = ~p1;
  }
  launch_read(gx, lx, dev_mem, mem_size, p1, err_cnt, err_addr, err_expect, err_current,
              err_second_read);
  unsigned err_count = 0;
  return check(err_cnt, &err_count);
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
  std::printf("Note: x indicates an error and . indicates no error when running each test\n");

  unsigned err_count = 0;
  unsigned long err_records[MAX_ERR_RECORD_COUNT] = {};
  const unsigned long mem_size = 2UL * 1024UL * 1024UL * 1024UL;
  char* host_mem = static_cast<char*>(std::malloc(mem_size));
  if (!host_mem) {
    std::printf("Unable to allocate host mirror for memtest\n");
    platform.finalize();
    return 1;
  }

  iris::DMem mem_err_cnt(&err_count, sizeof(unsigned));
  iris::DMem mem_err_addr(err_records, sizeof(err_records));
  iris::DMem mem_err_expect(err_records, sizeof(err_records));
  iris::DMem mem_err_current(err_records, sizeof(err_records));
  iris::DMem mem_err_second_read(err_records, sizeof(err_records));
  iris::DMem mem_dev(host_mem, mem_size);
  reset_error(mem_err_cnt);

  const size_t block = 64;
  std::printf("\ntest0: ");
  for (int i = 0; i < repeat; i++) {
    launch1("kernel0_write", 1024 * block, block, mem_dev, mem_size);
    launch_check("kernel0_read", 1024 * block, block, mem_dev, mem_size, mem_err_cnt,
                 mem_err_addr, mem_err_expect, mem_err_current, mem_err_second_read);
  }
  bool all_ok = check(mem_err_cnt, &err_count);

  std::printf("\ntest1: ");
  for (int i = 0; i < repeat; i++) {
    launch1("kernel1_write", 1024 * block, block, mem_dev, mem_size);
    launch_check("kernel1_read", 1024 * block, block, mem_dev, mem_size, mem_err_cnt,
                 mem_err_addr, mem_err_expect, mem_err_current, mem_err_second_read);
  }
  all_ok = check(mem_err_cnt, &err_count) && all_ok;

  std::printf("\ntest2: ");
  for (int i = 0; i < repeat; i++) {
    unsigned long p1 = 0;
    all_ok = moving_inversion(mem_err_cnt, mem_err_addr, mem_err_expect, mem_err_current,
                              mem_err_second_read, mem_dev, mem_size, p1) && all_ok;
    all_ok = moving_inversion(mem_err_cnt, mem_err_addr, mem_err_expect, mem_err_current,
                              mem_err_second_read, mem_dev, mem_size, ~p1) && all_ok;
  }

  std::printf("\ntest3: ");
  for (int i = 0; i < repeat; i++) {
    unsigned long p1 = 0x8080808080808080UL;
    all_ok = moving_inversion(mem_err_cnt, mem_err_addr, mem_err_expect, mem_err_current,
                              mem_err_second_read, mem_dev, mem_size, p1) && all_ok;
    all_ok = moving_inversion(mem_err_cnt, mem_err_addr, mem_err_expect, mem_err_current,
                              mem_err_second_read, mem_dev, mem_size, ~p1) && all_ok;
  }

  std::printf("\ntest4: ");
  std::srand(123);
  for (int i = 0; i < repeat; i++) {
    unsigned long p1 = std::rand();
    p1 = (p1 << 32) | std::rand();
    all_ok = moving_inversion(mem_err_cnt, mem_err_addr, mem_err_expect, mem_err_current,
                              mem_err_second_read, mem_dev, mem_size, p1) && all_ok;
  }

  std::printf("\ntest5: ");
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    launch1("kernel5_init", 64 * 1024 * block, block, mem_dev, mem_size);
    launch1("kernel5_move", 64 * 1024 * block, block, mem_dev, mem_size);
    launch_check("kernel5_check", 64 * 1024 * block, block, mem_dev, mem_size, mem_err_cnt,
                 mem_err_addr, mem_err_expect, mem_err_current, mem_err_second_read);
  }
  unsigned final_errors = 0;
  all_ok = check(mem_err_cnt, &final_errors) && all_ok;
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("\nAverage kernel execution time (test5): %f (s)\n", (time * 1e-9f) / repeat);
  std::printf("%s\n", all_ok ? "PASS" : "FAIL");

  std::free(host_mem);
  platform.finalize();
  return 0;
}

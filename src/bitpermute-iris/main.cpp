#include <iris/iris.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define WARP_SZ 32
#define MAX_LG_DOMAIN_SIZE 28

#if MAX_LG_DOMAIN_SIZE <= 32
typedef unsigned int index_t;
#else
typedef size_t index_t;
#endif

typedef long fr_t;

template <typename T>
static T reverseBits(T num)
{
  const unsigned int NO_OF_BITS = sizeof(T) * 8;
  T reverse_num = 0;
  for (unsigned int i = 0; i < NO_OF_BITS; i++) {
    if ((num & (static_cast<T>(1) << i))) {
      reverse_num |= static_cast<T>(1) << ((NO_OF_BITS - 1) - i);
    }
  }
  return reverse_num;
}

template <typename T>
static T rev(T i, unsigned int nbits)
{
  if (sizeof(T) == 4 || nbits <= 32) {
    return reverseBits(i) >> (8 * sizeof(unsigned int) - nbits);
  }
  return reverseBits(i) >> (8 * sizeof(unsigned long long) - nbits);
}

static void bit_rev_cpu(fr_t* out, const fr_t* in, uint32_t lg_domain_size)
{
  const uint32_t domain_size = 1u << lg_domain_size;
  for (uint32_t i = 0; i < domain_size; i++) {
    index_t r = rev(i, lg_domain_size);
    if (i < r) {
      out[r] = in[i];
      out[i] = in[r];
    } else if (i == r) {
      out[i] = in[i];
    }
  }
}

static void launch_bit_rev(iris::DMem& mem_inout, uint32_t lg_domain_size)
{
  const size_t domain_size = static_cast<size_t>(1) << lg_domain_size;
  const uint32_t Z_COUNT = 256 / sizeof(fr_t);
  const uint32_t bsize = std::max(Z_COUNT, static_cast<uint32_t>(WARP_SZ));

  iris::Task task;
  void* params[2] = { &mem_inout, &lg_domain_size };
  int pinfo[2] = { iris_rw, sizeof(lg_domain_size) };

  if (domain_size <= 1024) {
    size_t gws[1] = { domain_size };
    size_t lws[1] = { domain_size };
    task.kernel("bit_rev_permutation", 1, nullptr, gws, lws, 2, params, pinfo);
  } else if (domain_size < bsize * Z_COUNT) {
    size_t gws[1] = { domain_size };
    size_t lws[1] = { WARP_SZ };
    task.kernel("bit_rev_permutation", 1, nullptr, gws, lws, 2, params, pinfo);
  } else {
    size_t gws[1] = { (domain_size / Z_COUNT / bsize) * bsize };
    size_t lws[1] = { bsize };
    task.kernel("bit_rev_permutation_z", 1, nullptr, gws, lws, 2, params, pinfo);
  }

  task.submit(iris_gpu, nullptr, 1);
}

static int bit_permute(uint32_t lg_domain_size, int repeat)
{
  assert(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
  const size_t domain_size = static_cast<size_t>(1) << lg_domain_size;
  const size_t domain_size_bytes = sizeof(fr_t) * domain_size;

  std::printf("Domain size is %zu\n", domain_size);

  fr_t* inout = static_cast<fr_t*>(std::malloc(domain_size_bytes));
  fr_t* out = static_cast<fr_t*>(std::malloc(domain_size_bytes));
  if (!inout || !out) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(inout);
    std::free(out);
    return 1;
  }

  for (size_t i = 0; i < domain_size; i++) {
    out[i] = inout[i] = static_cast<fr_t>(i);
  }

  iris::DMem mem_inout(inout, domain_size_bytes);
  mem_inout.update(inout);
  launch_bit_rev(mem_inout, lg_domain_size);
  bit_rev_cpu(out, inout, lg_domain_size);

  iris::Task flush_task;
  flush_task.flush_out(mem_inout);
  flush_task.submit(iris_gpu, nullptr, 1);
  mem_inout.fetch(inout);

  const int error = std::memcmp(out, inout, domain_size_bytes);
  std::printf("%s\n", error ? "FAIL" : "PASS");

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    launch_bit_rev(mem_inout, lg_domain_size);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of the kernel: %f (us)\n\n", (time * 1e-3f) / repeat);

  std::free(inout);
  std::free(out);
  return error ? 1 : 0;
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
  int errors = 0;
  const char* bound = std::getenv("HECBENCH_BITPERMUTE_MAX_LOG");
  const int max_log = bound ? std::atoi(bound) : MAX_LG_DOMAIN_SIZE;
  errors += bit_permute(10, repeat);
  errors += bit_permute(11, repeat);
  errors += bit_permute(15, repeat);
  if (max_log >= 27) errors += bit_permute(27, repeat);
  if (max_log >= 28) errors += bit_permute(28, repeat);

  platform.finalize();
  return errors == 0 ? 0 : 1;
}

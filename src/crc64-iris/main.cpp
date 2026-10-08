#include <iris/iris.hpp>

#include <stdint.h>
#include <time.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static const uint64_t crc64_poly = UINT64_C(0xc96c5795d7870f42);
static const size_t crc64_min_thread_bytes = 1024;

static void make_crc64_table(uint64_t table[256])
{
  for (unsigned i = 0; i < 256; ++i) {
    uint64_t cs = i;
    for (unsigned bit = 0; bit < 8; ++bit) {
      cs = (cs >> 1) ^ ((cs & 1) ? crc64_poly : 0);
    }
    table[i] = cs;
  }
}

static uint64_t crc64_table_update(const void* input, size_t nbytes,
                                   const uint64_t table[256])
{
  const unsigned char* data = static_cast<const unsigned char*>(input);
  uint64_t cs = UINT64_C(0xffffffffffffffff);

  while (nbytes--) {
    uint32_t idx = static_cast<uint32_t>(cs ^ *data++) & 0xff;
    cs = table[idx] ^ (cs >> 8);
  }

  return cs ^ UINT64_C(0xffffffffffffffff);
}

static uint64_t crc64_multiply(uint64_t a, uint64_t b)
{
  if ((a ^ (a - 1)) < (b ^ (b - 1))) {
    uint64_t t = a;
    a = b;
    b = t;
  }

  if (a == 0) {
    return 0;
  }

  uint64_t r = 0;
  uint64_t h = UINT64_C(1) << 63;
  for (; a != 0; a <<= 1) {
    if (a & h) {
      r ^= b;
      a ^= h;
    }

    b = (b >> 1) ^ ((b & 1) ? crc64_poly : 0);
  }

  return r;
}

static uint64_t crc64_x_pow_n(uint64_t n)
{
  static const uint64_t x_pow_2n[64] = {
      UINT64_C(0x4000000000000000), UINT64_C(0x2000000000000000),
      UINT64_C(0x0800000000000000), UINT64_C(0x0080000000000000),
      UINT64_C(0x0000800000000000), UINT64_C(0x0000000080000000),
      UINT64_C(0xc96c5795d7870f42), UINT64_C(0x6d5f4ad7e3c3afa0),
      UINT64_C(0xd49f7e445077d8ea), UINT64_C(0x040fb02a53c216fa),
      UINT64_C(0x6bec35957b9ef3a0), UINT64_C(0xb0e3bb0658964afe),
      UINT64_C(0x218578c7a2dff638), UINT64_C(0x6dbb920f24dd5cf2),
      UINT64_C(0x7a140cfcdb4d5eb5), UINT64_C(0x41b3705ecbc4057b),
      UINT64_C(0xd46ab656accac1ea), UINT64_C(0x329beda6fc34fb73),
      UINT64_C(0x51a4fcd4350b9797), UINT64_C(0x314fa85637efae9d),
      UINT64_C(0xacf27e9a1518d512), UINT64_C(0xffe2a3388a4d8ce7),
      UINT64_C(0x48b9697e60cc2e4e), UINT64_C(0xada73cb78dd62460),
      UINT64_C(0x3ea5454d8ce5c1bb), UINT64_C(0x5e84e3a6c70feaf1),
      UINT64_C(0x90fd49b66cbd81d1), UINT64_C(0xe2943e0c1db254e8),
      UINT64_C(0xecfa6adeca8834a1), UINT64_C(0xf513e212593ee321),
      UINT64_C(0xf36ae57331040916), UINT64_C(0x63fbd333b87b6717),
      UINT64_C(0xbd60f8e152f50b8b), UINT64_C(0xa5ce4a8299c1567d),
      UINT64_C(0x0bd445f0cbdb55ee), UINT64_C(0xfdd6824e20134285),
      UINT64_C(0xcead8b6ebda2227a), UINT64_C(0xe44b17e4f5d4fb5c),
      UINT64_C(0x9b29c81ad01ca7c5), UINT64_C(0x1b4366e40fea4055),
      UINT64_C(0x27bca1551aae167b), UINT64_C(0xaa57bcd1b39a5690),
      UINT64_C(0xd7fce83fa1234db9), UINT64_C(0xcce4986efea3ff8e),
      UINT64_C(0x3602a4d9e65341f1), UINT64_C(0x722b1da2df516145),
      UINT64_C(0xecfc3ddd3a08da83), UINT64_C(0x0fb96dcca83507e6),
      UINT64_C(0x125f2fe78d70f080), UINT64_C(0x842f50b7651aa516),
      UINT64_C(0x09bc34188cd9836f), UINT64_C(0xf43666c84196d909),
      UINT64_C(0xb56feb30c0df6ccb), UINT64_C(0xaa66e04ce7f30958),
      UINT64_C(0xb7b1187e9af29547), UINT64_C(0x113255f8476495de),
      UINT64_C(0x8fb19f783095d77e), UINT64_C(0xaec4aacc7c82b133),
      UINT64_C(0xf64e6d09218428cf), UINT64_C(0x036a72ea5ac258a0),
      UINT64_C(0x5235ef12eb7aaa6a), UINT64_C(0x2fed7b1685657853),
      UINT64_C(0x8ef8951d46606fb5), UINT64_C(0x9d58c1090f034d14)};

  uint64_t r = UINT64_C(1) << 63;
  for (size_t i = 0; n != 0; n >>= 1, ++i) {
    if (n & 1) {
      r = crc64_multiply(r, x_pow_2n[i]);
    }
  }

  return r;
}

static uint64_t crc64_combine(uint64_t cs1, uint64_t cs2, size_t nbytes2)
{
  return cs2 ^ crc64_multiply(cs1, crc64_x_pow_n(8 * nbytes2));
}

static void crc64_invert(uint64_t cs, void* check_bytes)
{
  unsigned char* bytes = static_cast<unsigned char*>(check_bytes);
  cs ^= UINT64_C(0xffffffffffffffff);

  bytes[7] = (cs >> 56) & 0xff;
  bytes[6] = (cs >> 48) & 0xff;
  bytes[5] = (cs >> 40) & 0xff;
  bytes[4] = (cs >> 32) & 0xff;
  bytes[3] = (cs >> 24) & 0xff;
  bytes[2] = (cs >> 16) & 0xff;
  bytes[1] = (cs >> 8) & 0xff;
  bytes[0] = cs & 0xff;
}

static uint64_t crc64_parallel_iris(const void* input, size_t nbytes,
                                    const uint64_t table[256])
{
  if (nbytes <= 2 * crc64_min_thread_bytes) {
    return crc64_table_update(input, nbytes, table);
  }

  int nthreads = 96 * 8 * 32;
  if (nbytes < static_cast<size_t>(nthreads) * crc64_min_thread_bytes) {
    nthreads = static_cast<int>(nbytes / crc64_min_thread_bytes);
  }

  std::vector<uint64_t> thread_cs(nthreads);
  std::vector<size_t> thread_sz(nthreads);

  iris::DMem dev_thread_sz(thread_sz.data(), thread_sz.size() * sizeof(size_t));
  iris::DMem dev_thread_cs(thread_cs.data(), thread_cs.size() * sizeof(uint64_t));
  iris::DMem dev_data(const_cast<void*>(input), nbytes);
  iris::DMem dev_table(const_cast<uint64_t*>(table), 256 * sizeof(uint64_t));

  size_t lws[1] = {64};
  size_t gws[1] = {static_cast<size_t>((nthreads + 63) / 64) * 64};

  iris::Task task;
  void* params[6] = {&dev_thread_sz, &dev_thread_cs, &dev_data,
                     &dev_table, &nbytes, &nthreads};
  int pinfo[6] = {iris_w, iris_w, iris_r, iris_r, sizeof(nbytes),
                  sizeof(nthreads)};
  task.kernel("crc64_kernel", 1, nullptr, gws, lws, 6, params, pinfo);
  task.flush_out(dev_thread_sz);
  task.flush_out(dev_thread_cs);
  task.submit(iris_gpu, nullptr, 1);

  dev_thread_sz.fetch(thread_sz.data());
  dev_thread_cs.fetch(thread_cs.data());

  uint64_t cs = thread_cs[0];
  for (int i = 1; i < nthreads; ++i) {
    cs = crc64_combine(cs, thread_cs[i], thread_sz[i]);
  }

  return cs;
}

int main(int argc, char** argv)
{
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int ntests = 10;
  if (argc > 1) {
    ntests = std::atoi(argv[1]);
  }

  int seed = 5;
  if (argc > 2) {
    seed = std::atoi(argv[2]);
  }

  int max_test_length = 2097152;
  if (argc > 3) {
    max_test_length = std::atoi(argv[3]);
  }

  std::cout << "Running " << ntests << " tests with seed " << seed << std::endl;

  uint64_t table[256];
  make_crc64_table(table);
  srand48(seed);

#ifdef __bgp__
#define THE_CLOCK CLOCK_REALTIME
#else
#define THE_CLOCK CLOCK_THREAD_CPUTIME_ID
#endif

  double tot_time = 0.0;
  double tot_bytes = 0.0;

  int ntest = 0;
  while (++ntest <= ntests) {
    std::cout << ntest << " ";

    size_t test_length = static_cast<size_t>(max_test_length * (drand48() + 1));
    std::cout << test_length << " ";

    std::vector<unsigned char> input_buffer(test_length);
    for (size_t i = 0; i < test_length; ++i) {
      input_buffer[i] = static_cast<unsigned char>(255 * drand48());
    }

    timespec b_start;
    timespec b_end;
    clock_gettime(THE_CLOCK, &b_start);
    uint64_t cs = crc64_parallel_iris(input_buffer.data(), test_length, table);
    clock_gettime(THE_CLOCK, &b_end);

    double b_time = (b_end.tv_sec - b_start.tv_sec);
    b_time += 1e-9 * (b_end.tv_nsec - b_start.tv_nsec);

    if (ntest > 1) {
      tot_time += b_time;
      tot_bytes += test_length;
    }

    size_t tlend = 8;
    input_buffer.resize(test_length + tlend, 0);
    crc64_invert(cs, &input_buffer[test_length]);

    std::string pass("pass");
    std::string fail("fail");
    uint64_t csc = crc64_table_update(input_buffer.data(), test_length + tlend, table);
    std::cout << ((csc == UINT64_C(0xffffffffffffffff)) ? pass : fail) << " ";

    size_t div_pt = static_cast<size_t>(test_length * drand48());
    uint64_t cs1 = crc64_table_update(input_buffer.data(), div_pt, table);
    uint64_t cs2 = crc64_table_update(&input_buffer[div_pt], test_length - div_pt, table);
    csc = crc64_combine(cs1, cs2, test_length - div_pt);
    std::cout << ((csc == cs) ? pass : fail);
    std::cout << std::endl;
  }

  std::cout << (tot_bytes / (1024 * 1024)) / tot_time << " MB/s" << std::endl;

  platform.finalize();
  return 0;
}

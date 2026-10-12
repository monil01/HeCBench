#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))

#define mix(a,b,c) \
{ \
  a -= c;  a ^= rot(c, 4);  c += b; \
  b -= a;  b ^= rot(a, 6);  a += c; \
  c -= b;  c ^= rot(b, 8);  b += a; \
  a -= c;  a ^= rot(c,16);  c += b; \
  b -= a;  b ^= rot(a,19);  a += c; \
  c -= b;  c ^= rot(b, 4);  b += a; \
}

#define final(a,b,c) \
{ \
  c ^= b; c -= rot(b,14); \
  a ^= c; a -= rot(c,11); \
  b ^= a; b -= rot(a,25); \
  c ^= b; c -= rot(b,16); \
  a ^= c; a -= rot(c,4);  \
  b ^= a; b -= rot(a,14); \
  c ^= b; c -= rot(b,24); \
}

static unsigned int mixRemainder(unsigned int a,
                                 unsigned int b,
                                 unsigned int c,
                                 unsigned int k0,
                                 unsigned int k1,
                                 unsigned int k2,
                                 unsigned int length)
{
  switch(length)
  {
    case 12: c+=k2; b+=k1; a+=k0; break;
    case 11: c+=k2&0xffffff; b+=k1; a+=k0; break;
    case 10: c+=k2&0xffff; b+=k1; a+=k0; break;
    case 9 : c+=k2&0xff; b+=k1; a+=k0; break;
    case 8 : b+=k1; a+=k0; break;
    case 7 : b+=k1&0xffffff; a+=k0; break;
    case 6 : b+=k1&0xffff; a+=k0; break;
    case 5 : b+=k1&0xff; a+=k0; break;
    case 4 : a+=k0; break;
    case 3 : a+=k0&0xffffff; break;
    case 2 : a+=k0&0xffff; break;
    case 1 : a+=k0&0xff; break;
    case 0 : return c;
  }

  final(a,b,c);
  return c;
}

static unsigned int hashlittle(const void *key, size_t length, unsigned int initval)
{
  unsigned int a,b,c;
  a = b = c = 0xdeadbeef + ((unsigned int)length) + initval;

  const unsigned int *k = (const unsigned int *)key;

  while (length > 12)
  {
    a += k[0];
    b += k[1];
    c += k[2];
    mix(a,b,c);
    length -= 12;
    k += 3;
  }

  switch(length)
  {
    case 12: c+=k[2]; b+=k[1]; a+=k[0]; break;
    case 11: c+=k[2]&0xffffff; b+=k[1]; a+=k[0]; break;
    case 10: c+=k[2]&0xffff; b+=k[1]; a+=k[0]; break;
    case 9 : c+=k[2]&0xff; b+=k[1]; a+=k[0]; break;
    case 8 : b+=k[1]; a+=k[0]; break;
    case 7 : b+=k[1]&0xffffff; a+=k[0]; break;
    case 6 : b+=k[1]&0xffff; a+=k[0]; break;
    case 5 : b+=k[1]&0xff; a+=k[0]; break;
    case 4 : a+=k[0]; break;
    case 3 : a+=k[0]&0xffffff; break;
    case 2 : a+=k[0]&0xffff; break;
    case 1 : a+=k[0]&0xff; break;
    case 0 : return c;
  }

  final(a,b,c);
  return c;
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    std::printf("Usage: %s <block size> <number of strings> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  int block_size = std::atoi(argv[1]);
  unsigned long N = std::atol(argv[2]);
  int repeat = std::atoi(argv[3]);

  const char* str = "Four score and seven years ago";
  unsigned int c = hashlittle(str, 30, 1);
  std::printf("input string: %s hash is %.8x\n", str, c);

  const size_t key_words = static_cast<size_t>(N) * 16;
  const size_t keys_bytes = sizeof(unsigned int) * key_words;
  const size_t vector_bytes = sizeof(unsigned int) * static_cast<size_t>(N);

  unsigned int *keys = nullptr;
  unsigned int *lens = nullptr;
  unsigned int *initvals = nullptr;
  unsigned int *out = nullptr;
  if (posix_memalign((void**)&keys, 1024, keys_bytes) ||
      posix_memalign((void**)&lens, 1024, vector_bytes) ||
      posix_memalign((void**)&initvals, 1024, vector_bytes) ||
      posix_memalign((void**)&out, 1024, vector_bytes)) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(keys);
    std::free(lens);
    std::free(initvals);
    std::free(out);
    platform.finalize();
    return 1;
  }

  srand(2);
  char src[64];
  std::memcpy(src, str, 64);
  for (unsigned long i = 0; i < N; i++) {
    std::memcpy((unsigned char*)keys+i*16*sizeof(unsigned int), src, 64);
    lens[i] = rand()%61;
    initvals[i] = i%2;
    out[i] = 0;
  }

  iris::DMem mem_keys(keys, keys_bytes);
  iris::DMem mem_lens(lens, vector_bytes);
  iris::DMem mem_initvals(initvals, vector_bytes);
  iris::DMem mem_out(out, vector_bytes);

  size_t gws[1] = {
      static_cast<size_t>((N + block_size - 1) / block_size * block_size)};
  size_t lws[1] = {static_cast<size_t>(block_size)};
  int n_int = static_cast<int>(N);

  auto start = std::chrono::steady_clock::now();
  for (int n = 0; n < repeat; n++) {
    iris::Task task;
    void* params[5] = {&mem_lens, &mem_initvals, &mem_keys, &mem_out, &n_int};
    int pinfo[5] = {iris_r, iris_r, iris_r, iris_w, sizeof(n_int)};
    task.kernel("jk3_hash_kernel", 1, nullptr, gws, lws, 5, params, pinfo);
    if (n == repeat - 1) {
      task.flush_out(mem_out);
    }
    task.submit(iris_gpu, nullptr, 1);
  }

  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time : %f (s)\n",
              (time * 1e-9f) / repeat);

  mem_out.fetch(out);

  std::printf("Verify the results computed on the device..\n");
  bool error = false;
  for (unsigned long i = 0; i < N; i++) {
    c = hashlittle(&keys[i*16], lens[i], initvals[i]);
    if (out[i] != c) {
      std::printf("Error: at %lu gpu hash is %.8x  cpu hash is %.8x\n", i, out[i], c);
      error = true;
      break;
    }
  }

  std::printf("%s\n", error ? "FAIL" : "PASS");

  std::free(keys);
  std::free(lens);
  std::free(initvals);
  std::free(out);

  platform.finalize();
  return error ? 2 : 0;
}

#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);

  uint8_t char_to_uint[256] = {};
  for (int i = 0; i < 10; i++) char_to_uint[i + '0'] = i;
  for (int i = 0; i < 26; i++) char_to_uint[i + 'a'] = i + 10;
  for (int i = 0; i < 26; i++) char_to_uint[i + 'A'] = i + 10;

  const char* h_key = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
  const char* h_nonce = "0001020304050607";
  const char* h_keystream =
      "f798a189f195e66982105ffb640bb7757f579da31602fc93ec01ac56f85ac3c134a4547b733b46413042c9440049176905d3be59ea1c53f15916155c2be8241a38008b9a26bc35941e2444177c8ade6689de95264986d95889fb60e84629c9bd9a5acb1cc118be563eb9b3a4a472f82e09a7e778492b562ef7130e88dfe031c79db9d4f7c7a899151b9a475032b63fc385245fe054e3dd5a97a5f576fe064025d3ce042c566ab2c507b138db853e3d6959660996546cc9c4a6eafdc777c040d70eaf46f76dad3979e5c5360c3317166a1c894c94a371876a94df7628fe4eaaf2ccb27d5aaae0ad7ad0f9d4b6ad3b54098746d4524d38407a6deb3ab78fab78c9";

  const int key_len = std::strlen(h_key);
  const int nonce_len = std::strlen(h_nonce);
  const int keystream_len = std::strlen(h_keystream);
  const int result_len = keystream_len / 2;

  uint8_t* result = static_cast<uint8_t*>(std::malloc(result_len));
  uint8_t* raw_key = static_cast<uint8_t*>(std::malloc(key_len / 2));
  uint8_t* raw_nonce = static_cast<uint8_t*>(std::malloc(nonce_len / 2));
  uint8_t* raw_keystream = static_cast<uint8_t*>(std::malloc(result_len));
  if (!result || !raw_key || !raw_nonce || !raw_keystream) {
    std::fprintf(stderr, "host allocation failed\n");
    std::free(result);
    std::free(raw_key);
    std::free(raw_nonce);
    std::free(raw_keystream);
    platform.finalize();
    return 1;
  }

  iris::DMem mem_char_to_uint(char_to_uint, sizeof(char_to_uint));
  iris::DMem mem_key(const_cast<char*>(h_key), key_len);
  iris::DMem mem_nonce(const_cast<char*>(h_nonce), nonce_len);
  iris::DMem mem_keystream(const_cast<char*>(h_keystream), keystream_len);
  iris::DMem mem_raw_key(raw_key, key_len / 2);
  iris::DMem mem_raw_nonce(raw_nonce, nonce_len / 2);
  iris::DMem mem_raw_keystream(raw_keystream, result_len);
  iris::DMem mem_result(result, result_len);

  const size_t gws[1] = {256};
  const size_t lws[1] = {256};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    std::memset(result, 0, result_len);
    mem_result.update(result);

    iris::Task task;
    void* params[11] = {
        &mem_key, &mem_nonce, &mem_keystream, &mem_char_to_uint,
        &mem_raw_key, &mem_raw_nonce, &mem_raw_keystream, &mem_result,
        const_cast<int*>(&key_len), const_cast<int*>(&nonce_len),
        const_cast<int*>(&keystream_len)};
    int pinfo[11] = {
        iris_r, iris_r, iris_r, iris_r, iris_w, iris_w, iris_w, iris_rw,
        sizeof(key_len), sizeof(nonce_len), sizeof(keystream_len)};
    task.kernel("test_keystreams", 1, nullptr, const_cast<size_t*>(gws),
                const_cast<size_t*>(lws), 11, params, pinfo);
    task.submit(iris_gpu, nullptr, 1);
  }
  iris::Task flush;
  flush.flush_out(mem_result);
  flush.flush_out(mem_raw_keystream);
  flush.submit(iris_gpu, nullptr, 1);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of kernels: %f (us)\n", (time * 1e-3f) / repeat);

  mem_result.fetch(result);
  mem_raw_keystream.fetch(raw_keystream);

  int error = std::memcmp(result, raw_keystream, result_len);
  std::printf("%s\n", error ? "FAIL" : "PASS");

  std::free(result);
  std::free(raw_key);
  std::free(raw_nonce);
  std::free(raw_keystream);
  platform.finalize();
  return error ? 1 : 0;
}

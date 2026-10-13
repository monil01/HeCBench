#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <chrono>

#define ERT_ALIGN           256
#define ERT_NUM_EXPERIMENTS 1
#define ERT_MEMORY_MAX      33554432
#define ERT_WORKING_SET_MIN 128
#define ERT_TRIALS_MIN      1
#define ERT_WSS_MULT        1.3

#include "initialize.hpp"
#include "iris_support.hpp"
int device;


template <typename T>
T *alloc(uint64_t psize)
{
  T* buffer = (T *)calloc(psize/sizeof(T), sizeof(T));
  if (buffer == nullptr) {
    fprintf(stderr, "Out of memory!\n");
    exit(1);
  }
  return buffer;
}

template <typename T>
inline void launchKernel(uint64_t n, uint64_t t, iris::DMem& memory) {
  uint32_t trials = t, count = n;
  const char* name = std::is_same<T, half2>::value ? "ert_half2" :
                     (std::is_same<T, float>::value ? "ert_float" : "ert_double");
  void* p[] = { &trials, &count, &memory };
  int info[] = { sizeof(trials), sizeof(count), iris_rw };
  launch_iris(device, name, size_t(gpu_blocks)*gpu_threads, gpu_threads, 3, p, info);
}

template <typename T>
void run(uint64_t PSIZE, T *buf)
{

  uint64_t nsize = PSIZE;
  nsize          = nsize & (~(ERT_ALIGN - 1));
  nsize          = nsize / sizeof(T);

  iris::DMem memory(buf, nsize*sizeof(T));
  uint64_t n, nNew;
  uint64_t t;
  int bytes_per_elem;
  int mem_accesses_per_elem;

  n = ERT_WORKING_SET_MIN;
  while (n <= nsize) { // working set - nsize

    uint64_t ntrials = nsize / n;
    if (ntrials < ERT_TRIALS_MIN)
      ntrials = ERT_TRIALS_MIN;

    // initialize small chunck of buffer within each thread
    float value = -1.f;
    initialize<T>(nsize, buf, value);

    memory.update(buf);

    for (t = 1; t <= ntrials; t = t * 2) { // working set - ntrials
      launchKernel<T>(n, t, memory);
    } // working set - ntrials

    flush_iris(device, memory);

    nNew = ERT_WSS_MULT * n;
    if (nNew == n) {
      nNew = n + 1;
    }

    n = nNew;
  } // working set - nsize


}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    fprintf(stderr, "Usage: %s gpu_blocks gpu_threads\n", argv[0]);
    return -1;
  }

  gpu_blocks  = atoi(argv[1]);
  gpu_threads = atoi(argv[2]);
  printf("\n");
  printf("GPU_BLOCKS     %d\n", gpu_blocks);
  printf("GPU_THREADS    %d\n", gpu_threads);

  iris::Platform platform;
  device = init_iris(platform, argc, argv);
  uint64_t TSIZE = ERT_MEMORY_MAX;
  if (const char* bytes = getenv("ERT_VALIDATION_BYTES")) TSIZE = strtoull(bytes, nullptr, 10);
  uint64_t PSIZE = TSIZE;
  double checksum;

  // FP16
  half2 *hlfbuf = alloc<half2>(PSIZE);
  auto start = std::chrono::steady_clock::now();
  run<half2>(PSIZE, hlfbuf);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("runtime (half2): %lf (s)\n", time * 1e-9);
  checksum = 0; 
  for (uint64_t i = 0; i < PSIZE / sizeof(half2); i++) {
    float2 t = __half22float2(hlfbuf[i]);
    checksum += t.x + t.y;
  }
  printf("checksum: %lf\n", checksum);
  free(hlfbuf);

  // FP32
  float *sglbuf = alloc<float>(PSIZE);
  start = std::chrono::steady_clock::now();
  run<float>(PSIZE, sglbuf);
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("runtime (float): %lf (s)\n", time * 1e-9);
  checksum = 0; 
  for (uint64_t i = 0; i < PSIZE/sizeof(float); i++) {
    checksum += sglbuf[i];
  }
  printf("checksum: %lf\n", checksum);
  free(sglbuf);

  // FP64
  double *dblbuf = alloc<double>(PSIZE);
  start = std::chrono::steady_clock::now();
  run<double>(PSIZE, dblbuf);
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("runtime (double): %lf (s)\n", time * 1e-9);
  checksum = 0; 
  for (uint64_t i = 0; i < PSIZE/sizeof(double); i++) {
    checksum += dblbuf[i];
  }
  printf("checksum: %lf\n", checksum);
  free(dblbuf);

  int errors = platform.error_count();
  check_iris(platform.finalize());
  return errors ? 1 : 0;
}

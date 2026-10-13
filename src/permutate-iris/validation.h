#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <chrono>
#include <csignal>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

// Validation-only lease covers the GPU routine, never CPU statistics/compression.
// Each executable queues independently; wait <=120s and GPU phase <=90s.
class ValidationGPULease {
  int fd = -1;
  using Handler = void (*)(int);
  Handler previous = SIG_DFL;
  std::chrono::steady_clock::time_point acquired;
  static void wait_timeout(int) {
    const char msg[] = "Infrastructure GPU lock wait timeout (120s)\n";
    (void)write(STDERR_FILENO, msg, sizeof(msg)-1); _exit(75);
  }
  static void phase_timeout(int) {
    const char msg[] = "GPU lease bounded phase timeout (90s)\n";
    (void)write(STDERR_FILENO, msg, sizeof(msg)-1); _exit(124);
  }
public:
  ValidationGPULease() {
    if (!getenv("PERMUTATE_GPU_LEASE")) return;
    fd = open("/tmp/hecbench-iris-validation.lock", O_CREAT | O_RDWR, 0666);
    if (fd < 0) _exit(75);
    previous = signal(SIGALRM, wait_timeout); alarm(120);
    auto start = std::chrono::steady_clock::now();
    if (flock(fd, LOCK_EX)) _exit(75);
    acquired = std::chrono::steady_clock::now();
    alarm(0); signal(SIGALRM, phase_timeout); alarm(90);
    printf("lock_wait_seconds %.6f\n", std::chrono::duration<double>(acquired-start).count());
    fflush(stdout);
  }
  ~ValidationGPULease() {
    if (fd < 0) return;
    alarm(0); signal(SIGALRM, previous);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-acquired).count();
    flock(fd, LOCK_UN); close(fd);
    printf("gpu_lease_seconds %.6f (released before CPU compression)\n", seconds);
    fflush(stdout);
  }
  ValidationGPULease(const ValidationGPULease&) = delete;
  ValidationGPULease& operator=(const ValidationGPULease&) = delete;
};

// Opt-in lossless observation only: no changes to statistics or stopping rules.
inline void validation_batch(uint32_t batch, const uint32_t* counts) {
  const char* prefix = getenv("PERMUTATE_DUMP");
  if (!prefix) return;
  std::string path = std::string(prefix) + ".batches";
  FILE* f = fopen(path.c_str(), batch == 0 ? "wb" : "ab");
  if (!f || fwrite(counts, sizeof(uint32_t), 54, f) != 54 || fclose(f)) abort();
}
inline void validation_shuffle(const uint8_t* data, size_t len, const uint8_t* bits, size_t blen) {
  const char* prefix = getenv("PERMUTATE_SHUFFLE_DUMP");
  if (!prefix) return;
  FILE* f = fopen(prefix, "wb");
  if (!f || fwrite(data, 1, len, f) != len || (blen && fwrite(bits, 1, blen, f) != blen) || fclose(f)) abort();
}
inline void validation_final(double mean, double median, const double* results,
                             const uint32_t* counts, bool iid, uint32_t size, uint32_t len, uint32_t N) {
  const char* prefix = getenv("PERMUTATE_DUMP");
  if (!prefix) return;
  std::string path = std::string(prefix) + ".final";
  FILE* f = fopen(path.c_str(), "w");
  if (!f) abort();
  fprintf(f, "%u %u %u %u\n%a %a\n", size, len, N, unsigned(iid), mean, median);
  for (int i = 0; i < 19; ++i) fprintf(f, "%a\n", results[i]);
  for (int i = 0; i < 57; ++i) fprintf(f, "%u\n", counts[i]);
  if (fclose(f)) abort();
}

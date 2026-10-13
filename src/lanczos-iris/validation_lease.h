#pragma once
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <csignal>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
class ValidationGPULease {
  int fd = -1;
  using Handler = void (*)(int);
  Handler previous = SIG_DFL;
  std::chrono::steady_clock::time_point acquired;
  static void wait_timeout(int) {
    const char msg[] = "Infrastructure GPU lock wait timeout (300s)\n";
    (void)write(STDERR_FILENO, msg, sizeof(msg)-1); _exit(75);
  }
  static void phase_timeout(int) {
    const char msg[] = "GPU lease bounded phase timeout (90s)\n";
    (void)write(STDERR_FILENO, msg, sizeof(msg)-1); _exit(124);
  }
public:
  explicit ValidationGPULease(const char* variable) {
    if (!getenv(variable)) return;
    fd = open("/tmp/hecbench-iris-validation.lock", O_CREAT | O_RDWR, 0666);
    if (fd < 0) _exit(75);
    previous = signal(SIGALRM, wait_timeout); alarm(300);
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
    printf("gpu_lease_seconds %.6f (released before CPU postprocessing)\n", seconds);
    fflush(stdout);
  }
  ValidationGPULease(const ValidationGPULease&) = delete;
  ValidationGPULease& operator=(const ValidationGPULease&) = delete;
};

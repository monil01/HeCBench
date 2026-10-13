#pragma once
#include <cstdio>
#include <cstdlib>

// Optional evidence only; the normal arguments and stdout are unchanged.
inline void cobahh_dump(int N, float* ge, float* gi, float* h, float* m,
                        float* n, float* v, float* lastspike, char* refractory,
                        float* h_ge, float* h_gi, float* h_h, float* h_m,
                        float* h_n, float* h_v, float* h_lastspike,
                        char* h_refractory) {
  const char* path = std::getenv("COBAHH_DUMP");
  if (!path) return;
  FILE* out = std::fopen(path, "wb");
  if (!out) { std::perror(path); std::exit(1); }
  float* arrays[] = {ge, gi, h, m, n, v, lastspike,
                    h_ge, h_gi, h_h, h_m, h_n, h_v, h_lastspike};
  for (float* array : arrays)
    if (std::fwrite(array, sizeof(float), N, out) != static_cast<size_t>(N))
      { std::fprintf(stderr, "short evidence write\n"); std::exit(1); }
  if (std::fwrite(refractory, 1, N, out) != static_cast<size_t>(N) ||
      std::fwrite(h_refractory, 1, N, out) != static_cast<size_t>(N) ||
      std::fclose(out)) {
    std::fprintf(stderr, "evidence write failed\n"); std::exit(1);
  }
}

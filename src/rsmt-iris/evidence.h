#pragma once
#include <cstdlib>
// Optional observational output. Normal command line and benchmark output are unchanged.
static void dumpResults(const int* idxin, const int* idxout, const coord_t* x,
                        const coord_t* y, const edge* e, int nets) {
  const char* path = getenv("RSMT_DUMP");
  if (!path) return;
  FILE* f = fopen(path, "w");
  if (!f) { perror(path); exit(1); }
  fprintf(f, "nets %d\n", nets);
  for (int i = 0; i <= nets; ++i) fprintf(f, "idx %d %d %d\n", i, idxin[i], idxout[i]);
  for (int i = 0; i < 2 * idxin[nets]; ++i)
    fprintf(f, "slot %d %d %d %d %d\n", i, x[i], y[i], int(e[i].src), int(e[i].dst));
  if (fclose(f)) { perror(path); exit(1); }
}

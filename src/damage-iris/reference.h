#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>

inline bool validate(
    const int block_size,
    const int m,
    const int n,
    const int* __restrict__ nlist,
    const int* __restrict__ family,
    int* __restrict__ n_neigh,
    double* __restrict__ damage)
{
  double* damage_ref = static_cast<double*>(std::malloc(sizeof(double) * m));
  int* n_neigh_ref = static_cast<int*>(std::malloc(sizeof(int) * m));

  for (int i = 0; i < m; i++) {
    const int lb = i * block_size;
    int ub = lb + block_size;
    if (ub > n) ub = n;

    int sum = 0;
    for (int j = lb; j < ub; j++) {
      sum += nlist[j] != -1 ? 1 : 0;
    }

    n_neigh_ref[i] = sum;
    damage_ref[i] = 1.0 - static_cast<double>(sum) / static_cast<double>(family[i]);
  }

  bool ok = true;
  for (int i = 0; i < m; i++) {
    if (n_neigh[i] != n_neigh_ref[i] || std::fabs(damage[i] - damage_ref[i]) > 1e-6) {
      ok = false;
      break;
    }
  }
  std::printf("%s\n", ok ? "PASS" : "FAIL");

  std::free(damage_ref);
  std::free(n_neigh_ref);
  return ok;
}

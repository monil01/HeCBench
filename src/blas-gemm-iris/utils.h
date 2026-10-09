#pragma once

#include <cstdlib>
#include <iostream>

template <typename fp>
void rand_matrix(fp* M, int n_row, int n_col)
{
  for (int i = 0; i < n_row; i++) {
    for (int j = 0; j < n_col; j++) {
      M[i * n_col + j] = static_cast<fp>(std::rand() % 2);
    }
  }
}

inline void performance(int m, int n, int k, bool is_integer, double avg_time)
{
  double total_ops = double(m) * double(n) * double(k) * 2 + double(m) * double(n);
  std::cout << "Average GEMM execution time: " << avg_time * 1e-3 << " (us), ";
  double perf = total_ops / avg_time;

  auto scale_string = "G";
  auto unit_string = is_integer ? "OP/s" : "FLOP/s";
  if (perf >= 1000) {
    perf /= 1000;
    scale_string = "T";
  }
  std::cout << perf << " " << scale_string << unit_string << std::endl;
}

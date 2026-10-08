extern "C" __global__ void chi_kernel(const unsigned int rows,
                                       const unsigned int cols,
                                       const int cRows,
                                       const int contRows,
                                       const unsigned char* __restrict__ snpdata,
                                       float* __restrict__ results)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= static_cast<int>(cols)) return;

  int cases[3] = {1, 1, 1};
  int controls[3] = {1, 1, 1};
  for (int m = 0; m < cRows; m++) {
    unsigned char y = snpdata[static_cast<size_t>(m) * cols + tid];
    if (y == '0') cases[0]++;
    else if (y == '1') cases[1]++;
    else if (y == '2') cases[2]++;
  }
  for (int n = cRows; n < cRows + contRows; n++) {
    unsigned char y = snpdata[static_cast<size_t>(n) * cols + tid];
    if (y == '0') controls[0]++;
    else if (y == '1') controls[1]++;
    else if (y == '2') controls[2]++;
  }

  int tot_cases = 1;
  int tot_controls = 1;
  for (unsigned int p = 0; p < 3; p++) {
    tot_cases += cases[p];
    tot_controls += controls[p];
  }
  int total = tot_cases + tot_controls;

  float chisquare = 0.0f;
  for (unsigned int p = 0; p < 3; p++) {
    float exp = static_cast<float>(cases[p]) + controls[p];
    float Cexpected = tot_cases * exp / total;
    float Conexpected = tot_controls * exp / total;
    float numerator1 = static_cast<float>(cases[p]) - Cexpected;
    float numerator2 = static_cast<float>(controls[p]) - Conexpected;
    chisquare += numerator1 * numerator1 / Cexpected +
                 numerator2 * numerator2 / Conexpected;
  }
  results[tid] = chisquare;
}

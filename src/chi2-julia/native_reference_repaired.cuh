// Scoped native reference correction: original function preserved in ../chi2-cuda/chi2.cu.
// Only semantic changes: initialize row marginal accumulators to0, not1, so totals equal actual pseudocounted contingency-table margins.
__global__ void chi_kernel_reference_repaired(
  const unsigned int rows,
  const unsigned int cols,
  const int cRows,
  const int contRows,
  const unsigned char *__restrict__ snpdata,
  float *__restrict__ results)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= cols) return;

  unsigned char y;
  int m, n;
  unsigned int p = 0;
  int tot_cases = 0;
  int tot_controls= 0;
  int total = 1;
  float chisquare = 0.0f;
  float exp[3];        
  float Conexpected[3];        
  float Cexpected[3];
  float numerator1;
  float numerator2;

  int cases[3] = {1,1,1};
  int controls[3] = {1,1,1};

  // read cases: each thread reads a column of snpdata matrix
  for ( m = 0 ; m < cRows ; m++ ) {
    y = snpdata[(size_t)m * (size_t)cols + tid];
    if ( y == '0') { cases[0]++; }
    else if ( y == '1') { cases[1]++; }
    else if ( y == '2') { cases[2]++; }
  }

  // read controls: each thread reads a column of snpdata matrix
  for ( n = cRows ; n < cRows + contRows ; n++ ) {
    y = snpdata[(size_t)n * (size_t)cols + tid];
    if ( y == '0' ) { controls[0]++; }
    else if ( y == '1') { controls[1]++; }
    else if ( y == '2') { controls[2]++; }
  }

  for( p = 0 ; p < 3; p++ ) {
    tot_cases += cases[p];
    tot_controls += controls[p];
  }
  total = tot_cases + tot_controls;

  for( p = 0 ; p < 3; p++ ) {
    exp[p] = (float)cases[p] + controls[p]; 
    Cexpected[p] = tot_cases * exp[p] / total;
    Conexpected[p] = tot_controls * exp[p] / total;
    numerator1 = (float)cases[p] - Cexpected[p];
    numerator2 = (float)controls[p] - Conexpected[p];
    chisquare += numerator1 * numerator1 / Cexpected[p] +
                 numerator2 * numerator2 / Conexpected[p];
  }
  results[tid] = chisquare;
}


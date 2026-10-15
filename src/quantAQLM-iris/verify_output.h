#pragma once
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// Independent decoded matrix dot product, not a copy of the GPU reduction.
static void verify_output(const __half* input, const short* codes,
                          const __half* book, const __half* output, int batches) {
  if (const char* path = getenv("HECBENCH_DUMP")) {
    FILE* f = fopen(path, "wb");
    if (!f || fwrite(output, sizeof(__half), batches*12288, f) != (size_t)batches*12288 || fclose(f)) exit(3);
  }
  if (!getenv("HECBENCH_VERIFY")) return;
  std::vector<float> x(batches*4096), cb(3*65536*8);
  for (size_t i=0;i<x.size();++i) x[i]=__half2float(input[i]);
  for (size_t i=0;i<cb.size();++i) cb[i]=__half2float(book[i]);
  size_t bad=0;
  double max_abs=0, max_rel=0;
  for (int b=0;b<batches;++b) for (int row=0;row<12288;++row) {
    double sum=0;
    int partition=row/4096;
    for (int code=0;code<512;++code) {
      unsigned idx=static_cast<uint16_t>(codes[row*512+code]);
      for (int j=0;j<8;++j)
        sum += double(cb[(partition*65536+idx)*8+j])*x[b*4096+code*8+j];
    }
    double got=__half2float(output[b*12288+row]);
    double err=fabs(got-sum), rel=err/fmax(fabs(sum),1e-30);
    max_abs=fmax(max_abs,err); max_rel=fmax(max_rel,rel);
    bad += !std::isfinite(got) || err > 0.01 + 0.002*fabs(sum);
  }
  printf("Independent full decoded-dot oracle: outputs=%d max_abs=%.9g max_rel=%.9g atol=0.01 rtol=0.002 mismatches=%zu %s\n",
         batches*12288,max_abs,max_rel,bad,bad?"FAIL":"PASS");
  if (bad) exit(2);
}

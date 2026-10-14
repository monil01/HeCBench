#pragma once
#include <stdio.h>
#include <stdlib.h>
static void dump_histograms(const options& args, const long long* dd,
                            const long long* dr, const long long* rr) {
  const char* path = getenv("HECBENCH_TPACF_HISTOGRAM_DUMP");
  if (!path) return;
  FILE* file = fopen(path, "wb");
  const char magic[8] = "TPACFH1";
  const int shape[5] = {NUMBINS, args.njk, args.random_count, args.ndpoints, args.nrpoints};
  const size_t ddsize = static_cast<size_t>(NUMBINS)*args.njk;
  const size_t drsize = ddsize*args.random_count;
  const size_t rrsize = static_cast<size_t>(NUMBINS)*args.random_count;
  if (!file || fwrite(magic,1,8,file)!=8 || fwrite(shape,sizeof(int),5,file)!=5 ||
      fwrite(dd,sizeof(long long),ddsize,file)!=ddsize ||
      fwrite(dr,sizeof(long long),drsize,file)!=drsize ||
      fwrite(rr,sizeof(long long),rrsize,file)!=rrsize) {
    fprintf(stderr,"TPACF histogram dump failed\n");
    if(file) fclose(file);
    exit(1);
  }
  fclose(file);
}

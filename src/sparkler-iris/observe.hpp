#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <numeric>
// Optional full-matrix diagnostic, identical in isolated native baseline.
inline void observe(int rank,int iteration,int step,size_t left,size_t right,size_t m,size_t n,size_t k,size_t ld,const float* c) {
  const char* prefix=std::getenv("SPARKLER_DUMP");
  if(!prefix)return;
  char path[4096];
  std::snprintf(path,sizeof(path),"%s.rank%d.bin",prefix,rank);
  FILE* f=std::fopen(path,iteration==1&&step==1?"wb":"ab");
  if(!f)std::abort();
  uint64_t header[]={uint64_t(rank),uint64_t(iteration),uint64_t(step),left,right,m,n,k};
  if(std::fwrite(header,sizeof(header),1,f)!=1)std::abort();
  for(size_t col=0;col<n;++col) {
    if(std::fwrite(c+col*ld,sizeof(float),m,f)!=m)std::abort();
    for(size_t row=0;row<m;++row) {
      size_t a=1+(left+row)%499,b=1+(right+col)%499;
      float expected=1+(k-1)/std::lcm(a,b);
      if(c[row+col*ld]!=expected) {
        std::fprintf(stderr,"Full matrix oracle failure rank%d step%d (%zu,%zu): %.9g expected %.9g\n",rank,step,row,col,c[row+col*ld],expected);
        std::abort();
      }
    }
  }
  if(std::fclose(f))std::abort();
}

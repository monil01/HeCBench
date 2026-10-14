#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
inline void diagnostic_dump_intervals(I_CPU* output,int* counts) {
 const char* path=getenv("HECBENCH_DUMP");if(!path) return;
 static_assert(sizeof(int)==sizeof(int32_t),"count format");
 FILE* f=fopen(path,"wb");if(!f) throw std::runtime_error("open interval dump");
 uint64_t n=THREADS;
 if(fwrite(&n,sizeof(n),1,f)!=1 || fwrite(counts,sizeof(int),THREADS,f)!=THREADS) throw std::runtime_error("count dump");
 for(int thread=0;thread<THREADS;thread++) {
  if(counts[thread]<0 || counts[thread]>DEPTH_RESULT) throw std::runtime_error("invalid interval count");
  for(int i=0;i<counts[thread];i++) {
   T pair[]={output[THREADS*i+thread].lower(),output[THREADS*i+thread].upper()};
   if(fwrite(pair,sizeof(T),2,f)!=2) throw std::runtime_error("endpoint dump");
  }
 }
 fclose(f);
}

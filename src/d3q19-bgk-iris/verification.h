#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <string>
static int control(const char* key,int fallback){const char* s=getenv(key);return s?atoi(s):fallback;}
static void record_velocity(const float* data,size_t n,int run,int iter){
 uint64_t hash=1469598103934665603ULL;size_t bad=0;
 for(size_t i=0;i<3*n;++i)if(!std::isfinite(data[i]))++bad;
 const auto* bytes=reinterpret_cast<const unsigned char*>(data);
 for(size_t i=0;i<3*n*sizeof(float);++i){hash^=bytes[i];hash*=1099511628211ULL;}
 printf("VELOCITY run=%d iter=%d elements=%zu hash=%016llx nonfinite=%zu\n",run,iter,3*n,(unsigned long long)hash,bad);
 if(bad)exit(2);
 if(const char* path=getenv("HECBENCH_LBM_DUMP")){
  const std::string file=std::string(path)+"/run_"+std::to_string(run)+"_iter_"+std::to_string(iter)+".bin";
  FILE* f=fopen(file.c_str(),"wb");if(!f)exit(3);
  const size_t wrote=fwrite(data,sizeof(float),3*n,f);if(fclose(f)||wrote!=3*n)exit(3);
 }
}

#pragma once
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
static size_t validation_cap(const char* name,size_t original) {
  const char* v=getenv(name);if(!v)return original;
  size_t cap=strtoul(v,nullptr,10);if(!cap)exit(2);return std::min(cap,original);
}
static void validate_native(float** buffers,size_t count,size_t values,const float* expected,const char* mode) {
  const char* prefix=getenv("HECBENCH_SCCD_DUMP");
  if(!prefix && !getenv("HECBENCH_SCCD_VALIDATE"))return;
  static unsigned serial=0;FILE* f=nullptr;
  if(prefix) {
    std::string path=std::string(prefix)+"."+mode+"."+std::to_string(serial)+".bin";
    f=fopen(path.c_str(),"wb");if(!f)exit(2);
  }
  ++serial;std::vector<float> data(values);
  for(size_t b=0;b<count;++b) {
    if(hipMemcpy(data.data(),buffers[b],values*sizeof(float),hipMemcpyDeviceToHost)!=hipSuccess)exit(2);
    for(size_t i=0;i<values;++i)if(data[i]!=expected[i]) {fprintf(stderr,"FAIL buffer%zu element%zu\n",b,i);exit(1);}
    if(f && fwrite(data.data(),sizeof(float),values,f)!=values)exit(2);
  }
  if(f && fclose(f))exit(2);
  printf("PASS native %s: all%zu buffers x%zu floats\n",mode,count,values);
}

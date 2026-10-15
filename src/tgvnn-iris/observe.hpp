#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
// Identical opt-in full-stage diagnostic for native and IRIS. Never computes
// a replacement result: reads actual GPU outputs only.
inline void tgv_observe(const char* name,const void* pointer,size_t bytes){
 const char* prefix=std::getenv("TGVNN_DUMP");if(!prefix)return;
 std::vector<unsigned char> host(bytes);
#ifdef TGVNN_IRIS
 tgv_dmem::check(tgv_dmem::copy(host.data(),pointer,bytes,hipMemcpyDeviceToHost));
#else
 if(hipMemcpy(host.data(),pointer,bytes,hipMemcpyDeviceToHost)!=hipSuccess)std::abort();
#endif
 char path[4096];std::snprintf(path,sizeof(path),"%s.%s.bin",prefix,name);
 FILE* file=std::fopen(path,"wb");if(!file)std::abort();
 if(std::fwrite(host.data(),1,bytes,file)!=bytes||std::fclose(file))std::abort();
}

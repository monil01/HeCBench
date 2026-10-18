
#include <dlfcn.h>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <vector>
// Exact runtime API allocation/free pattern of the source, exported as a narrow host CABI.
static void* handle=nullptr;
static int(*alloc_managed)(void**,size_t,unsigned int)=nullptr;
static int(*alloc_device)(void**,size_t)=nullptr;
static int(*alloc_host)(void**,size_t,unsigned int)=nullptr;
static int(*free_device)(void*)=nullptr;
static int(*free_host)(void*)=nullptr;
static int(*device_sync)()=nullptr;
extern "C" int study_init(char*path,int capacity,int*version){
 handle=dlopen("/usr/lib/x86_64-linux-gnu/libcudart.so.12.0.146",RTLD_NOW|RTLD_LOCAL);if(!handle)return -100;
 alloc_managed=(decltype(alloc_managed))dlsym(handle,"cudaMallocManaged");
 alloc_device=(decltype(alloc_device))dlsym(handle,"cudaMalloc");
 alloc_host=(decltype(alloc_host))dlsym(handle,"cudaHostAlloc");
 free_device=(decltype(free_device))dlsym(handle,"cudaFree");free_host=(decltype(free_host))dlsym(handle,"cudaFreeHost");
 device_sync=(decltype(device_sync))dlsym(handle,"cudaDeviceSynchronize");
 auto set=(int(*)(int))dlsym(handle,"cudaSetDevice");auto ver=(int(*)(int*))dlsym(handle,"cudaRuntimeGetVersion");
 if(!alloc_managed||!alloc_device||!alloc_host||!free_device||!free_host||!device_sync||!set||!ver)return -101;
 Dl_info info;if(!dladdr((void*)alloc_device,&info))return -102;strncpy(path,info.dli_fname,capacity-1);path[capacity-1]=0;
 int status=set(0);if(status)return status;status=ver(version);if(status)return status;return device_sync();
}
extern "C" int study_batch(int type,size_t size,int count,void**p,int*allocation_status,unsigned char*nonnull,int*free_status,double*times){
 if(!handle)return -103;
 // Synchronization is outside allocation/free intervals; source calls themselves are synchronous host APIs.
 int status=device_sync();if(status)return status;
 auto start=std::chrono::steady_clock::now();
 for(int j=0;j<count;j++){
  p[j]=nullptr;
  if(type==0)allocation_status[j]=alloc_managed(&p[j],size,1u); // cudaMemAttachGlobal default source flag
  else if(type==1)allocation_status[j]=alloc_device(&p[j],size);
  else allocation_status[j]=alloc_host(&p[j],size,2u); // cudaHostAllocMapped
  nonnull[j]=p[j]!=nullptr;
 }
 auto stop=std::chrono::steady_clock::now();times[0]=std::chrono::duration<double>(stop-start).count()/count;
 start=std::chrono::steady_clock::now();
 for(int j=0;j<count;j++){
  if(p[j])free_status[j]=type==2?free_host(p[j]):free_device(p[j]);else free_status[j]=-104;
  p[j]=nullptr;
 }
 stop=std::chrono::steady_clock::now();times[1]=std::chrono::duration<double>(stop-start).count()/count;
 int bad=0;for(int j=0;j<count;j++)if(allocation_status[j]||!nonnull[j]||free_status[j]||p[j])bad++;
 status=device_sync();if(status)return status;return bad? -105:0;
}

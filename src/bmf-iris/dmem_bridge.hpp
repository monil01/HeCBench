#pragma once
#include <iris/iris.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <cstdint>
struct DMemEntry {
 void* storage;size_t bytes;iris::DMem memory;bool device_dirty=false;
 DMemEntry(void*p,size_t n):storage(p),bytes(n),memory(p,n){}
};
inline std::map<uintptr_t,std::unique_ptr<DMemEntry>> dmem_entries;
inline std::mutex dmem_mutex;
inline DMemEntry& entry(const void*p){
 auto it=dmem_entries.upper_bound(reinterpret_cast<uintptr_t>(p));
 if(it==dmem_entries.begin())throw std::runtime_error("unregistered DMem");
 --it;if(reinterpret_cast<uintptr_t>(p)>=it->first+it->second->bytes)throw std::runtime_error("DMem range");
 return *it->second;
}
template<class T> void irisAllocate(T**p,size_t n){
 *p=static_cast<T*>(calloc(1,n));if(!*p)throw std::bad_alloc();
 std::lock_guard<std::mutex> lock(dmem_mutex);
 dmem_entries.emplace(reinterpret_cast<uintptr_t>(*p),std::make_unique<DMemEntry>(*p,n));
}
template<class T> void hostAllocate(T**p,size_t n){*p=static_cast<T*>(malloc(n));if(!*p)throw std::bad_alloc();}
inline void irisFree(void*p){std::lock_guard<std::mutex> lock(dmem_mutex);dmem_entries.erase(reinterpret_cast<uintptr_t>(p));free(p);}
enum CopyDirection {HostToDevice,DeviceToHost,HostToHost};
inline void irisCopy(void*dst,const void*src,size_t n,CopyDirection dir,int=0){
 std::lock_guard<std::mutex> lock(dmem_mutex);
 if(dir==DeviceToHost){
  auto&e=entry(src);if(e.device_dirty){iris::Task t;t.flush_out(e.memory);if(t.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("flush");e.device_dirty=false;}
 }else if(dir==HostToDevice){auto&e=entry(dst);e.memory.update(e.storage);e.device_dirty=false;}
 memcpy(dst,src,n);
}
inline void irisSet(void*dst,int value,size_t n,int=0){std::lock_guard<std::mutex> lock(dmem_mutex);auto&e=entry(dst);memset(dst,value,n);e.memory.update(e.storage);e.device_dirty=false;}
inline iris::DMem* irisArgument(const void*p,int access){
 auto&e=entry(p);if(p!=e.storage)throw std::runtime_error("offset launch parameter");if(access!=iris_r)e.device_dirty=true;return &e.memory;
}

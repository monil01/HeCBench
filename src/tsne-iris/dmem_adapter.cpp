#include "dmem_adapter.hpp"
#include "../../tools/iris_vendor_bridge.hpp"
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include <cstring>
#include <cstdio>

namespace iris_dmem {
struct Allocation {
  size_t size;void* host;std::unique_ptr<iris::DMem> mem;
  Allocation(size_t n):size(n),host(std::malloc(n)),mem(std::make_unique<iris::DMem>(host,n)) {if(!host)std::abort();}
  ~Allocation(){mem.reset();std::free(host);}
};
static std::map<uintptr_t,std::unique_ptr<Allocation>> allocations;
static std::recursive_mutex operation_mutex;
static size_t tasks=0;
static int invoke(void* selected,unsigned device,void*const* args){
  auto callback=hecbench_vendor_arg<Callback>(args,1);
  auto context=hecbench_vendor_arg<void*>(args,2);
  try { callback(context,static_cast<hipStream_t>(selected),device); }
  catch(const std::exception& error){std::fprintf(stderr,"IRIS callback exception: %s\n",error.what());return -1;}
  return static_cast<int>(hipStreamSynchronize(static_cast<hipStream_t>(selected)));
}
static void submit(Callback callback,void* context,const std::vector<iris::DMem*>& buffers){
  iris::Task task;
  hecbench_vendor_callback wrapper=invoke;
  void* args[]={&wrapper,&callback,&context};
  int info[]={sizeof(wrapper),sizeof(callback),sizeof(context)};
  size_t one=1;
  if(task.kernel("hecbench_vendor",1,nullptr,&one,&one,3,args,info))std::abort();
  // Embedded DMem addresses stay stable on pinned device zero. The public
  // hidden-memory API tracks all ownership without wrapper argument limits or
  // first-command-only output bookkeeping for multi-command tasks.
  for(auto buffer:buffers)if(task.add_hidden_dmem(buffer,iris_rw))std::abort();
  if(task.submit(0,nullptr,1)!=IRIS_SUCCESS || iris_error_count())std::abort();
  ++tasks;
}
void dispatch(Callback callback,void* context){
  std::lock_guard<std::recursive_mutex> guard(operation_mutex);
  std::vector<iris::DMem*> buffers;
  for(auto& entry:allocations)buffers.push_back(entry.second->mem.get());
  submit(callback,context,buffers);
}
hipError_t allocate(void** result,size_t n){
  std::lock_guard<std::recursive_mutex> guard(operation_mutex);
  if(!n){*result=nullptr;return hipSuccess;}
  auto entry=std::make_unique<Allocation>(n);
  struct Capture{void* pointer=nullptr;} capture;
  // Resolve the physical DMem address through the wrapper's setmem binding.
  hecbench_vendor_callback callback=[](void*,unsigned,void*const* args){
    auto out=hecbench_vendor_arg<Capture*>(args,1);
    out->pointer=hecbench_vendor_arg<void*>(args,2);return 0;
  };
  auto out=&capture;void* args[]={&callback,&out,entry->mem.get()};
  int info[]={sizeof(callback),sizeof(out),iris_w};size_t one=1;
  iris::Task task;task.kernel("hecbench_vendor",1,nullptr,&one,&one,3,args,info);
  if(task.submit(0,nullptr,1) || !capture.pointer)std::abort();
  *result=capture.pointer;allocations.emplace(reinterpret_cast<uintptr_t>(*result),std::move(entry));
  return hipSuccess;
}
static std::pair<Allocation*,size_t> lookup(const void* p,size_t size){
  auto address=reinterpret_cast<uintptr_t>(p);auto it=allocations.upper_bound(address);
  if(it==allocations.begin())std::abort();--it;
  size_t offset=address-it->first;
  if(offset>it->second->size || size>it->second->size-offset)std::abort();
  return {it->second.get(),offset};
}
static void flush(Allocation* allocation){
  iris::Task task;task.flush_out(*allocation->mem);
  if(task.submit(0,nullptr,1))std::abort();
}
__global__ void copy_bytes(unsigned char* destination,const unsigned char* source,size_t n){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)destination[i]=source[i];
}
__global__ void fill_bytes(unsigned char* destination,unsigned char value,size_t n){
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n)destination[i]=value;
}
hipError_t copy(void* destination,const void* source,size_t n,hipMemcpyKind kind){
  std::lock_guard<std::recursive_mutex> guard(operation_mutex);
  if(!n)return hipSuccess;
  auto move_gpu=[&](void* d,const void* s){kernel(copy_bytes,dim3((n+255)/256),dim3(256),static_cast<unsigned char*>(d),static_cast<const unsigned char*>(s),n);};
  if(kind==hipMemcpyDeviceToHost){
    lookup(source,n);void* staging;allocate(&staging,n);
    move_gpu(staging,source);auto [a,offset]=lookup(staging,n);flush(a);
    std::memcpy(destination,a->host,n);release(staging);
  }else if(kind==hipMemcpyHostToDevice){
    lookup(destination,n);void* staging;allocate(&staging,n);
    auto [a,offset]=lookup(staging,n);std::memcpy(a->host,source,n);a->mem->update(a->host);
    move_gpu(destination,staging);release(staging);
  }else if(kind==hipMemcpyDeviceToDevice){lookup(source,n);lookup(destination,n);move_gpu(destination,source);}
  else std::memmove(destination,source,n);
  return hipSuccess;
}
hipError_t fill(void* destination,int value,size_t n){
  std::lock_guard<std::recursive_mutex> guard(operation_mutex);
  if(!n)return hipSuccess;lookup(destination,n);
  kernel(fill_bytes,dim3((n+255)/256),dim3(256),static_cast<unsigned char*>(destination),static_cast<unsigned char>(value),n);
  return hipSuccess;
}
hipError_t release(void* p){std::lock_guard<std::recursive_mutex> guard(operation_mutex);if(p)allocations.erase(reinterpret_cast<uintptr_t>(p));return hipSuccess;}
hipError_t synchronize(){return iris_synchronize()==IRIS_SUCCESS?hipSuccess:hipErrorUnknown;}
hipError_t reset(){return synchronize();}
hipError_t set_limit(hipLimit_t limit,size_t bytes){
  return call([=](hipStream_t){
    auto status=hipDeviceSetLimit(limit,bytes);size_t actual=0;
    auto queried=hipDeviceGetLimit(&actual,limit);
    std::fprintf(stderr,"IRIS worker resource limit=%d requested=%zu actual=%zu status=%d query=%d\n",int(limit),bytes,actual,int(status),int(queried));
    if(status!=hipSuccess || queried!=hipSuccess)std::abort();
    return status;
  });
}
hipError_t get_limit(size_t* bytes,hipLimit_t limit){return call([=](hipStream_t){return hipDeviceGetLimit(bytes,limit);});}
hipError_t memory_info(size_t* available,size_t* total){return call([=](hipStream_t){return hipMemGetInfo(available,total);});}
hipError_t last_error(){return call([](hipStream_t){return hipGetLastError();});}
hipError_t peek_error(){return call([](hipStream_t){return hipPeekAtLastError();});}
void cleanup(){std::lock_guard<std::recursive_mutex> guard(operation_mutex);allocations.clear();std::printf("IRIS DMem adapter tasks: %zu; actual HIP only\n",tasks);}
}

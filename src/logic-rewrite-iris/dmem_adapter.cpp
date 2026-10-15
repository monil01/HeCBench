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
  std::vector<iris::DMem*> buffers;
  for(auto& entry:allocations)buffers.push_back(entry.second->mem.get());
  submit(callback,context,buffers);
}
hipError_t allocate(void** result,size_t n){
  if(!n){*result=nullptr;return hipSuccess;}
  auto entry=std::make_unique<Allocation>(n);
  struct Capture{void* pointer=nullptr;} capture;
  // Resolve the physical DMem address through the wrapper's setmem binding.
  hecbench_vendor_callback callback=[](void*,unsigned,void*const* args){
    auto out=hecbench_vendor_arg<Capture*>(args,1);
    out->pointer=hecbench_vendor_arg<void*>(args,2);return 0;
  };
  auto out=&capture;void* args[]={&callback,&out,entry->mem.get()};
  int info[]={sizeof(callback),sizeof(out),iris_rw};size_t one=1;
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
hipError_t copy(void* destination,const void* source,size_t n,hipMemcpyKind kind){
  if(!n)return hipSuccess;
  if(kind==hipMemcpyDeviceToHost){auto [a,offset]=lookup(source,n);flush(a);std::memcpy(destination,static_cast<char*>(a->host)+offset,n);}
  else if(kind==hipMemcpyHostToDevice){auto [a,offset]=lookup(destination,n);flush(a);std::memcpy(static_cast<char*>(a->host)+offset,source,n);a->mem->update(a->host);}
  else if(kind==hipMemcpyDeviceToDevice){
    // Preserve values through IRIS-owned host staging; no native device copies.
    // This explicit P1 adaptation changes transfer performance, not graph math.
    auto [a,offset]=lookup(source,n);auto [b,target]=lookup(destination,n);flush(a);if(a!=b)flush(b);
    std::memmove(static_cast<char*>(b->host)+target,static_cast<char*>(a->host)+offset,n);b->mem->update(b->host);
  }else std::memmove(destination,source,n);
  return hipSuccess;
}
hipError_t fill(void* destination,int value,size_t n){
  if(!n)return hipSuccess;auto [a,offset]=lookup(destination,n);flush(a);
  std::memset(static_cast<char*>(a->host)+offset,value,n);a->mem->update(a->host);return hipSuccess;
}
hipError_t release(void* p){if(p)allocations.erase(reinterpret_cast<uintptr_t>(p));return hipSuccess;}
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
void cleanup(){allocations.clear();std::printf("IRIS DMem adapter tasks: %zu; actual HIP only\n",tasks);}
}

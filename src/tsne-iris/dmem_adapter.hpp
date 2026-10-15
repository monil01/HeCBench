#pragma once
#include <iris/iris.hpp>
#include <hip/hip_runtime.h>
#if defined(__HIPCC__)
#include <thrust/system/hip/execution_policy.h>
#endif
#include <type_traits>
#include <optional>
#include <tuple>
#include <utility>
#include <cstddef>
#include <cstdio>

namespace iris_dmem {
using Callback = void (*)(void*,hipStream_t,unsigned);
void dispatch(Callback,void*);
template<class F> auto call(F&& operation) {
  using Fn=std::remove_reference_t<F>;
  using R=std::invoke_result_t<Fn,hipStream_t>;
  if constexpr(std::is_void_v<R>) {
    dispatch([](void* context,hipStream_t stream,unsigned){(*static_cast<Fn*>(context))(stream);},&operation);
  } else {
    std::optional<R> result;
    auto callback=[&](hipStream_t stream){result.emplace(operation(stream));};
    using C=decltype(callback);
    dispatch([](void* context,hipStream_t stream,unsigned){(*static_cast<C*>(context))(stream);},&callback);
    return *result;
  }
}
hipError_t allocate(void**,size_t);
template<class T> hipError_t allocate(T** p,size_t n){return allocate(reinterpret_cast<void**>(p),n);}
hipError_t release(void*);
hipError_t copy(void*,const void*,size_t,hipMemcpyKind);
inline hipError_t copy_async(void* d,const void* s,size_t n,hipMemcpyKind kind,hipStream_t=0){return copy(d,s,n,kind);}
hipError_t fill(void*,int,size_t);
hipError_t synchronize();
hipError_t reset();
hipError_t set_limit(hipLimit_t,size_t);
hipError_t get_limit(size_t*,hipLimit_t);
hipError_t memory_info(size_t*,size_t*);
hipError_t last_error();
hipError_t peek_error();
void cleanup();
template<class... P,class... A> void kernel_shared(void (*function)(P...),dim3 grid,dim3 block,size_t shared,A... arguments) {
  using F=decltype(function);
  // Native kernel calls convert every argument to the declared parameter type.
  // Preserve those conversions before passing the raw kernel argument array.
  std::tuple<std::decay_t<P>...> values(arguments...);
  void* params[sizeof...(A)];
  std::apply([&](auto&... v){size_t i=0;((params[i++]=&v),...);},values);
  struct Launch { F function;dim3 grid,block;size_t shared;void**params; } launch{function,grid,block,shared,params};
  dispatch([](void* context,hipStream_t stream,unsigned device){
    auto& l=*static_cast<Launch*>(context);hipFunction_t native;
    std::fprintf(stderr,"IRIS GPU kernel: %s grid=%u block=%u\n",hipKernelNameRefByPtr(reinterpret_cast<const void*>(l.function),stream),l.grid.x,l.block.x);
    if(hipGetFuncBySymbol(&native,reinterpret_cast<const void*>(l.function))!=hipSuccess) std::abort();
    if(iris_vendor_kernel_launch(device,reinterpret_cast<void*>(native),l.grid.x,l.grid.y,l.grid.z,
         l.block.x,l.block.y,l.block.z,l.shared,reinterpret_cast<void*>(stream),l.params)!=IRIS_SUCCESS) std::abort();
    if(hipStreamSynchronize(stream)!=hipSuccess)std::abort();
  },&launch);
}
}

namespace iris_dmem {
template<class... P,class... A> void kernel(void(*function)(P...),dim3 grid,dim3 block,A... args){kernel_shared(function,grid,block,0,args...);}
}

#if defined(__HIPCC__)
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
namespace iris_dmem {
template<class T> T read(const T* address){T value;if(copy(&value,address,sizeof(T),hipMemcpyDeviceToHost)!=hipSuccess)std::abort();return value;}
// Same persistent-vector value initialization and iterator interface, but
// IRIS owns all device storage and tracks every GPU operation's dependencies.
template<class T> class vector {
 T* pointer_=nullptr;size_t count_=0;
 void reserve(size_t n){count_=n;if(n && allocate(&pointer_,n*sizeof(T))!=hipSuccess)std::abort();}
 public:
 using value_type=T;
 using iterator=thrust::device_ptr<T>;
 using const_iterator=thrust::device_ptr<const T>;
 vector()=default;
 explicit vector(size_t n){reserve(n);if(n && fill(pointer_,0,n*sizeof(T))!=hipSuccess)std::abort();}
 vector(size_t n,const T& value){reserve(n);if(n)call([&](hipStream_t stream){thrust::fill(thrust::hip::par.on(stream),begin(),end(),value);});}
 vector(const T* first,const T* last){reserve(last-first);if(count_ && copy(pointer_,first,count_*sizeof(T),hipMemcpyHostToDevice)!=hipSuccess)std::abort();}
 vector(const vector&)=delete;vector& operator=(const vector&)=delete;
 ~vector(){clear();}
 size_t size()const{return count_;}
 iterator data(){return iterator(pointer_);}const_iterator data()const{return const_iterator(pointer_);}
 iterator begin(){return data();}iterator end(){return data()+count_;}
 const_iterator begin()const{return data();}const_iterator end()const{return data()+count_;}
 T operator[](size_t i)const{if(i>=count_)std::abort();return read(pointer_+i);}
 void clear(){if(pointer_ && release(pointer_)!=hipSuccess)std::abort();pointer_=nullptr;count_=0;}
 void shrink_to_fit(){}
};
}
#endif

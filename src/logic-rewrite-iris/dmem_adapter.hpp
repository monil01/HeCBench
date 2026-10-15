#pragma once
#include <iris/iris.hpp>
#include <hip/hip_runtime.h>
#include <thrust/system/hip/execution_policy.h>
#include <type_traits>
#include <optional>
#include <tuple>
#include <utility>
#include <cstddef>

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
hipError_t fill(void*,int,size_t);
hipError_t synchronize();
hipError_t reset();
hipError_t set_limit(hipLimit_t,size_t);
hipError_t get_limit(size_t*,hipLimit_t);
hipError_t memory_info(size_t*,size_t*);
hipError_t last_error();
hipError_t peek_error();
void cleanup();
template<class... P,class... A> void kernel(void (*function)(P...),dim3 grid,dim3 block,A... arguments) {
  using F=decltype(function);
  // Native kernel calls convert every argument to the declared parameter type.
  // Preserve those conversions before passing the raw kernel argument array.
  std::tuple<std::decay_t<P>...> values(arguments...);
  void* params[sizeof...(A)];
  std::apply([&](auto&... v){size_t i=0;((params[i++]=&v),...);},values);
  struct Launch { F function;dim3 grid,block;void**params; } launch{function,grid,block,params};
  dispatch([](void* context,hipStream_t stream,unsigned device){
    auto& l=*static_cast<Launch*>(context);hipFunction_t native;
    if(hipGetFuncBySymbol(&native,reinterpret_cast<const void*>(l.function))!=hipSuccess) std::abort();
    if(iris_vendor_kernel_launch(device,reinterpret_cast<void*>(native),l.grid.x,l.grid.y,l.grid.z,
         l.block.x,l.block.y,l.block.z,0,reinterpret_cast<void*>(stream),l.params)!=IRIS_SUCCESS) std::abort();
    if(hipStreamSynchronize(stream)!=hipSuccess)std::abort();
  },&launch);
}
}

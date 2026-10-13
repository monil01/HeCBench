#pragma once
#include <iris/iris_runtime.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>
#include <type_traits>
#include <array>
#include <cstdint>

inline void iris_check(int rc) {
  if (rc != IRIS_SUCCESS) { fprintf(stderr, "IRIS failure %d\n", rc); exit(2); }
}
// Native-shaped host handles below refer only to host storage + IRIS data memory.
// Every computation is an IRIS HIP task; no native allocation/copy/launch is used.
struct IrisBuffer { void* host; size_t bytes; iris_mem mem; bool dirty = false; };
struct IrisState {
  std::map<uintptr_t, IrisBuffer> buffers;
  int gpu = -1;
  bool finalized = false;
  IrisState() {
    int argc=0; char** argv=nullptr;
    iris_check(iris_init(&argc, &argv, 1));
    int n=0; iris_check(iris_device_count(&n));
    for(int d=0;d<n;++d) {
      int type=0; iris_check(iris_device_info(d,iris_type,&type,nullptr));
      if((type & iris_amd) && gpu<0) gpu=d;
    }
    if(gpu<0) { fprintf(stderr,"No AMD GPU\n"); exit(2); }
    iris_overview();
  }
  void finish() {
    if (finalized) return;
    for(auto &p:buffers) { iris_mem_release(p.second.mem); free(p.second.host); }
    buffers.clear();
    iris_check(iris_finalize());
    finalized = true;
  }
  ~IrisState() { if (!finalized) finish(); }
};
inline IrisState& iris_state() { static IrisState state; return state; }
inline void iris_finish() { iris_state().finish(); }
inline IrisBuffer& iris_find(const void* ptr) {
  auto &b=iris_state().buffers;
  auto it=b.upper_bound(reinterpret_cast<uintptr_t>(ptr));
  if(it==b.begin()) { fprintf(stderr,"Unknown DMem pointer\n"); exit(2); }
  --it;
  if(reinterpret_cast<uintptr_t>(ptr)-it->first>=it->second.bytes) {
    fprintf(stderr,"Out-of-range DMem pointer\n"); exit(2);
  }
  return it->second;
}
inline void iris_flush(IrisBuffer& b) {
  if(!b.dirty) return;
  iris_task t; iris_check(iris_task_create(&t));
  iris_check(iris_task_dmem_flush_out(t,b.mem));
  iris_check(iris_task_submit(t,iris_state().gpu,nullptr,1));
  // Non-permanent submitted tasks are owned and released by IRIS.
  b.dirty=false;
}
inline int iris_alloc(void** ptr,size_t bytes) {
  auto &state=iris_state();
  void* h=calloc(1,bytes);
  if(!h) return -1;
  IrisBuffer b{h,bytes,{},false};
  iris_check(iris_data_mem_create(&b.mem,h,bytes));
  state.buffers.emplace(reinterpret_cast<uintptr_t>(h),b); *ptr=h;
  return IRIS_SUCCESS;
}
template<class T> inline int iris_alloc(T** ptr,size_t bytes) { return iris_alloc(reinterpret_cast<void**>(ptr),bytes); }
inline int iris_free(void* ptr) {
  if(!ptr) return IRIS_SUCCESS;
  auto &b=iris_find(ptr);
  iris_check(iris_mem_release(b.mem));
  iris_state().buffers.erase(reinterpret_cast<uintptr_t>(ptr));
  free(ptr); return IRIS_SUCCESS;
}
enum IrisCopyKind { iris_h2d, iris_d2h, iris_d2d };
inline int iris_copy(void* dst,const void* src,size_t bytes,IrisCopyKind kind) {
  if(kind==iris_d2h || kind==iris_d2d) iris_flush(iris_find(src));
  if(kind==iris_h2d || kind==iris_d2d) {
    auto &b=iris_find(dst);
    if(dst!=b.host || bytes!=b.bytes) iris_flush(b);
    memcpy(dst,src,bytes);
    iris_check(iris_data_mem_update(b.mem,b.host)); b.dirty=false;
  } else memcpy(dst,src,bytes);
  return IRIS_SUCCESS;
}
inline int iris_fill(void* dst,int value,size_t bytes) {
  auto &b=iris_find(dst);
  if(dst!=b.host || bytes!=b.bytes) iris_flush(b);
  memset(dst,value,bytes);
  iris_check(iris_data_mem_update(b.mem,b.host)); b.dirty=false;
  return IRIS_SUCCESS;
}
inline int iris_sync() { return IRIS_SUCCESS; }
template<class T> inline void iris_param(T& arg, void*& p,int& mode,size_t& offset) {
  if constexpr(std::is_pointer_v<T>) {
    auto &b=iris_find(reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(arg)));
    p=&b.mem; offset=reinterpret_cast<uintptr_t>(arg)-reinterpret_cast<uintptr_t>(b.host);
    mode=std::is_const_v<std::remove_pointer_t<T>> ? iris_r : iris_rw;
    if(mode!=iris_r) b.dirty=true;
  } else { p=&arg; mode=sizeof(T); offset=0; }
}
template<class Tuple,size_t... I>
inline void iris_launch_tuple(const char* name,dim3 grid,dim3 block,Tuple& args,std::index_sequence<I...>) {
  constexpr size_t N=sizeof...(I);
  std::array<void*,N> params{};
  std::array<int,N> access{};
  std::array<size_t,N> offsets{};
  (iris_param(std::get<I>(args),params[I],access[I],offsets[I]),...);
  size_t gws[]={size_t(grid.x)*block.x,size_t(grid.y)*block.y,size_t(grid.z)*block.z};
  size_t lws[]={block.x,block.y,block.z};
  int dim=(gws[2]>1)?3:((gws[1]>1)?2:1);
  iris_task task; iris_check(iris_task_create(&task));
  iris_check(iris_task_kernel_v2(task,name,dim,nullptr,gws,lws,N,params.data(),offsets.data(),access.data()));
  iris_check(iris_task_submit(task,iris_state().gpu,nullptr,1));
  // Match iris::Task: runtime owns a submitted non-permanent task.
}
template<class... A> inline void iris_launch(const char* name,dim3 grid,dim3 block,A... a) {
  auto args=std::make_tuple(a...);
  iris_launch_tuple(name,grid,block,args,std::index_sequence_for<A...>{});
}

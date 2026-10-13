#pragma once
#include <iris/iris.hpp>
#include <cstdlib>
#include <utility>
struct Dim {size_t x,y,z;Dim(size_t a=1,size_t b=1,size_t c=1):x(a),y(b),z(c){}};
template<class T> void* ew_ptr(T& a) {return const_cast<void*>(static_cast<const void*>(&a));}
template<class T> int ew_info(T&) {return sizeof(T);}
inline int ew_info(iris::DMem&) {return iris_rw;}
inline int ew_info(const iris::DMem&) {return iris_r;}
template<class... A> void ew_launch(const char* name,Dim grid,Dim block,A&&... a) {
 void* args[]={ew_ptr(a)...};int flags[]={ew_info(a)...};
 size_t gws[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},lws[]={block.x,block.y,block.z};
 iris::Task task;
 if(task.kernel(name,3,nullptr,gws,lws,sizeof...(A),args,flags)!=IRIS_SUCCESS ||
 task.submit(0,nullptr,1)!=IRIS_SUCCESS) std::exit(1);
}
template<class... M> void ew_flush(M&... mem) {
 iris::Task out;
 if(((out.flush_out(mem)!=IRIS_SUCCESS)||...) || out.submit(0,nullptr,1)!=IRIS_SUCCESS) std::exit(1);
}

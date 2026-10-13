#pragma once
#include <iris/iris.hpp>
#include <cstdlib>
#include <utility>
struct Dim {size_t x,y,z;Dim(size_t a=1,size_t b=1,size_t c=1):x(a),y(b),z(c){}};
template<class T> void* ptr(T& a) {return const_cast<void*>(static_cast<const void*>(&a));}
template<class T> int info(T&) {return sizeof(T);}
inline int info(iris::DMem&) {return iris_rw;}
inline int info(const iris::DMem&) {return iris_r;}
template<class... A> void launch(const char* name,Dim grid,Dim block,A&&... a) {
 void* args[]={ptr(a)...};int flags[]={info(a)...};
 size_t gws[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},lws[]={block.x,block.y,block.z};
 iris::Task task;
 if(task.kernel(name,3,nullptr,gws,lws,sizeof...(A),args,flags)!=IRIS_SUCCESS ||
 task.submit(0,nullptr,1)!=IRIS_SUCCESS) std::exit(1);
}


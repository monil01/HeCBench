#pragma once
#include <iris/iris.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <type_traits>
#include <algorithm>
using std::min;
struct alignas(4) uchar4 {unsigned char x,y,z,w;};
struct Dim {size_t x,y,z;Dim(size_t a=1,size_t b=1,size_t c=1):x(a),y(b),z(c){}};
template<class A> void* arg(A& a) {return const_cast<void*>(static_cast<const void*>(&a));}
inline void* arg(iris::DMem*& a) {return a;}
template<class A> int info(A&) {return sizeof(A);}
inline int info(iris::DMem*&) {return iris_rw;}
template<class... A> void launch(const char* name,Dim grid,Dim block,A&&... a) {
 void* args[]={arg(a)...};int flags[]={info(a)...};
 size_t gws[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},lws[]={block.x,block.y,block.z};
 iris::Task task;
 if(task.kernel(name,3,nullptr,gws,lws,sizeof...(A),args,flags)!=IRIS_SUCCESS ||
 task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) std::exit(1);
}


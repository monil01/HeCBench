#pragma once
#include "memory.h"
#include "utils.hpp"
struct Dim3 { size_t x,y,z; Dim3(size_t a=1,size_t b=1,size_t c=1):x(a),y(b),z(c) {} };
inline void launch_init_c(Dim3 grid,Dim3 block,iris::DMem *c,real inc,real c0) {
 iris::Task task;size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={c,&inc,&c0};
 int info[]={iris_rw,sizeof(inc),sizeof(c0)};
 task.kernel("init_c",3,nullptr,g,l,3,p,info);
 task.flush_out(*c);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
inline void launch_init_half(Dim3 grid,Dim3 block,iris::DMem *scalco,iris::DMem *gx,iris::DMem *gy,iris::DMem *sx,iris::DMem *sy,iris::DMem *h) {
 iris::Task task;size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={scalco,gx,gy,sx,sy,h};
 int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_rw};
 task.kernel("init_half",3,nullptr,g,l,6,p,info);
 task.flush_out(*h);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
inline void launch_compute_semblances(Dim3 grid,Dim3 block,iris::DMem *h,iris::DMem *c,iris::DMem *samples,iris::DMem *num,iris::DMem *stt,int t_id0,int t_idf,real _idt,real _dt,int _tau,int _w,int nc,int ns) {
 iris::Task task;size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={h,c,samples,num,stt,&t_id0,&t_idf,&_idt,&_dt,&_tau,&_w,&nc,&ns};
 int info[]={iris_r,iris_r,iris_r,iris_rw,iris_rw,sizeof(t_id0),sizeof(t_idf),sizeof(_idt),sizeof(_dt),sizeof(_tau),sizeof(_w),sizeof(nc),sizeof(ns)};
 task.kernel("compute_semblances",3,nullptr,g,l,13,p,info);
 task.flush_out(*num); task.flush_out(*stt);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}
inline void launch_redux_semblances(Dim3 grid,Dim3 block,iris::DMem *num,iris::DMem *stt,iris::DMem *ctr,iris::DMem *str,iris::DMem *stk,int nc,int cdp_id,int ns) {
 iris::Task task;size_t g[]={grid.x*block.x,grid.y*block.y,grid.z*block.z},l[]={block.x,block.y,block.z};
 void *p[]={num,stt,ctr,str,stk,&nc,&cdp_id,&ns};
 int info[]={iris_r,iris_r,iris_rw,iris_rw,iris_rw,sizeof(nc),sizeof(cdp_id),sizeof(ns)};
 task.kernel("redux_semblances",3,nullptr,g,l,8,p,info);
 task.flush_out(*ctr); task.flush_out(*str); task.flush_out(*stk);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS) exit(3);
}

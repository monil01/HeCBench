#pragma once
#include <iris/iris.hpp>
#include <memory>
#include <cstdio>
#include <cstdlib>
inline void iris_check(int rc) { if(rc!=IRIS_SUCCESS) { fprintf(stderr,"IRIS error %d\n",rc);exit(1); } }
inline void launch_sss(const char *name,int count,iris::DMem &input,iris::DMem &output) {
  if(count==0)return;
  size_t gws[]={size_t(count)*32},lws[]={32};void *args[]={&input,&output};int info[]={iris_r,iris_rw};
  iris::Task task;iris_check(task.kernel(name,1,nullptr,gws,lws,2,args,info));iris_check(task.flush_out(output));iris_check(task.submit(iris_gpu,nullptr,true));
}
inline void launch_sss(const char *name,int count,iris::DMem &input,iris::DMem &extra,iris::DMem &output) {
  if(count==0)return;
  size_t gws[]={size_t(count)*32},lws[]={32};void *args[]={&input,&extra,&output};int info[]={iris_r,iris_r,iris_rw};
  iris::Task task;iris_check(task.kernel(name,1,nullptr,gws,lws,3,args,info));iris_check(task.flush_out(output));iris_check(task.submit(iris_gpu,nullptr,true));
}

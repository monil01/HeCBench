#pragma once
#include <iris/iris.hpp>
#include <array>
#include <algorithm>
#include <vector>
#include <cstdio>
#include "kernel.h"
#define LOG2_GRID_SIZE 14
#define NUMTHREADS 128
#define MAXNUMBLOCKS 16384
#define MAXBLOCKSEND 32
#define DATAPERBLOCK (NUMTHREADS*63)
#define MEMPERBLOCK (NUMTHREADS*NUMBINS)

struct device_cartesian { iris::DMem *x,*y,*z; };
static std::array<double,31> boundary_values;
static iris::DMem* boundary_mem;
static std::vector<unsigned int> histogram_values;
static iris::DMem* histogram_mem;
static int iDivUp(int a,int b) { return (a%b!=0)?a/b+1:a/b; }
static void launchSymm(device_cartesian& input, iris::DMem& output) {
  iris::Task task;
  void* params[]={input.x,input.y,input.z,&output,boundary_mem};
  int info[]={iris_r,iris_r,iris_r,iris_w,iris_r};
  size_t global[]={16384,128},local[]={128,1};
  task.kernel("ACFKernelSymm",2,nullptr,global,local,5,params,info);
  task.submit(iris_gpu,nullptr,1);
}
static void launchPair(device_cartesian& first, device_cartesian& second, iris::DMem& output) {
  iris::Task task;
  void* params[]={first.x,first.y,first.z,second.x,second.y,second.z,&output,boundary_mem};
  int info[]={iris_r,iris_r,iris_r,iris_r,iris_r,iris_r,iris_w,iris_r};
  size_t global[]={16384,128},local[]={128,1};
  task.kernel("ACFKernel",2,nullptr,global,local,8,params,info);
  task.submit(iris_gpu,nullptr,1);
}
static void histoInit() {
  histogram_values.resize(MAXNUMBLOCKS*NUMBINS);
  histogram_mem=new iris::DMem(histogram_values.data(),histogram_values.size()*sizeof(unsigned int));
}
static void histoClose() { delete histogram_mem; }
static void GPUHistogram(unsigned int* result,iris::DMem* input,int offset,int num) {
  int size=num>>2;
  int blocks=iDivUp(size,DATAPERBLOCK);
  if(blocks>MAXNUMBLOCKS) { printf("Data set too large for GPU Histogram."); return; }
  iris::Task task;
  void* params[]={histogram_mem,input,&size,&offset};
  int info[]={iris_w,iris_r,sizeof(size),sizeof(offset)};
  size_t global[]={static_cast<size_t>(blocks)*NUMTHREADS},local[]={NUMTHREADS};
  task.kernel("histoKernel",1,nullptr,global,local,4,params,info);
  if(blocks<=MAXBLOCKSEND) task.flush_out(*histogram_mem);
  task.submit(iris_gpu,nullptr,1);
  if(blocks>MAXBLOCKSEND) {
    iris::Task merge;
    void* mparams[]={histogram_mem,&blocks};
    int minfo[]={iris_rw,sizeof(blocks)};
    size_t mg[]={MAXBLOCKSEND*NUMBINS},ml[]={NUMBINS};
    merge.kernel("mergeKernel",1,nullptr,mg,ml,2,mparams,minfo);
    merge.flush_out(*histogram_mem);
    merge.submit(iris_gpu,nullptr,1);
  }
  int end=std::min(MAXBLOCKSEND,blocks);
  for(int i=1;i<end;++i) for(int j=0;j<NUMBINS;++j) histogram_values[j]+=histogram_values[i*NUMBINS+j];
  for(int i=0;i<NUMBINS;++i) result[i]=histogram_values[i];
}

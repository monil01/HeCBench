#include <iris/iris.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "FDTD3dGPU.h"
#include "shrUtils.h"
bool fdtdGPU(float* output,const float* input,const float* coeff,const int dimx,const int dimy,const int dimz,const int radius,const int timesteps,const int argc,const char** argv) {
  iris::Platform platform;if(platform.init(nullptr,nullptr,1)!=IRIS_SUCCESS)return false;
  bool gpu=false;for(int d=0;d<platform.ndevs();d++) {
    int backend=0,type=0;char name[256]={};size_t bytes=0;
    iris_device_info(d,iris_backend,&backend,&bytes);iris_device_info(d,iris_type,&type,&bytes);iris_device_info(d,iris_name,name,&bytes);
    printf("DEVICE backend=%d type=%d name=%s\n",backend,type,name);gpu |= backend==3 && type==256;
  }if(!gpu)return false;
  size_t volume=size_t(dimx+2*radius)*(dimy+2*radius)*(dimz+2*radius);
  int padding=128/sizeof(float)-radius,x=dimx,y=dimy,z=dimz;
  std::vector<float> a(volume+padding),b(volume+padding);
  std::copy(input,input+volume,a.begin()+padding);std::copy(input,input+volume,b.begin()+padding);
  iris::DMem ma(a.data(),a.size()*sizeof(float)),mb(b.data(),b.size()*sizeof(float)),mc(const_cast<float*>(coeff),(radius+1)*sizeof(float));
  iris::DMem* in=&ma;iris::DMem* out=&mb;
  size_t lws[]={32,8},gws[]={size_t((dimx+31)/32)*32,size_t((dimy+7)/8)*8};
  shrLog(" set block size to %dx%d\n",32,8);
  shrLog(" set grid size to %dx%d\n",(dimx+31)/32,(dimy+7)/8);
  auto start=std::chrono::steady_clock::now();
  for(int it=0;it<timesteps;it++) {
    iris::Task task;void* args[]={out,in,&mc,&x,&y,&z,&padding};int info[]={iris_rw,iris_r,iris_r,sizeof(x),sizeof(y),sizeof(z),sizeof(padding)};
    if(task.kernel("finite_difference",2,nullptr,gws,lws,7,args,info)!=IRIS_SUCCESS)return false;
    if(it==timesteps-1)task.flush_out(*out);
    if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return false;
    std::swap(in,out);
  }
  printf("Average kernel execution time %f (s)\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/timesteps);
  memcpy(output,(in==&ma ? a.data() : b.data())+padding,volume*sizeof(float));
  if(const char* path=getenv("HECBENCH_ARTIFACT")){FILE* f=fopen(path,"wb");if(!f)return false;fwrite(output,sizeof(float),volume,f);fclose(f);}
  int errors=platform.error_count();platform.finalize();return !errors;
}

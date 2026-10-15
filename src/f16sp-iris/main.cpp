#include <iris/iris.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include "config.h"
struct HalfPair {_Float16 x,y;};
static_assert(sizeof(HalfPair)==4,"packed half2 layout");
void check(int rc){if(rc!=IRIS_SUCCESS)throw std::runtime_error("IRIS failure");}
void hip_device(){int count=0;check(iris_device_count(&count));for(int d=0;d<count;++d){int b=0,t=0;char n[256]={};check(iris_device_info(d,iris_backend,&b,nullptr));check(iris_device_info(d,iris_type,&t,nullptr));check(iris_device_info(d,iris_name,n,nullptr));if(b==iris_hip&&(t&iris_gpu)){printf("IRIS backend=%d type=%d name=%s\n",b,t,n);check(iris_device_set_default(d));return;}}throw std::runtime_error("HIP GPU required");}
void launch(const char* name,size_t blocks,size_t threads,int count,void**args,int*info,iris::DMem*output=nullptr){
 iris::Task task;size_t g[]={blocks*threads},l[]={threads};check(task.kernel(name,1,nullptr,g,l,count,args,info));if(output)check(task.flush_out(*output));check(task.submit(iris_gpu,nullptr,1));
}
int main(int argc,char**argv){
 if(argc!=2){printf("Usage: %s <repeat>\n",argv[0]);return 1;}int repeat=atoi(argv[1]);if(repeat<1)return 1;
 setvbuf(stdout,nullptr,_IOLBF,0);iris::Platform platform;check(platform.init(&argc,&argv,1));hip_device();
 {
 const size_t size=(size_t)NUM_OF_BLOCKS*NUM_OF_THREADS;size_t bytes=size*sizeof(HalfPair);
 std::vector<HalfPair>a(size),b(size);srand(123);
 for(size_t i=0;i<size;++i){a[i].x=a[i].y=(_Float16)sqrt(32752.0/size);b[i]=a[i];}
 float result=0;_Float16 half_result=0;int blocks=1024;std::vector<float>partial(blocks,0.f);
 iris::DMem ma(a.data(),bytes),mb(b.data(),bytes),mr(&result,sizeof(result)),mh(&half_result,sizeof(half_result)),mp(partial.data(),partial.size()*sizeof(float));
 printf("\nNumber of elements in the vectors is %zu\n",size*2);
 const char* names[]={"scalarProductKernel_intrinsics","scalarProductKernel_native_fp32","scalarProductKernel_native2_fp32"};
 auto report=[&](int grid,int path,float value){printf("RESULT grid=%d path=%d value=%a\n",grid,path,(double)value);printf("Error rate: %e\n",fabsf(value-65504.f)/65504.f);};
 for(int grid=NUM_OF_BLOCKS;grid>=NUM_OF_BLOCKS/16;grid/=2){
  printf("\nGPU grid size is %d\n",grid);
  for(int path=0;path<3;++path){
   auto once=[&](bool flush){void*c[]={&mr};int ci[]={iris_w};launch("clear_result",1,1,1,c,ci);void*p[]={&ma,&mb,&mr,(void*)&size};int pi[]={iris_r,iris_r,iris_rw,sizeof(size)};launch(names[path],grid,NUM_OF_THREADS,4,p,pi,flush?&mr:nullptr);};
   for(int i=0;i<F16SP_WARMUP;++i)once(false);
   auto begin=std::chrono::steady_clock::now();for(int i=0;i<repeat;++i)once(i==repeat-1);auto end=std::chrono::steady_clock::now();
   printf("Average kernel%d execution time %f (us)\n",path+1,std::chrono::duration<double,std::micro>(end-begin).count()/repeat);report(grid,path,result);
  }
 }
 for(int rounded=0;rounded<2;++rounded){
  auto once=[&](bool flush){void*p[]={&ma,&mb,&mp,(void*)&size,&rounded};int pi[]={iris_r,iris_r,iris_w,sizeof(size),sizeof(rounded)};launch("vendor_partial",blocks,256,5,p,pi);void*q[]={&mp,&mr,&mh,&blocks,&rounded};int qi[]={iris_r,iris_w,iris_w,sizeof(blocks),sizeof(rounded)};launch("vendor_finish",1,256,5,q,qi,flush?(rounded?&mr:&mh):nullptr);};
  for(int i=0;i<F16SP_WARMUP;++i)once(false);
  auto begin=std::chrono::steady_clock::now();for(int i=0;i<repeat;++i)once(i==repeat-1);auto end=std::chrono::steady_clock::now();
  printf("Average %s equivalent execution time %f (us)\n",rounded?"cub::DeviceReduce::Sum":"cublasDotEx_64",std::chrono::duration<double,std::micro>(end-begin).count()/repeat);report(0,rounded+3,rounded?result:(float)half_result);
 }
 }
 check(platform.finalize());return 0;
}

/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <iris/iris.hpp>
#include "../../tools/iris_vendor_bridge.hpp"
#include <hip/hip_runtime.h>
#define HIPBLAS_V2
#include <hipblas/hipblas.h>
#include <thrust/inner_product.h>
#include <thrust/system/hip/execution_policy.h>
#include <chrono>
#include <vector>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <limits>
static size_t shrRoundUp(int group,size_t size){if(size==0)return group;int r=size%group;return r?size+group-r:size;}
static void check(int s){if(s){fprintf(stderr,"DP IRIS/HIP failure %d\n",s);std::abort();}}
template<class T> static int blas(void* stream,unsigned,void*const*args){
 size_t n=hecbench_vendor_arg<size_t>(args,1);int repeat=hecbench_vendor_arg<int>(args,2);
 auto a=hecbench_vendor_arg<const T*>(args,3),b=hecbench_vendor_arg<const T*>(args,4);auto out=hecbench_vendor_arg<T*>(args,5);
 auto ms=hecbench_vendor_arg<double*>(args,6);hipblasHandle_t h;
 if(hipblasCreate(&h)!=HIPBLAS_STATUS_SUCCESS)return 1;
 if(hipblasSetStream(h,static_cast<hipStream_t>(stream))!=HIPBLAS_STATUS_SUCCESS||hipblasSetPointerMode(h,HIPBLAS_POINTER_MODE_DEVICE)!=HIPBLAS_STATUS_SUCCESS)return 2;
 hipDataType type=sizeof(T)==8?HIP_R_64F:HIP_R_32F;
 for(int i=0;i<100;++i)if(hipblasDotEx(h,n,a,type,1,b,type,1,out,type,type)!=HIPBLAS_STATUS_SUCCESS)return 3;
 if(hipStreamSynchronize(static_cast<hipStream_t>(stream))!=hipSuccess)return 4;
 auto start=std::chrono::steady_clock::now();
 for(int i=0;i<repeat;++i)if(hipblasDotEx(h,n,a,type,1,b,type,1,out,type,type)!=HIPBLAS_STATUS_SUCCESS)return 5;
 if(hipStreamSynchronize(static_cast<hipStream_t>(stream))!=hipSuccess)return 6;
 *ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeat;
 return hipblasDestroy(h)==HIPBLAS_STATUS_SUCCESS?0:7;
}
template<class T> static int stdpar_backend(void* stream,unsigned,void*const*args){
 size_t n=hecbench_vendor_arg<size_t>(args,1);int repeat=hecbench_vendor_arg<int>(args,2);
 auto a=hecbench_vendor_arg<const T*>(args,3),b=hecbench_vendor_arg<const T*>(args,4);
 // Like std::transform_reduce, Thrust's actual HIP backend returns a host scalar.
 auto out=hecbench_vendor_arg<T*>(args,5);auto ms=hecbench_vendor_arg<double*>(args,6);
 auto policy=thrust::hip::par.on(static_cast<hipStream_t>(stream));
 for(int i=0;i<100;++i)*out=thrust::inner_product(policy,a,a+n,b,T(0));
 auto start=std::chrono::steady_clock::now();
 for(int i=0;i<repeat;++i)*out=thrust::inner_product(policy,a,a+n,b,T(0));
 if(hipStreamSynchronize(static_cast<hipStream_t>(stream))!=hipSuccess)return 1;
 *ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeat;return 0;
}
template<class T> int dot(size_t n,int repeat){
 size_t size=shrRoundUp(1024,n),grid=shrRoundUp(1,size/(1024*4)),groups=size/4;
 printf("Global Work Size \t\t= %zu\nLocal Work Size \t\t= 1024\n",size);
 std::vector<T>a(size,0),b(size,0);T ref=0,out=0;int64_t exact=0;
 std::mt19937 engine(19937);std::uniform_int_distribution<int> dis(-32,32);
 for(size_t i=0;i<n;++i){a[i]=dis(engine);b[i]=dis(engine);ref+=a[i]*b[i];exact+=int64_t(a[i])*int64_t(b[i]);}
 iris::DMem da(a.data(),size*sizeof(T)),db(b.data(),size*sizeof(T)),dout(&out,sizeof(T));
 size_t g[]={grid*1024},l[]={1024};int errors=0;
 auto result=[&](const char* path){printf("VALUE %zu %.17g %.17g\n",sizeof(T),double(out),double(ref));bool ok=std::isfinite(double(out))&&out==ref;printf("%s\n\n",ok?"PASS":"FAIL");if(!ok)++errors;printf("Independent integer sum %s %.17g\n",path,double(exact));if(out!=T(exact))++errors;};
 auto kernel=[&](bool flush){
  out=0;check(dout.update(&out));void*p[]={&da,&db,&dout,&groups};int pi[]={iris_r,iris_r,iris_rw,sizeof(groups)};
  iris::Task task;check(task.kernel(sizeof(T)==8?"dot_double":"dot_float",1,nullptr,g,l,4,p,pi));if(flush)check(task.flush_out(dout));check(task.submit(0,nullptr,1));
 };
 for(int i=0;i<100;++i)kernel(false);
 auto start=std::chrono::steady_clock::now();for(int i=0;i<repeat;++i)kernel(i==repeat-1);
 printf("Average kernel execution time %f (ms)\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeat);result("custom");
 out=std::numeric_limits<T>::quiet_NaN();check(dout.update(&out));
 double ms=-1;double* timing=&ms;hecbench_vendor_callback callback=blas<T>;
 {void*p[]={&callback,&n,&repeat,&da,&db,&dout,&timing};int pi[]={sizeof(callback),sizeof(n),sizeof(repeat),iris_r,iris_r,iris_rw,sizeof(timing)};size_t one[]={1};iris::Task task;check(task.kernel("hecbench_vendor",1,nullptr,one,one,7,p,pi));check(task.flush_out(dout));check(task.submit(0,nullptr,1));}
 if(ms<0)++errors;
 printf("Average hipblasDotEx execution time %f (ms)\n",ms);result("DotEx");
 out=std::numeric_limits<T>::quiet_NaN();ms=-1;callback=stdpar_backend<T>;T* host_result=&out;
 {void*p[]={&callback,&n,&repeat,&da,&db,&host_result,&timing};int pi[]={sizeof(callback),sizeof(n),sizeof(repeat),iris_r,iris_r,sizeof(host_result),sizeof(timing)};size_t one[]={1};iris::Task task;check(task.kernel("hecbench_vendor",1,nullptr,one,one,7,p,pi));check(task.submit(0,nullptr,1));}
 if(ms<0)++errors;
 printf("Average std::transform_reduce execution time %f (ms)\n",ms);result("GPU stdpar backend");
 return errors;
}
int main(int argc,char**argv){
 if(argc!=3){printf("Usage: %s <number of elements> <repeat>\n",argv[0]);return 1;}
 size_t n=strtoull(argv[1],nullptr,10);int repeat=atoi(argv[2]);if(repeat<1||n>2147483647ULL)return 1;
 setenv("IRIS_KERNEL_HOST2HIP","./libhecbench_vendor.so",0);iris::Platform platform;check(platform.init(&argc,&argv,1));
 printf("------------- Data type is Float32 ---------------\n");int errors=dot<float>(n,repeat);
 printf("------------- Data type is Float64 ---------------\n");errors+=dot<double>(n,repeat);
 errors|=iris_error_count()!=0;check(platform.finalize());return errors?1:0;
}

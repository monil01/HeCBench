// Validation-only compatibility layer: compile the unchanged current CUDA
// translation unit with hipcc, not the incompatible legacy HIP data harness.
#pragma once
#include <hip/hip_runtime.h>
#include "../validation_dump.h"
inline void native_check(hipError_t rc){if(rc!=hipSuccess){std::fprintf(stderr,"Native HIP: %s\n",hipGetErrorString(rc));std::exit(1);}}
template<class T> inline void cudaMalloc(T**p,size_t bytes){native_check(hipMalloc(p,bytes));}
inline void cudaFree(void*p){native_check(hipFree(p));}
inline void cudaDeviceSynchronize(){
 static bool shown=false;
 if(!shown){int device;hipDeviceProp_t props;native_check(hipGetDevice(&device));native_check(hipGetDeviceProperties(&props,device));std::printf("Native HIP GPU=%s arch=%s\n",props.name,props.gcnArchName);shown=true;}
 native_check(hipDeviceSynchronize());
}
constexpr auto cudaMemcpyHostToDevice=hipMemcpyHostToDevice;
constexpr auto cudaMemcpyDeviceToHost=hipMemcpyDeviceToHost;
inline void cudaMemcpy(void*dest,const void*src,size_t bytes,hipMemcpyKind kind){
 static int batch=0,field=0;
 native_check(hipMemcpy(dest,src,bytes,kind));
 if(kind==hipMemcpyHostToDevice)projection_dump(batch,"input",src,bytes);
 if(kind==hipMemcpyDeviceToHost){const char*fields[]={"x","y","z","valid"};projection_dump(batch,fields[field],dest,bytes);if(++field==4){field=0;++batch;}}
}

// Only the original standard-library/container primitives are exposed here.
// No DC3 controller or custom source kernels are present in this library.
#include "thrust_bridge.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/version.h>
#include <cub/version.cuh>
#include <string>
#include <stdexcept>
#include <cstdio>
using SADeviceVector=thrust::device_vector<int32_t>;
static thread_local std::string failure;
static SADeviceVector&vec(void*h){if(!h)throw std::runtime_error("nullvectorhandle");return *static_cast<SADeviceVector*>(h);}
#define SA_GUARD(expression) try{expression;return 0;}catch(const std::exception&e){failure=e.what();fprintf(stderr,"SA_THRUST_BRIDGE:%s\n",e.what());return 1;}
extern "C" void*sa_vec_new(size_t n){try{return new SADeviceVector(n,0);}catch(const std::exception&e){failure=e.what();fprintf(stderr,"SA_THRUST_BRIDGE_ALLOC:%s\n",e.what());return nullptr;}}
extern "C" void sa_vec_delete(void*h){delete static_cast<SADeviceVector*>(h);}
extern "C" int32_t*sa_vec_data(void*h){return thrust::raw_pointer_cast(vec(h).data());}
extern "C" size_t sa_vec_size(void*h){return vec(h).size();}
extern "C" int sa_vec_assign_host(void*h,const int32_t*s,size_t n){SA_GUARD(vec(h).assign(s,s+n))}
extern "C" int sa_vec_copy_host(void*h,int32_t*d,size_t n){SA_GUARD(if(n>vec(h).size())throw std::runtime_error("copyrange");thrust::copy(vec(h).begin(),vec(h).begin()+n,d))}
extern "C" int sa_vec_assign_vec(void*d,void*s){SA_GUARD(vec(d)=vec(s))}
extern "C" int sa_vec_get(void*h,size_t i,int32_t*out){SA_GUARD(if(i>=vec(h).size())throw std::runtime_error("getrange");*out=vec(h)[i])}
extern "C" int sa_sort_by_key(void*k,void*v,size_t n){SA_GUARD(if(n>vec(k).size()||n>vec(v).size())throw std::runtime_error("sortrange");thrust::sort_by_key(vec(k).begin(),vec(k).begin()+n,vec(v).begin()))}
extern "C" int sa_exclusive_scan(void*h,size_t n){SA_GUARD(if(n>vec(h).size())throw std::runtime_error("scanrange");thrust::exclusive_scan(vec(h).begin(),vec(h).begin()+n,vec(h).begin()))}
extern "C" int sa_runtime_version(){int n=0;cudaError_t e=cudaRuntimeGetVersion(&n);return e==cudaSuccess?n:-int(e);}
extern "C" int sa_thrust_version(){return THRUST_VERSION;}
extern "C" int sa_cub_version(){return CUB_VERSION;}
extern "C" const char*sa_last_error(){return failure.c_str();}

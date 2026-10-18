#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/version.h>
#include <exception>
extern "C" int remap_sequence(int*a,int*b,int n,void*s){try{auto p=thrust::cuda::par.on((cudaStream_t)s);thrust::sequence(p,a,a+n);thrust::sequence(p,b,b+n);return (int)cudaDeviceSynchronize();}catch(...){return -1;}}
extern "C" int remap_sort(int*k,int*o,int n,void*s){try{thrust::sort_by_key(thrust::cuda::par.on((cudaStream_t)s),k,k+n,o);return (int)cudaDeviceSynchronize();}catch(...){return -2;}}
extern "C" int remap_unique(int*k,int*o,int n,int*count,void*s){try{auto end=thrust::unique_by_key(thrust::cuda::par.on((cudaStream_t)s),k,k+n,o);*count=end.first-k;return (int)cudaDeviceSynchronize();}catch(...){return -3;}}
extern "C" int remap_thrust_version(){return THRUST_VERSION;}
extern "C" int remap_runtime_version(){int v=0;cudaRuntimeGetVersion(&v);return v;}

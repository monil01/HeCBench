#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <type_traits>
#define C10_LAUNCH_BOUNDS_1(n) __launch_bounds__(n)
#define CUDA_1D_KERNEL_LOOP(i,n) for(decltype(n) i=(blockIdx.x*blockDim.x)+threadIdx.x;i<(n);i+=(blockDim.x*gridDim.x))
namespace at {
template<typename T>struct RestrictPtrTraits {using PtrType=T*;};
template<typename T,int D,template<typename> class Traits,typename I>struct GenericPackedTensorAccessor;
template<typename T,template<typename>class Traits,typename I>struct GenericPackedTensorAccessor<T,3,Traits,I>{
 T* data;I sizes[3],strides[3];
 __host__ __device__ I size(int d)const{return sizes[d];}
 struct Row{T*p;I stride;__device__ T&operator[](I k)const{return p[k*stride];}};
 struct Plane{T*p;I s1,s2;__device__ Row operator[](I i)const{return {p+i*s1,s2};}};
 __device__ Plane operator[](I b)const{return {data+b*strides[0],strides[1],strides[2]};}
};
}
template<typename T>__device__ __forceinline__ int m_signum(const T&x){return (T(0)<x)-(x<T(0));}
template<typename T>at::GenericPackedTensorAccessor<T,3,at::RestrictPtrTraits,int> acc(T*p,int a,int b,int c){return {p,{a,b,c},{b*c,c,1}};}

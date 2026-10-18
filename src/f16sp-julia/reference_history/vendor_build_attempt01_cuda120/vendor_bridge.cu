// Exact original vendor operators/APIs retained through a CUDA-specific C ABI.
#define main preserved_unexecuted_original_cli
#include "../f16sp-cuda/main.cu"
#undef main
#include <dlfcn.h>
extern "C" int study_create(void**p){cublasHandle_t h;auto s=cublasCreate(&h);if(s)return int(s);s=cublasSetPointerMode(h,CUBLAS_POINTER_MODE_DEVICE);*p=h;return int(s);}
extern "C" int study_destroy(void*p){return int(cublasDestroy((cublasHandle_t)p));}
extern "C" int study_cublas(void*h,const void*a,const void*b,void*r,long long n){return int(cublasDotEx_64((cublasHandle_t)h,n,a,CUDA_R_16F,1,b,CUDA_R_16F,1,r,CUDA_R_16F,CUDA_R_32F));}
extern "C" int study_cub(const void*a,const void*b,void*r,long long n){
 thrust::counting_iterator<size_t> counting(0);auto iter=thrust::make_transform_iterator(counting,Half2Mul((const half2*)a,(const half2*)b));
 void*temp=nullptr;size_t bytes=0;auto s=cub::DeviceReduce::Sum(temp,bytes,iter,(float*)r,size_t(n));if(s)return int(s);if(bytes){s=cudaMalloc(&temp,bytes);if(s)return int(s);}s=cub::DeviceReduce::Sum(temp,bytes,iter,(float*)r,size_t(n));if(bytes){auto e=cudaFree(temp);if(s==cudaSuccess)s=e;}return int(s);
}
extern "C" const char*study_cublas_path(){Dl_info i;if(dladdr((void*)&cublasDotEx_64,&i))return i.dli_fname;return "unknown";}
extern "C" int study_cublas_version(void*h){int v=-1;cublasGetVersion((cublasHandle_t)h,&v);return v;}

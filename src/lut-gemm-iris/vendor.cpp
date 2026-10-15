#include "../../tools/iris_vendor_bridge.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define HIPBLAS_V2
#include <hipblas/hipblas.h>
static int lut_reference(void* stream,unsigned,void* const* args) {
 auto input=hecbench_vendor_arg<__half*>(args,1);
 auto weight=hecbench_vendor_arg<__half*>(args,2);
 auto output=hecbench_vendor_arg<__half*>(args,3);
 int m=hecbench_vendor_arg<int>(args,4),n=hecbench_vendor_arg<int>(args,5),k=hecbench_vendor_arg<int>(args,6);
 hipblasHandle_t h;
 if(hipblasCreate(&h)!=HIPBLAS_STATUS_SUCCESS)return 1;
 if(hipblasSetStream(h,static_cast<hipStream_t>(stream))!=HIPBLAS_STATUS_SUCCESS)return 2;
 __half alpha=1,beta=0;
 auto status=hipblasGemmEx(h,HIPBLAS_OP_N,HIPBLAS_OP_N,n,m,k,&alpha,weight,HIP_R_16F,n,input,HIP_R_16F,k,&beta,output,HIP_R_16F,n,HIPBLAS_COMPUTE_16F,HIPBLAS_GEMM_DEFAULT);
 auto sync=hipStreamSynchronize(static_cast<hipStream_t>(stream));auto destroy=hipblasDestroy(h);
 return status==HIPBLAS_STATUS_SUCCESS&&sync==hipSuccess&&destroy==HIPBLAS_STATUS_SUCCESS?0:3;
}

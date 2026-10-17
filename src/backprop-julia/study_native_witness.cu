#include <cuda_runtime.h>
#include <cstdio>
#include "../backprop-cuda/backprop.h"
#include "../backprop-cuda/bpnn_layerforward.h"
int main(){float h[17],w[289],got[16];for(float&x:h)x=1.0f/16;for(float&x:w)x=1.0f/16;float *di,*dw,*do_;cudaMalloc(&di,sizeof(h));cudaMalloc(&dw,sizeof(w));cudaMalloc(&do_,sizeof(got));cudaMemcpy(di,h,sizeof(h),cudaMemcpyHostToDevice);cudaMemcpy(dw,w,sizeof(w),cudaMemcpyHostToDevice);kernel_layerforward<<<dim3(1,1),dim3(16,16)>>>(di,dw,do_,16);cudaDeviceSynchronize();cudaMemcpy(got,do_,sizeof(got),cudaMemcpyDeviceToHost);bool doubled=true;for(float v:got)doubled&=v==.125f;printf("{\"native_partial0\":%.17g,\"standard_dot_expected\":0.0625,\"all16_are_exact_double\":%s}\n",got[0],doubled?"true":"false");cudaFree(di);cudaFree(dw);cudaFree(do_);return doubled?0:2;}

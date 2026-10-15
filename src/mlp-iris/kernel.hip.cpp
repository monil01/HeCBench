#include <hip/hip_runtime.h>
extern "C" __global__ void mlp_layer(const float* in,const float* w,const float* bias,float* out,int batch,int ni,int no,int use_bias,int use_relu) {
 size_t i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=size_t(batch)*no)return;
 int b=i/no,o=i%no;float sum=0;
 for(int k=0;k<ni;++k)sum+=w[size_t(o)*ni+k]*in[size_t(b)*ni+k];
 if(use_bias==1)sum+=bias[o];
 if(use_relu==1)sum=fmaxf(sum,0.f);
 out[i]=sum;
}

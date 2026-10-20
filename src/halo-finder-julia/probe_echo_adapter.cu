#include <cuda_runtime.h>
#include <cstdio>
#include "halo_dispatch.h"
__global__ void echo_actual_inputs(HaloArgs a,uint32_t*out){
 int k=blockIdx.x*blockDim.x+threadIdx.x,total=7*a.count+4*a.count1;
 if(k<total){const float*p=nullptr;int i;if(k<7*a.count){int field=k/a.count;i=k%a.count;
  if(field==0)p=a.x;else if(field==1)p=a.y;else if(field==2)p=a.z;else if(field==3)p=a.mass;else if(field==4)p=a.vx;else if(field==5)p=a.vy;else p=a.vz;
 }else{int q=k-7*a.count,field=q/a.count1;i=q%a.count1;if(field==0)p=a.x1;else if(field==1)p=a.y1;else if(field==2)p=a.z1;else p=a.mass1;}out[k]=__float_as_uint(p[i]);}
}
int main(){float*p;cudaMallocHost(&p,11*4);for(int i=0;i<11;i++)p[i]=i;
HaloArgs a{1,1,1,1,32,4,p,p+1,p+2,p+3,p+4,p+5,p+6,p+7,p+8,p+9,p+10,9,.01f,1,nullptr};uint32_t*out;cudaMalloc(&out,11*4);
echo_actual_inputs<<<1,256>>>(a,out);auto e=cudaDeviceSynchronize();printf("sizeofArgs%zu CUDA%s\n",sizeof(a),cudaGetErrorString(e));cudaFree(out);cudaFreeHost(p);return e==cudaSuccess?0:1;}

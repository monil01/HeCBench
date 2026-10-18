#include <cuda_runtime.h>
#include <cstdio>
#include <climits>
__global__ void cast_witness(unsigned*out,float*fp){volatile float value=float(UINT_MAX);*fp=value;out[0]=value;out[1]=float(4096);out[2]=double(UINT_MAX);}
int main(){unsigned*d,out[3];float*f,fp;cudaMalloc(&d,12);cudaMalloc(&f,4);cast_witness<<<1,1>>>(d,f);auto e=cudaDeviceSynchronize();if(e!=cudaSuccess){puts(cudaGetErrorString(e));return 2;}cudaMemcpy(out,d,12,cudaMemcpyDefault);cudaMemcpy(&fp,f,4,cudaMemcpyDefault);printf("SOURCE_INTERNAL_PADDING_FLOAT32 %.17g CAST_U32 %u VALID4096 %u DOUBLE_UINTMAX %u\n",double(fp),out[0],out[1],out[2]);return out[0]==UINT_MAX&&out[1]==4096&&out[2]==UINT_MAX?0:3;}

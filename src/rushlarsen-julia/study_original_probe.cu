#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include "../rushlarsen-cuda/utils.h"
#include "../rushlarsen-cuda/kernels.cu"
int main(){const int n=1;std::vector<double>s(19),p(53),cpu;init_state_values(s.data(),n);init_parameters_values(p.data(),n);cpu=s;forward_rush_larsen(cpu.data(),0.,2e-5,p.data(),n);double *ds,*dp;cudaMalloc(&ds,s.size()*8);cudaMalloc(&dp,p.size()*8);cudaMemcpy(ds,s.data(),s.size()*8,cudaMemcpyHostToDevice);cudaMemcpy(dp,p.data(),p.size()*8,cudaMemcpyHostToDevice);k_forward_rush_larsen<<<1,256>>>(ds,0.,2e-5,dp,n);auto e=cudaDeviceSynchronize();if(e!=cudaSuccess){printf("CUDA_ERROR %s\n",cudaGetErrorString(e));return 2;}cudaMemcpy(s.data(),ds,s.size()*8,cudaMemcpyDeviceToHost);for(int f=0;f<19;f++)printf("STATE %d CPU=%.17g GPU=%.17g CPU_FINITE=%d GPU_FINITE=%d\n",f,cpu[f],s[f],std::isfinite(cpu[f]),std::isfinite(s[f]));cudaFree(ds);cudaFree(dp);return 0;}

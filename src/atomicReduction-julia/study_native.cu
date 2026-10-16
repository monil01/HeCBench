#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include "atomicReduction-cuda/kernels.h"
void launch_reduce(int width,int*in,int*out,int n,int blocks,int bs){
 if(width==1)atomic_reduction<<<blocks,bs>>>(in,out,n);
 if(width==2)atomic_reduction_v2<<<blocks,bs>>>(in,out,n);
 if(width==4)atomic_reduction_v4<<<blocks,bs>>>(in,out,n);
 if(width==8)atomic_reduction_v8<<<blocks,bs>>>(in,out,n);
 if(width==16)atomic_reduction_v16<<<blocks,bs>>>(in,out,n);
}
int main(int argc,char**argv){
 int n=argc>1?atoi(argv[1]):65536,repeat=argc>2?atoi(argv[2]):10;if(n%16)return 2;
 std::vector<int>input(n);int expected=0;for(int i=0;i<n;i++){input[i]=i%2;expected+=input[i];}
 int *dx,*dy;cudaMalloc(&dx,n*sizeof(int));cudaMalloc(&dy,sizeof(int));cudaMemcpy(dx,input.data(),n*sizeof(int),cudaMemcpyHostToDevice);
 for(int bs=128;bs<=1024;bs*=2)for(int width=1;width<=16;width*=2){int blocks=((n+bs-1)/bs<2048?(n+bs-1)/bs:2048)/width;if(blocks<1)blocks=1;cudaMemset(dy,0,sizeof(int));launch_reduce(width,dx,dy,n,blocks,bs);}cudaDeviceSynchronize();
 for(int trial=1;trial<=5;trial++){
  std::vector<double>times;std::vector<int>values;bool correct=true;double total=0;
  for(int bs=128;bs<=1024;bs*=2)for(int width=1;width<=16;width*=2){
   int blocks=((n+bs-1)/bs<2048?(n+bs-1)/bs:2048)/width;if(blocks<1)blocks=1;
   cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();
   for(int r=0;r<repeat;r++){cudaMemset(dy,0,sizeof(int));launch_reduce(width,dx,dy,n,blocks,bs);}
   cudaDeviceSynchronize();auto stop=std::chrono::steady_clock::now();double time=std::chrono::duration<double>(stop-start).count()/repeat;times.push_back(time);total+=time;
   int value;cudaMemcpy(&value,dy,sizeof(int),cudaMemcpyDeviceToHost);values.push_back(value);correct &= value==expected;
  }
  char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);FILE*f=fopen(filename,"wb");fwrite(values.data(),sizeof(int),20,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":20,\"max_abs_error\":0.0,\"timing_region\":\"sum_twenty_reset_plus_kernel_loop_means\",\"jit_excluded\":true,\"internal_repeats\":%d,\"subkernel_times_seconds\":[",trial,total,correct?"true":"false",repeat);
  for(int i=0;i<20;i++)printf("%s%.12g",i?",":"",times[i]);
  printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":20,\"byte_order\":\"little\"}}\n",filename);if(!correct)return 1;
 }
 cudaFree(dx);cudaFree(dy);return 0;
}

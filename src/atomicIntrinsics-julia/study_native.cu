#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cstdint>
#include <algorithm>
#include "atomicIntrinsics-cuda/kernel.h"
template<class T>std::vector<T> initialize_intrinsics(){return {T(0),T(0),T(-256),T(256),T(255),T(0),T(255),T(0),T(0)};}
template<class T>bool run_intrinsics(void*raw,int n,int repeat,double &time,std::vector<uint32_t>&values){
 T*data=static_cast<T*>(raw);std::vector<T>initial=initialize_intrinsics<T>(),result(9),expected=initial;
 cudaMemcpy(data,initial.data(),9*sizeof(T),cudaMemcpyHostToDevice);cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();
 for(int r=0;r<repeat;r++)testKernel<<<(n+255)/256,256>>>(data,n);
 cudaDeviceSynchronize();auto stop=std::chrono::steady_clock::now();time=std::chrono::duration<double>(stop-start).count()/repeat;
 cudaMemcpy(result.data(),data,9*sizeof(T),cudaMemcpyDeviceToHost);
 for(int r=0;r<repeat;r++)for(int i=0;i<n;i++){
  expected[0]=T(uint32_t(expected[0])+10u);expected[1]=T(uint32_t(expected[1])-10u);expected[2]=std::max(expected[2],T(i));expected[3]=std::min(expected[3],T(i));expected[4]&=T(2*i+7);expected[5]|=T(uint32_t(1)<<(i&31));expected[6]^=T(i);
  uint32_t inc=uint32_t(expected[7]);expected[7]=T(inc>=17?0:inc+1);uint32_t dec=uint32_t(expected[8]);expected[8]=T((dec==0||dec>137)?137:dec-1);
 }
 bool correct=true;for(int i=0;i<9;i++){correct &= result[i]==expected[i];values.push_back(uint32_t(result[i]));}return correct;
}
int main(int argc,char**argv){
 int n=argc>1?atoi(argv[1]):4096,repeat=argc>2?atoi(argv[2]):10;void*a,*b;cudaMalloc(&a,9*4);cudaMalloc(&b,9*4);double times[2];std::vector<uint32_t>warm;
 run_intrinsics<int>(a,n,1,times[0],warm);run_intrinsics<unsigned>(b,n,1,times[1],warm);
 for(int trial=1;trial<=5;trial++){
  std::vector<uint32_t>values;bool correct=run_intrinsics<int>(a,n,repeat,times[0],values);correct &= run_intrinsics<unsigned>(b,n,repeat,times[1],values);
  char filename[64];snprintf(filename,sizeof(filename),"trial_%d.bin",trial);FILE*f=fopen(filename,"wb");fwrite(values.data(),sizeof(uint32_t),18,f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":18,\"max_abs_error\":0.0,\"timing_region\":\"sum_two_atomic_intrinsic_loop_means\",\"jit_excluded\":true,\"internal_repeats\":%d,\"subkernel_times_seconds\":[%.12g,%.12g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":18,\"byte_order\":\"little\"}}\n",trial,times[0]+times[1],correct?"true":"false",repeat,times[0],times[1],filename);if(!correct)return 1;
 }cudaFree(a);cudaFree(b);return 0;
}

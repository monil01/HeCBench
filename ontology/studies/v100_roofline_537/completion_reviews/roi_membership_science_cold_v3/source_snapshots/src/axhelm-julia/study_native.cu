#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
using dfloat=float;
#define p_Nggeo 7
#define p_G00ID 1
#define p_G01ID 2
#define p_G02ID 3
#define p_G11ID 4
#define p_G12ID 5
#define p_G22ID 6
#define p_GWJID 0
#include "../axhelm-cuda/axhelmKernel.cpp"
#include "../axhelm-cuda/axhelmReference.cpp"
#include "study_fixture.h"
template<size_t N>std::vector<float> decode(const uint32_t(&a)[N]){std::vector<float>out(N);memcpy(out.data(),a,4*N);return out;}
const int E=7,offset=E*512;
struct Result{std::vector<float>out;double elapsed,error;bool ok;};
Result onecase(int fields){auto G=decode(STUDY_G_BITS),Q=decode(STUDY_Q_BITS),D=decode(STUDY_D_BITS),L=decode(STUDY_LAMBDA_BITS),expected=decode(STUDY_EXPECTED_BITS);std::vector<float>out(fields*offset),cpu(fields*offset);for(int f=0;f<fields;f++)axhelmReference(8,E,1.125f,G.data(),D.data(),Q.data()+f*offset,cpu.data()+f*offset);
 cudaDeviceSynchronize();auto t=std::chrono::steady_clock::now();float *g,*q,*a,*d,*l;cudaMalloc(&g,4*G.size());cudaMalloc(&q,4*fields*offset);cudaMalloc(&a,4*fields*offset);cudaMalloc(&d,4*D.size());cudaMalloc(&l,4*L.size());cudaMemcpy(g,G.data(),4*G.size(),cudaMemcpyHostToDevice);cudaMemcpy(q,Q.data(),4*fields*offset,cudaMemcpyHostToDevice);cudaMemcpy(d,D.data(),4*D.size(),cudaMemcpyHostToDevice);cudaMemcpy(l,L.data(),4*L.size(),cudaMemcpyHostToDevice);for(int rep=0;rep<100;rep++){if(fields==1)axhelm<<<E,dim3(8,8)>>>(E,offset,g,d,l,q,a);else axhelm_n3<<<E,dim3(8,8)>>>(E,offset,g,d,l,q,a);}cudaMemcpy(out.data(),a,4*fields*offset,cudaMemcpyDeviceToHost);cudaFree(g);cudaFree(q);cudaFree(a);cudaFree(d);cudaFree(l);cudaDeviceSynchronize();double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/100;double err=0;bool ok=true;for(size_t i=0;i<out.size();i++){err=std::max(err,std::abs(double(out[i])-expected[i]));if(std::abs(double(out[i])-cpu[i])>=1e-3)ok=false;}ok&=err<1e-3;return{out,elapsed,err,ok};}
int main(){onecase(1);onecase(3);for(int trial=1;trial<=5;trial++){auto a=onecase(1),b=onecase(3);std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/trial_"+std::to_string(trial)+".bin";FILE*f=fopen(path.c_str(),"wb");fwrite(a.out.data(),4,a.out.size(),f);fwrite(b.out.data(),4,b.out.size(),f);fclose(f);bool ok=a.ok&&b.ok;printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":14336,\"max_abs_error\":%.17g,\"timing_region\":\"sum_scalar_fused3_allocate_h2d_100kernels_d2h_free_perrepeat\",\"jit_excluded\":true,\"internal_repeats\":100,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":14336,\"byte_order\":\"little\"}}\n",trial,a.elapsed+b.elapsed,ok?"true":"false",std::max(a.error,b.error),path.c_str());if(!ok)return 8;}}

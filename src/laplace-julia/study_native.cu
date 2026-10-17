#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <string>
#include <algorithm>
#include "study_kernels.h"
#include "../laplace-cuda/reference.h"
#include "corrected_coeffs.h"
#define CK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
int main(){const int N=NUM*NUM,S=(NUM/2+2)*(NUM+2);std::vector<float>h[6];float*a[6];for(int j=0;j<6;j++){h[j].resize(N);CK(cudaMalloc(&a[j],N*4));}
 fill_coeffs_corrected(NUM,NUM,1.f,1.f/NUM,1.f/NUM,.01f,1.f,h[0].data(),h[1].data(),h[2].data(),h[3].data(),h[4].data(),h[5].data());
 std::vector<float>prefix;for(int j=0;j<6;j++){CK(cudaMemcpy(a[j],h[j].data(),N*4,cudaMemcpyHostToDevice));prefix.insert(prefix.end(),h[j].begin(),h[j].end());}
 std::vector<float>er(S),eb(S);int ci=0;float cn=0;for(ci=1;ci<=1000000;ci++){float nr=red_ref(h[0].data(),h[1].data(),h[2].data(),h[3].data(),h[4].data(),h[5].data(),eb.data(),er.data());float nb=black_ref(h[0].data(),h[1].data(),h[2].data(),h[3].data(),h[4].data(),h[5].data(),er.data(),eb.data());cn=sqrtf((nr+nb)/N);if(cn<1e-6f)break;}printf("LAPLACE_CPU iterations=%d change_rms=%.12g\n",ci,cn);if(ci>1000000)return 3;
 float*r,*b,*norm,*total;CK(cudaMalloc(&r,S*4));CK(cudaMalloc(&b,S*4));CK(cudaMalloc(&norm,S*4));CK(cudaMalloc(&total,4));CK(cudaMemset(r,0,S*4));CK(cudaMemset(b,0,S*4));CK(cudaMemset(norm,0,S*4));dim3 threads(128,2),blocks(1,128);
 red_kernel<<<blocks,threads>>>(a[0],a[1],a[2],a[3],a[4],a[5],b,r,norm);black_kernel<<<blocks,threads>>>(a[0],a[1],a[2],a[3],a[4],a[5],r,b,norm);size_t warmbytes=0;void*warm=nullptr;CK(cub::DeviceReduce::Sum(warm,warmbytes,norm,total,S));CK(cudaMalloc(&warm,warmbytes));CK(cub::DeviceReduce::Sum(warm,warmbytes,norm,total,S));CK(cudaDeviceSynchronize());CK(cudaFree(warm));bool all=true;
 for(int trial=1;trial<=5;trial++){CK(cudaMemset(r,0,S*4));CK(cudaMemset(b,0,S*4));CK(cudaMemset(norm,0,S*4));CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();void*scratch=nullptr;size_t bytes=0;CK(cub::DeviceReduce::Sum(scratch,bytes,norm,total,S));CK(cudaMalloc(&scratch,bytes));int it;float rms=0;
  for(it=1;it<=1000000;it++){float nr,nb;red_kernel<<<blocks,threads>>>(a[0],a[1],a[2],a[3],a[4],a[5],b,r,norm);CK(cub::DeviceReduce::Sum(scratch,bytes,norm,total,S));CK(cudaMemcpy(&nr,total,4,cudaMemcpyDeviceToHost));black_kernel<<<blocks,threads>>>(a[0],a[1],a[2],a[3],a[4],a[5],r,b,norm);CK(cub::DeviceReduce::Sum(scratch,bytes,norm,total,S));CK(cudaMemcpy(&nb,total,4,cudaMemcpyDeviceToHost));rms=sqrtf((nr+nb)/N);if(rms<1e-6f)break;}
  CK(cudaFree(scratch));CK(cudaDeviceSynchronize());double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::vector<float>vr(S),vb(S);CK(cudaMemcpy(vr.data(),r,S*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(vb.data(),b,S*4,cudaMemcpyDeviceToHost));double err=0;bool ok=it<=1000000;
  for(int k=0;k<S;k++){err=std::max(err,double(fabsf(vr[k]-er[k])));err=std::max(err,double(fabsf(vb[k]-eb[k])));ok=ok&&std::isfinite(vr[k])&&std::isfinite(vb[k])&&fabsf(vr[k]-er[k])<=1e-4f&&fabsf(vb[k]-eb[k])<=1e-4f&&vr[k]>=0&&vb[k]>=0&&vr[k]<=1.00001f&&vb[k]<=1.00001f;}
  all&=ok;auto out=prefix;out.insert(out.end(),vr.begin(),vr.end());out.insert(out.end(),vb.begin(),vb.end());std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(trial)+".bin";FILE*f=fopen(path.c_str(),"wb");if(!f)return 4;fwrite(out.data(),4,out.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":1,\"iterations\":%d,\"iterate_change_rms\":%.12g,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,sec,ok?"true":"false",2*S,err,it,rms,path.c_str(),out.size());
 }puts(all?"PASS":"FAIL");return all?0:1;
}

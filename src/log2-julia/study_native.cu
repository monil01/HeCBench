#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <chrono>
#ifdef STUDY_FIX_PRECISION
#include "study_corrected_kernel.h"
#else
#include "../log2-cuda/kernel.h"
#endif
int main(){
 FILE*f=fopen("study_input_f32.bin","rb");if(!f)return 2;fseek(f,0,SEEK_END);int n=ftell(f)/4;rewind(f);std::vector<float>host(n);if(fread(host.data(),4,n,f)!=(size_t)n)return 3;fclose(f);
 f=fopen("oracle_f64.bin","rb");if(!f)return 4;std::vector<double>oracle(n);if(fread(oracle.data(),8,n,f)!=(size_t)n)return 5;fclose(f);
 std::vector<int>ps={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,23};int np=ps.size(),repeat=17;const float guard=-12345.5f;std::vector<float>got(np*n+2),inputback(n),artifact(n+np+got.size());float*in,*out;
 if(cudaMalloc(&in,n*4)!=cudaSuccess||cudaMalloc(&out,got.size()*4)!=cudaSuccess)return 6;cudaMemcpy(in,host.data(),n*4,cudaMemcpyHostToDevice);
 auto suite=[&](){for(int p=0;p<np;p++)for(int r=0;r<repeat;r++)compute_log<<<(n+255)/256,256>>>(out+1,in,p,n,ps[p]);return cudaGetLastError()==cudaSuccess&&cudaDeviceSynchronize()==cudaSuccess;};
 if(!suite())return 7;bool allok=true;
 for(int trial=1;trial<=5;trial++){
  std::fill(got.begin(),got.end(),guard);cudaMemcpy(out,got.data(),got.size()*4,cudaMemcpyHostToDevice);cudaDeviceSynchronize();auto t=std::chrono::steady_clock::now();bool ok=suite();double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/repeat;
  ok &=cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost)==cudaSuccess;ok &=cudaMemcpy(inputback.data(),in,n*4,cudaMemcpyDeviceToHost)==cudaSuccess;ok &=inputback==host&&got.front()==guard&&got.back()==guard;
  size_t bad=0;double maxerr=0;for(int p=0;p<np;p++)for(int i=0;i<n;i++){float nearest=(float)oracle[i];double ulp=std::nextafter(nearest,INFINITY)-nearest;double bound=std::ldexp(1.,-ps[p])+16*std::ldexp(1.,-24)+2*ulp;double err=std::abs((double)got[1+p*n+i]-oracle[i]);if(!std::isfinite(got[1+p*n+i])||err>bound)bad++;maxerr=std::max(maxerr,err);}ok &=bad==0;allok &=ok;
  std::copy(inputback.begin(),inputback.end(),artifact.begin());for(int p=0;p<np;p++)artifact[n+p]=(float)ps[p];std::copy(got.begin(),got.end(),artifact.begin()+n+np);
  char path[80];snprintf(path,sizeof(path),"output_trial%d.bin",trial);f=fopen(path,"wb");if(!f)return 8;fwrite(artifact.data(),4,artifact.size(),f);fclose(f);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"independent_accuracy_failures\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"timing_region\":\"19precision_log_approx_kernels_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",artifact.size(),bad,maxerr,repeat,path,artifact.size());
 }cudaFree(in);cudaFree(out);puts(allok?"PASS":"FAIL");return allok?0:1;
}

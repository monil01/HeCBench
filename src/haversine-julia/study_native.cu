#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "../haversine-cuda/distance.cu"
int main(int argc,char**argv){bool stress=argc>1&&atoi(argv[1]);const char*name=stress?"input_stress_f64.bin":"input_normal_f64.bin";const char*oname=stress?"oracle_stress_f64.bin":"oracle_normal_f64.bin";FILE*f=fopen(name,"rb");if(!f)return 2;fseek(f,0,SEEK_END);int n=ftell(f)/32;rewind(f);std::vector<double4>host(n+2),back(n+2);double guard=-1234567.25;for(auto&v:host)v=make_double4(guard,guard,guard,guard);if(fread(host.data()+1,32,n,f)!=(size_t)n)return 3;fclose(f);f=fopen(oname,"rb");if(!f)return 4;std::vector<double>expected(n);fread(expected.data(),8,n,f);fclose(f);double atol=stress?1e-3:1e-6;
 double4*in;double*out;cudaMalloc(&in,(n+2)*32);cudaMalloc(&out,(n+2)*8);cudaMemcpy(in,host.data(),(n+2)*32,cudaMemcpyHostToDevice);std::vector<double>got(n+2),artifact(5*(n+2));int repeats=100;
 auto suite=[&](){for(int r=0;r<repeats;r++)compute_haversine_distance<<<(n+255)/256,256>>>(in+1,out+1,n);return cudaGetLastError()==cudaSuccess&&cudaDeviceSynchronize()==cudaSuccess;};if(!suite())return 5;bool allok=true;
 for(int trial=1;trial<=5;trial++){
  std::fill(got.begin(),got.end(),guard);cudaMemcpy(out,got.data(),got.size()*8,cudaMemcpyHostToDevice);cudaDeviceSynchronize();auto t=std::chrono::steady_clock::now();bool ok=suite();double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/repeats;
  ok &=cudaMemcpy(back.data(),in,back.size()*32,cudaMemcpyDeviceToHost)==cudaSuccess;ok &=memcmp(back.data(),host.data(),back.size()*32)==0;ok &=cudaMemcpy(got.data(),out,got.size()*8,cudaMemcpyDeviceToHost)==cudaSuccess;ok &=got.front()==guard&&got.back()==guard;size_t bad=0,nonfinite=0;double maxerr=0;
  for(int i=0;i<n;i++){if(!std::isfinite(got[i+1])){bad++;nonfinite++;}else{double e=std::abs(got[i+1]-expected[i]);maxerr=std::max(maxerr,e);bad+=e>atol;}}ok &=bad==0;allok &=ok;memcpy(artifact.data(),back.data(),back.size()*32);memcpy(artifact.data()+4*(n+2),got.data(),got.size()*8);
  char path[80],metric[80];snprintf(path,sizeof(path),"output_trial%d.bin",trial);f=fopen(path,"wb");if(!f)return 6;fwrite(artifact.data(),8,artifact.size(),f);fclose(f);if(nonfinite)strcpy(metric,"null");else snprintf(metric,sizeof(metric),"%.12g",maxerr);
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"independent_accuracy_failures\":%zu,\"nonfinite_outputs\":%zu,\"max_abs_error\":%s,\"max_rel_error\":null,\"timing_region\":\"haversine_compute_launches_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,sec,ok?"true":"false",artifact.size(),bad,nonfinite,metric,repeats,path,artifact.size());
 }cudaFree(in);cudaFree(out);puts(allok?"PASS":"FAIL");return allok?0:1;
}

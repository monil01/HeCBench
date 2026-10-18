#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include "../urng-cuda/urng.h"
#ifdef STUDY_ORIGINAL
#include "../urng-cuda/kernel.cu"
#else
#include "study_corrected_kernel.cuh"
#endif
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
std::vector<unsigned char>load(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)exit(3);size_t n=f.tellg();f.seekg(0);std::vector<unsigned char>v(n);f.read((char*)v.data(),n);return v;}
int main(int argc,char**argv){int repeats=argc>1?atoi(argv[1]):1000;auto input=load("fixture_rgba.bin"),expected=load("expected_rgba_guards.bin");const int ns[2]={16384,1031};std::vector<uchar4*>in,out;std::vector<unsigned char>prefix;size_t at=0;
for(int n:ns){uchar4*a,*b;CK(cudaMalloc(&a,n*4));CK(cudaMalloc(&b,(n+17)*4));CK(cudaMemcpy(a,input.data()+at,n*4,cudaMemcpyHostToDevice));CK(cudaMemset(b,165,(n+17)*4));std::vector<unsigned char>v(n*4);CK(cudaMemcpy(v.data(),a,n*4,cudaMemcpyDeviceToHost));prefix.insert(prefix.end(),v.begin(),v.end());in.push_back(a);out.push_back(b);at+=n*4;}
auto run=[&](int j){noise_uniform<<<(ns[j]+255)/256,256>>>(in[j],out[j],ns[j],FACTOR);CK(cudaGetLastError());};for(int j=0;j<2;j++)run(j);CK(cudaDeviceSynchronize());bool all=true;
for(int trial=1;trial<=5;trial++){std::vector<unsigned char>result;double times[2],means[2];size_t inp=0;
for(int j=0;j<2;j++){CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int rep=0;rep<repeats;rep++)run(j);CK(cudaDeviceSynchronize());times[j]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;std::vector<unsigned char>v((ns[j]+17)*4);CK(cudaMemcpy(v.data(),out[j],v.size(),cudaMemcpyDeviceToHost));double mean=0;for(int k=0;k<ns[j]*4;k++)mean+=int(v[k])-int(input[inp+k]);means[j]=mean/(ns[j]*4*25.);inp+=ns[j]*4;result.insert(result.end(),v.begin(),v.end());}
bool ok=result==expected;all &=ok;int errors=0,maxerr=0;for(size_t k=0;k<result.size();k++){int e=abs(int(result[k])-int(expected[k]));errors+=(e!=0);maxerr=std::max(maxerr,e);}auto data=prefix;data.insert(data.end(),result.begin(),result.end());const char*dir=getenv("STUDY_OUTPUT_DIR");char file[4096];snprintf(file,sizeof(file),"%s/urng_trial%d.bin",dir?dir:".",trial);std::ofstream f(file,std::ios::binary);f.write((char*)data.data(),data.size());f.close();
printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"component_seconds\":[%.12g,%.12g],\"correct\":%s,\"source_mean_check_pass\":%s,\"source_means\":[%.12g,%.12g],\"mismatched_values\":%d,\"checked_values\":%zu,\"max_abs_error\":%d,\"max_rel_error\":null,\"internal_repeats\":%d,\"jit_excluded\":true,\"timing_region\":\"sum_two_source_kernel_loopmeans_devicecompletion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint8\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,times[0]+times[1],times[0],times[1],ok?"true":"false",fabs(means[0])<1&&fabs(means[1])<1?"true":"false",means[0],means[1],errors,result.size(),maxerr,repeats,file,data.size());fflush(stdout);}
puts(all?"PASS":"FAIL");return all?0:1;}

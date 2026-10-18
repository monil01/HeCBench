#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstdio>
#include <vector>
#include <fstream>
#include <stdexcept>
#define A 1103515245
#define C 12345
#define M INT_MAX
#define PI 3.1415926535897932f
#define BLOCK_SIZE 256
#include "/home/imo/HeCBench_julia/src/particlefilter-cuda/kernel_normalize_weights.h"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
int main(){int N=400000;std::vector<float>w(N),cdf(N),u(N);for(int i=0;i<N;i++)w[i]=1.f+((i*7)%31)/32.f;float sum=0;for(auto x:w)sum+=x;std::vector<float>expected(N);float s=0;for(int i=0;i<N;i++){s+=w[i]/sum;expected[i]=s;}float *dw,*dc,*du,*ds;int*seed;CK(cudaMalloc(&dw,N*4));CK(cudaMalloc(&dc,N*4));CK(cudaMalloc(&du,N*4));CK(cudaMalloc(&ds,4));CK(cudaMalloc(&seed,N*4));std::vector<int>seeds(N);for(int i=0;i<N;i++)seeds[i]=i+1;for(int t=0;t<5;t++){std::vector<float>sentinel(N,-9.f);CK(cudaMemcpy(dw,w.data(),N*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(dc,sentinel.data(),N*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(du,sentinel.data(),N*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(ds,&sum,4,cudaMemcpyHostToDevice));CK(cudaMemcpy(seed,seeds.data(),N*4,cudaMemcpyHostToDevice));kernel_normalize_weights<<<(N+255)/256,256>>>(dw,ds,dc,du,seed,N);CK(cudaDeviceSynchronize());std::vector<float>wo(N);CK(cudaMemcpy(wo.data(),dw,N*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(cdf.data(),dc,N*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(u.data(),du,N*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(seeds.data(),seed,N*4,cudaMemcpyDeviceToHost));int badcdf=0,badu=0;double maxcdf=0;for(int i=0;i<N;i++){badcdf+=cdf[i]!=expected[i];maxcdf=fmax(maxcdf,fabs(cdf[i]-expected[i]));badu+=fabsf((u[i]-i/(float)N)-u[0])>1e-6f;}std::vector<double>v;for(auto z:wo)v.push_back(z);for(auto z:cdf)v.push_back(z);for(auto z:u)v.push_back(z);for(auto z:seeds)v.push_back(z);std::string path="/tmp/particlefilter_normalize_original_"+std::to_string(t)+".bin";std::ofstream(path,std::ios::binary).write((char*)v.data(),v.size()*8);printf("trial%d badcdf%d maxcdf%.12g badu%d actualu0%.12g path%s\n",t,badcdf,maxcdf,badu,u[0],path.c_str());}}

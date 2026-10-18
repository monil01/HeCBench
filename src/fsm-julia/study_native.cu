// Native FSM device bodies included unchanged. Named trace sizes only.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstring>
#include "../fsm-cuda/parameters.h"
#include "../fsm-cuda/kernels.h"
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
template<class T> std::vector<T> load(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)exit(3);size_t n=f.tellg();f.seekg(0);std::vector<T>v(n/sizeof(T));f.read((char*)v.data(),n);return v;}
template<class T> T* alloc(size_t n){T*p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
template<class T> void append(std::vector<uint64_t>&o,T*p,size_t n){std::vector<T>v(n);CK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));for(auto x:v)o.push_back(x);}
int main(){
 auto data=load<uint16_t>("fixture_u16.bin");auto expected=load<uint64_t>("expected_u64.bin");
 const int ns[3]={257,4099,1021};std::vector<uint16_t*> ds;
 size_t at=0;std::vector<uint64_t>prefix;
 for(int n:ns){auto d=alloc<uint16_t>(n);CK(cudaMemcpy(d,data.data()+at,n*2,cudaMemcpyHostToDevice));ds.push_back(d);append(prefix,d,n);at+=n;}
 auto best=alloc<int>(19);auto rng=alloc<unsigned>(262144);auto bfsm=alloc<unsigned char>(16384);auto same=alloc<unsigned char>(1024);auto smax=alloc<int>(1024);auto sbest=alloc<int>(1024);auto old=alloc<int>(1024);
 auto run=[&](int j){CK(cudaMemset(best,0,19*sizeof(int)));FSMKernel<<<POPCNT,POPSIZE>>>(ns[j],ds[j],best,rng,bfsm,same,smax,sbest,old);MaxKernel<<<1,1>>>(best,bfsm);CK(cudaGetLastError());};
 for(int j=0;j<3;j++)run(j);CK(cudaDeviceSynchronize());bool all=true;
 for(int trial=1;trial<=5;trial++){
  std::vector<uint64_t>result;double times[3];
  for(int j=0;j<3;j++){
   CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int r=0;r<REPEAT;r++)run(j);CK(cudaDeviceSynchronize());times[j]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/REPEAT;
   append(result,best,19);append(result,rng,262144);append(result,bfsm,16384);append(result,same,1024);append(result,smax,1024);append(result,sbest,1024);append(result,old,1024);
  }
  bool ok=result==expected;all &=ok;uint64_t err=0;for(size_t k=0;k<result.size();k++)err=std::max(err,result[k]>expected[k]?result[k]-expected[k]:expected[k]-result[k]);
  std::vector<uint64_t>out=prefix;out.insert(out.end(),result.begin(),result.end());const char*dir=getenv("STUDY_OUTPUT_DIR");char path[4096];snprintf(path,sizeof(path),"%s/fsm_trial%d.bin",dir?dir:".",trial);std::ofstream f(path,std::ios::binary);f.write((char*)out.data(),out.size()*8);f.close();
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"component_seconds\":[%.12g,%.12g,%.12g],\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%llu,\"max_rel_error\":%s,\"internal_repeats\":10,\"jit_excluded\":true,\"timing_region\":\"sum_three_trace_source_repeatmeans_reset_GA_Max_devicecompletion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,times[0]+times[1]+times[2],times[0],times[1],times[2],ok?"true":"false",result.size(),(unsigned long long)err,ok?"0":"null",path,out.size());fflush(stdout);
 }
 puts(all?"PASS":"FAIL");return all?0:1;
}

#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <cstring>
#include <chrono>
#define main historical_main
#include "../kiss-cuda/main.cu"
#undef main
using U=uint64_t;const U SENT=0xdeadbeef12345678ULL;
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
std::vector<U>load(const char*path){std::ifstream f(path,std::ios::binary|std::ios::ate);size_t n=size_t(f.tellg());std::vector<U>a(n/8);f.seekg(0);f.read((char*)a.data(),n);return a;}
struct Case{U n;uint32_t seed;int reps;U*raw;};
void run(Case&c){uniform_distribution<U,kiss::Kiss<U>>(c.raw+4,c.n,c.seed);}
int main(){auto f=load(KISS_FIXTURE),o=load(KISS_ORACLE);std::vector<Case>cases;for(int ci=0;ci<3;ci++){Case c{f[3*ci],uint32_t(f[3*ci+1]),int(f[3*ci+2]),nullptr};std::vector<U>z(c.n+8,SENT);ck(cudaMalloc(&c.raw,z.size()*8));ck(cudaMemcpy(c.raw,z.data(),z.size()*8,cudaMemcpyHostToDevice));cases.push_back(c);}for(auto&c:cases){run(c);ck(cudaDeviceSynchronize());}
for(int trial=1;trial<=5;trial++){auto payload=f;bool ok=true;double elapsed=0;std::vector<double>times;size_t oi=0;for(auto&c:cases){std::vector<U>z(c.n+8,SENT);ck(cudaMemcpy(c.raw,z.data(),z.size()*8,cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int rep=0;rep<c.reps;rep++)run(c);ck(cudaDeviceSynchronize());double t=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/c.reps;elapsed+=t;times.push_back(t);ck(cudaMemcpy(z.data(),c.raw,z.size()*8,cudaMemcpyDeviceToHost));for(int j=0;j<4;j++)ok&=z[j]==SENT&&z[c.n+4+j]==SENT;for(U j=0;j<c.n;j++)ok&=z[j+4]==o[oi++];payload.insert(payload.end(),z.begin(),z.end());}std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream out(path,std::ios::binary);out.write((char*)payload.data(),payload.size()*8);out.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%s,\"timing_region\":\"sum3_KISS64_3launch_means_complete\",\"jit_excluded\":true,\"internal_repeats\":3,\"timed_kernel_calls\":9,\"case_times_seconds\":[",trial,elapsed,ok?"true":"false",payload.size(),ok?"0":"null");for(size_t i=0;i<times.size();i++)printf("%s%.17g",i?",":"",times[i]);printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",path.c_str(),payload.size());if(!ok)return 3;}for(auto&c:cases)ck(cudaFree(c.raw));}

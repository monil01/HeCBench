#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <chrono>
#ifdef STUDY_ORIGINAL
#include "../fpdc-cuda/kernels.h"
#else
#include "native_corrected/kernels.h"
#endif
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA:%s\n",cudaGetErrorString(e));exit(2);}}while(0)
template<typename T>struct Buf{T*base;T*p;int n,g;T sentinel;Buf(int n,int g,T s):n(n),g(g),sentinel(s){CHECK(cudaMalloc(&base,(n+2*g)*sizeof(T)));p=base+g;std::vector<T>v(n+2*g,sentinel);CHECK(cudaMemcpy(base,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}~Buf(){cudaFree(base);}void put(const T*x){CHECK(cudaMemcpy(p,x,n*sizeof(T),cudaMemcpyHostToDevice));}std::vector<T>get(){std::vector<T>v(n);CHECK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}bool guards(std::vector<uint64_t>&out){std::vector<T>v(n+2*g);CHECK(cudaMemcpy(v.data(),base,v.size()*sizeof(T),cudaMemcpyDeviceToHost));bool ok=true;for(int i=0;i<g;i++){ok&=v[i]==sentinel;out.push_back((uint64_t)v[i]);}for(int i=n+g;i<n+2*g;i++){ok&=v[i]==sentinel;out.push_back((uint64_t)v[i]);}return ok;}};
template<typename T>std::vector<T>load(std::string name,int n){std::ifstream f("/home/imo/HeCBench_julia/src/fpdc-julia/"+name,std::ios::binary);std::vector<T>x(n);f.read((char*)x.data(),n*sizeof(T));if(!f)exit(3);return x;}
int main(){
#ifdef STUDY_ORIGINAL
 const int trials=1,R=1;const bool diagnostic=true;
#else
 const int trials=5,R=3;const bool diagnostic=false;
#endif
 Buf<unsigned long long>input(512,16,0xa55aa55aa55aa55aULL);Buf<unsigned char>bytes(4352,32,165);Buf<int>cuts(8,16,0x13572468),offsets(8,16,0x13572468);auto cv=load<int>("study_cuts_i32.bin",8);cuts.put(cv.data());bool all=true;
#ifndef STUDY_ORIGINAL
 auto warm=load<unsigned long long>("study_case1_u64.bin",512);input.put(warm.data());for(int d:{1,2,3,32}){CompressionKernel<<<4,64>>>(d,input.p,(char*)bytes.p,cuts.p,offsets.p);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());}
#endif
 for(int trial=1;trial<=trials;trial++){std::vector<uint64_t>prefix,outputs,guards;std::vector<double>times;bool ok=true;size_t mismatch=0;
 for(int c:{1,2})for(int d:{1,2,3,32}){auto host=load<unsigned long long>("study_case"+std::to_string(c)+"_u64.bin",512);auto oracle=load<unsigned char>("oracle_case"+std::to_string(c)+"_d"+std::to_string(d)+"_payload.bin",4352);auto oo=load<int>("oracle_case"+std::to_string(c)+"_d"+std::to_string(d)+"_off_i32.bin",8);input.put(host.data());CHECK(cudaMemset(bytes.p,165,4352));std::vector<int>sent(8,0x13572468);offsets.put(sent.data());CHECK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int k=0;k<R;k++){CompressionKernel<<<4,64>>>(d,input.p,(char*)bytes.p,cuts.p,offsets.p);CHECK(cudaGetLastError());}CHECK(cudaDeviceSynchronize());times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/R);auto av=input.get();auto ac=cuts.get();ok&=av==host&&ac==cv;prefix.insert(prefix.end(),av.begin(),av.end());for(auto x:ac)prefix.push_back(uint32_t(x));for(auto x:{512,4,2,d,R,64,4352,8})prefix.push_back(x);auto ob=bytes.get();auto of=offsets.get();for(size_t k=0;k<ob.size();k++){outputs.push_back(ob[k]);mismatch+=ob[k]!=oracle[k];}for(size_t k=0;k<of.size();k++){outputs.push_back(uint32_t(of[k]));mismatch+=of[k]!=oo[k];}ok&=ob==oracle&&of==oo;ok&=input.guards(guards);ok&=bytes.guards(guards);ok&=cuts.guards(guards);ok&=offsets.guards(guards);}
 all&=ok;auto artifact=prefix;artifact.insert(artifact.end(),outputs.begin(),outputs.end());artifact.insert(artifact.end(),guards.begin(),guards.end());std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/trial"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)artifact.data(),artifact.size()*8);f.close();double total=0;for(double v:times)total+=v;
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":",trial);if(diagnostic)printf("null");else printf("%.12g",total);printf(",\"correct\":%s,\"checked_values\":36160,\"max_abs_error\":%s,\"max_rel_error\":null,\"mismatch_values\":%zu,\"timing_region\":\"sum8_fpdc_compression_kernel_launch_loop_means\",\"jit_excluded\":true,\"internal_repeats\":%d,\"timed_cases\":8,\"diagnostic_only\":%s,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",ok?"true":"false",ok?"0":"null",mismatch,R,diagnostic?"true":"false",path.c_str(),artifact.size());}
 return diagnostic?1:(all?0:1);
}

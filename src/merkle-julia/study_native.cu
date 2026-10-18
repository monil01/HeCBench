#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdio>
#include "../merkle-cuda/ff_p.cu"
#include "../merkle-cuda/rescue_prime.cu"
#include "merkle_tree_corrected.inc"
const ulong GUARD=0xdeadbeef12345678ULL;
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
std::vector<ulong>load(const char*path){std::ifstream f(path,std::ios::binary|std::ios::ate);size_t z=f.tellg();std::vector<ulong>a(z/8);f.seekg(0);f.read((char*)a.data(),z);return a;}
struct Buf{std::vector<ulong>h;ulong*raw;ulong*ptr;Buf(std::vector<ulong>v):h(v.size()+8,GUARD){std::copy(v.begin(),v.end(),h.begin()+4);ck(cudaMalloc(&raw,h.size()*8));ptr=raw+4;reset();}void reset(){ck(cudaMemcpy(raw,h.data(),h.size()*8,cudaMemcpyHostToDevice));}std::vector<ulong>get(){std::vector<ulong>a(h.size());ck(cudaMemcpy(a.data(),raw,a.size()*8,cudaMemcpyDeviceToHost));return a;}bool guards(std::vector<ulong>&a){for(int i=0;i<4;i++)if(a[i]!=GUARD||a[a.size()-1-i]!=GUARD)return false;return true;}};
int main(){auto f=load(MERKLE_FIXTURE),oracle=load(MERKLE_ORACLE);int nc=f[0];size_t pos=1;auto take=[&](size_t n){std::vector<ulong>a(f.begin()+pos,f.begin()+pos+n);pos+=n;return a;};Buf mds(take(144)),a1(take(84)),a2(take(84));struct Case{size_t n;Buf*leaves;Buf*out;};std::vector<Case>cases;for(int i=0;i<nc;i++){size_t n=f[pos++];auto lv=take(4*n);cases.push_back({n,new Buf(lv),new Buf(std::vector<ulong>(4*n))});}auto launch=[&](Case&c){merklize_approach_1(c.leaves->ptr,c.out->ptr,c.n,32,(ulong4*)mds.ptr,(ulong4*)a1.ptr,(ulong4*)a2.ptr);ck(cudaGetLastError());};for(auto&c:cases)launch(c);
for(int trial=1;trial<=5;trial++){auto payload=f;bool ok=true;double total=0;std::vector<double>times;for(auto*b:{&mds,&a1,&a2}){auto a=b->get();ok&=b->guards(a)&&a==b->h;payload.insert(payload.end(),a.begin(),a.end());}size_t oi=0;
for(auto&c:cases){c.out->reset();ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();launch(c);double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();total+=elapsed;times.push_back(elapsed);auto l=c.leaves->get(),o=c.out->get();ok&=c.leaves->guards(l)&&l==c.leaves->h&&c.out->guards(o);for(size_t j=0;j<4*c.n;j++)ok&=(o[4+j]%MOD)==oracle[oi+j];oi+=4*c.n;payload.insert(payload.end(),l.begin(),l.end());payload.insert(payload.end(),o.begin(),o.end());}
std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream out(path,std::ios::binary);out.write((char*)payload.data(),payload.size()*8);out.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%s,\"timing_region\":\"sum3_full_merklize_contextsynchronized_alllevels\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_kernel_calls\":21,\"case_times_seconds\":[",trial,total,ok?"true":"false",payload.size(),ok?"0":"null");for(size_t j=0;j<times.size();j++)printf("%s%.17g",j?",":"",times[j]);printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",path.c_str(),payload.size());if(!ok)return 3;}}

#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../../bsw-cuda/kernel.cu"
#include "../study_params.h"
constexpr int SWAP_NA=NB,SWAP_NB=NA;
const unsigned char *SWAP_A=SEQ_B,*SWAP_B=SEQ_A;
const unsigned *SWAP_OA=OFF_B,*SWAP_OB=OFF_A;
#define NA SWAP_NA
#define NB SWAP_NB
#define SEQ_A SWAP_A
#define SEQ_B SWAP_B
#define OFF_A SWAP_OA
#define OFF_B SWAP_OB
#define C(x) do{auto status=(x);if(status!=cudaSuccess){fprintf(stderr,"CUDA:%s at %d\n",cudaGetErrorString(status),__LINE__);exit(1);}}while(0)
template<class T>struct Buffer{std::vector<T> init;T *base,*ptr;Buffer(std::vector<T> v,T guard){init.assign(16,guard);init.insert(init.end(),v.begin(),v.end());init.insert(init.end(),16,guard);C(cudaMalloc(&base,init.size()*sizeof(T)));ptr=base+16;reset();}void reset(){C(cudaMemcpy(base,init.data(),init.size()*sizeof(T),cudaMemcpyHostToDevice));}void capture(std::vector<int>& out){std::vector<T> a(init.size());C(cudaMemcpy(a.data(),base,a.size()*sizeof(T),cudaMemcpyDeviceToHost));for(auto v:a)out.push_back((int)v);}~Buffer(){cudaFree(base);}};
struct Work{
 Buffer<unsigned char>a,b;Buffer<unsigned>oa,ob;Buffer<short>scores,encoding;std::vector<Buffer<short>*> outputs;std::vector<short> endA,endB,startA,startB,top;std::vector<std::string> refs,queries;std::vector<unsigned char> packA,packB;std::vector<unsigned> offsetA,offsetB;int reverse;
 Work():a(std::vector<unsigned char>(NA,0),211),b(std::vector<unsigned char>(NB,0),211),oa(std::vector<unsigned>(N,0),24681357),ob(std::vector<unsigned>(N,0),24681357),scores(std::vector<short>(SCORING,SCORING+576),-12345),encoding(std::vector<short>(ENCODING,ENCODING+91),-12345),endA(N),endB(N),startA(N),startB(N),top(N),packA(NA),packB(NB),offsetA(N),offsetB(N){unsigned pa=0,pb=0;for(int k=0;k<N;k++){refs.emplace_back((const char*)SEQ_A+pa,OFF_A[k]-pa);queries.emplace_back((const char*)SEQ_B+pb,OFF_B[k]-pb);pa=OFF_A[k];pb=OFF_B[k];}for(int k=0;k<5;k++)outputs.push_back(new Buffer<short>(std::vector<short>(N,-12345),-12345));}
 ~Work(){for(auto p:outputs)delete p;}
 void reset(){a.reset();b.reset();oa.reset();ob.reset();for(auto o:outputs)o->reset();}
 void inputs(std::vector<int>&p){a.capture(p);b.capture(p);oa.capture(p);ob.capture(p);scores.capture(p);encoding.capture(p);}
 void capture_outputs(std::vector<int>&p){for(auto v:outputs)v->capture(p);}
 void launch(bool inv,int threads){sequence_aa_kernel<<<N,threads,SHMEM>>>(inv,(const char*)a.ptr,(const char*)b.ptr,oa.ptr,ob.ptr,outputs[0]->ptr,outputs[1]->ptr,outputs[2]->ptr,outputs[3]->ptr,outputs[4]->ptr,-6,-1,scores.ptr,encoding.ptr);C(cudaGetLastError());}
 void pipeline(){
  // Source its=1 hostpacking + transfer + dependent two-pass kernel pipeline.
  std::vector<std::string> sequencesA(refs.begin(),refs.end()),sequencesB(queries.begin(),queries.end());unsigned runA=0,runB=0;for(int k=0;k<N;k++){runA+=sequencesA[k].size();offsetA[k]=runA;runB+=sequencesB[k].size();offsetB[k]=runB;}unsigned posA=0,posB=0;for(int k=0;k<N;k++){memcpy(packA.data()+posA,sequencesA[k].data(),sequencesA[k].size());memcpy(packB.data()+posB,sequencesB[k].data(),sequencesB[k].size());posA+=sequencesA[k].size();posB+=sequencesB[k].size();}
  C(cudaMemcpyAsync(oa.ptr,offsetA.data(),N*4,cudaMemcpyHostToDevice,0));C(cudaMemcpyAsync(ob.ptr,offsetB.data(),N*4,cudaMemcpyHostToDevice,0));C(cudaMemcpyAsync(a.ptr,packA.data(),NA,cudaMemcpyHostToDevice,0));C(cudaMemcpyAsync(b.ptr,packB.data(),NB,cudaMemcpyHostToDevice,0));
  launch(false,FORWARD);C(cudaMemcpyAsync(endA.data(),outputs[1]->ptr,N*2,cudaMemcpyDeviceToHost,0));C(cudaMemcpyAsync(endB.data(),outputs[3]->ptr,N*2,cudaMemcpyDeviceToHost,0));C(cudaDeviceSynchronize());
  reverse=std::min(*std::max_element(endA.begin(),endA.end()),*std::max_element(endB.begin(),endB.end()));launch(true,reverse);
  C(cudaMemcpyAsync(startA.data(),outputs[0]->ptr,N*2,cudaMemcpyDeviceToHost,0));C(cudaMemcpyAsync(startB.data(),outputs[2]->ptr,N*2,cudaMemcpyDeviceToHost,0));C(cudaMemcpyAsync(top.data(),outputs[4]->ptr,N*2,cudaMemcpyDeviceToHost,0));C(cudaDeviceSynchronize());
 }
};
int main(int argc,char**argv){Work w;std::vector<int> before,initial;w.inputs(before);w.capture_outputs(initial);w.pipeline();std::vector<int> actual={N,128,160,-6,-1,FORWARD,w.reverse};actual.insert(actual.end(),before.begin(),before.end());w.inputs(actual);for(auto v:w.endA)actual.push_back(v);for(auto v:w.endB)actual.push_back(v);actual.insert(actual.end(),initial.begin(),initial.end());w.capture_outputs(actual);std::ifstream f(argv[2],std::ios::binary);std::vector<int> expected(COUNT);f.read((char*)expected.data(),COUNT*4);bool ok=f.gcount()==COUNT*4&&actual==expected;std::ofstream out(argv[1],std::ios::binary);out.write((char*)actual.data(),actual.size()*4);out.close();printf("DIAGNOSTIC swapped_orientation actual_values=%zu correct=%s timing_excluded=true\n",actual.size(),ok?"true":"false");return ok?0:1;}

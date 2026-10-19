#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include "../bsw-cuda/kernel.cu"
#include "study_params.h"
#define C(x) do{auto status=(x);if(status!=cudaSuccess){fprintf(stderr,"CUDA:%s at %d\n",cudaGetErrorString(status),__LINE__);exit(1);}}while(0)
template<class T>struct Buffer{std::vector<T> init;T *base,*ptr;Buffer(std::vector<T> v,T guard){init.assign(16,guard);init.insert(init.end(),v.begin(),v.end());init.insert(init.end(),16,guard);C(cudaMalloc(&base,init.size()*sizeof(T)));ptr=base+16;reset();}void reset(){C(cudaMemcpy(base,init.data(),init.size()*sizeof(T),cudaMemcpyHostToDevice));}void capture(std::vector<int>& out){std::vector<T> a(init.size());C(cudaMemcpy(a.data(),base,a.size()*sizeof(T),cudaMemcpyDeviceToHost));for(auto v:a)out.push_back((int)v);}~Buffer(){cudaFree(base);}};
struct Work{
 Buffer<unsigned char>a,b;Buffer<unsigned>oa,ob;Buffer<short>scores,encoding;std::vector<Buffer<short>*> outputs;std::vector<short> endA,endB,startA,startB,top;int reverse;
 Work():a(std::vector<unsigned char>(NA,0),211),b(std::vector<unsigned char>(NB,0),211),oa(std::vector<unsigned>(N,0),24681357),ob(std::vector<unsigned>(N,0),24681357),scores(std::vector<short>(SCORING,SCORING+576),-12345),encoding(std::vector<short>(ENCODING,ENCODING+91),-12345),endA(N),endB(N),startA(N),startB(N),top(N){for(int k=0;k<5;k++)outputs.push_back(new Buffer<short>(std::vector<short>(N,-12345),-12345));}
 ~Work(){for(auto p:outputs)delete p;}
 void reset(){a.reset();b.reset();oa.reset();ob.reset();for(auto o:outputs)o->reset();}
 void inputs(std::vector<int>&p){a.capture(p);b.capture(p);oa.capture(p);ob.capture(p);scores.capture(p);encoding.capture(p);}
 void capture_outputs(std::vector<int>&p){for(auto v:outputs)v->capture(p);}
 void launch(bool inv,int threads){sequence_aa_kernel<<<N,threads,SHMEM>>>(inv,(const char*)a.ptr,(const char*)b.ptr,oa.ptr,ob.ptr,outputs[0]->ptr,outputs[1]->ptr,outputs[2]->ptr,outputs[3]->ptr,outputs[4]->ptr,-6,-1,scores.ptr,encoding.ptr);C(cudaGetLastError());}
 void pipeline(){
  // Source its=1 hostpacking + transfer + dependent two-pass kernel pipeline.
  std::vector<unsigned char> packA(SEQ_A,SEQ_A+NA),packB(SEQ_B,SEQ_B+NB);std::vector<unsigned> offsetA(OFF_A,OFF_A+N),offsetB(OFF_B,OFF_B+N);
  C(cudaMemcpyAsync(oa.ptr,offsetA.data(),N*4,cudaMemcpyHostToDevice,0));C(cudaMemcpyAsync(ob.ptr,offsetB.data(),N*4,cudaMemcpyHostToDevice,0));C(cudaMemcpyAsync(a.ptr,packA.data(),NA,cudaMemcpyHostToDevice,0));C(cudaMemcpyAsync(b.ptr,packB.data(),NB,cudaMemcpyHostToDevice,0));
  launch(false,FORWARD);C(cudaMemcpyAsync(endA.data(),outputs[1]->ptr,N*2,cudaMemcpyDeviceToHost,0));C(cudaMemcpyAsync(endB.data(),outputs[3]->ptr,N*2,cudaMemcpyDeviceToHost,0));C(cudaDeviceSynchronize());
  reverse=std::min(*std::max_element(endA.begin(),endA.end()),*std::max_element(endB.begin(),endB.end()));launch(true,reverse);
  C(cudaMemcpyAsync(startA.data(),outputs[0]->ptr,N*2,cudaMemcpyDeviceToHost,0));C(cudaMemcpyAsync(startB.data(),outputs[2]->ptr,N*2,cudaMemcpyDeviceToHost,0));C(cudaMemcpyAsync(top.data(),outputs[4]->ptr,N*2,cudaMemcpyDeviceToHost,0));C(cudaDeviceSynchronize());
 }
};
int main(int argc,char**argv){std::ifstream f(argv[1],std::ios::binary);std::vector<int> expected(COUNT);f.read((char*)expected.data(),COUNT*4);if(f.gcount()!=COUNT*4)return 1;Work w;w.reset();w.pipeline();
 for(int trial=1;trial<=5;trial++){
  w.reset();std::vector<int> before,initial;w.inputs(before);w.capture_outputs(initial);C(cudaDeviceSynchronize());auto begin=std::chrono::steady_clock::now();w.pipeline();C(cudaDeviceSynchronize());double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
  std::vector<int> actual={N,160,128,-6,-1,FORWARD,w.reverse};actual.insert(actual.end(),before.begin(),before.end());w.inputs(actual);for(auto v:w.endA)actual.push_back(v);for(auto v:w.endB)actual.push_back(v);actual.insert(actual.end(),initial.begin(),initial.end());if(actual.size()!=PREFIX)return 2;w.capture_outputs(actual);bool ok=actual==expected;
  // Host copies are source-observable outputs too; check all against actualGPU buffers.
  int offset=PREFIX;for(int k=0;k<N;k++){ok=ok&&w.startA[k]==actual[offset+16+k]&&w.endA[k]==actual[offset+(N+32)+16+k]&&w.startB[k]==actual[offset+2*(N+32)+16+k]&&w.endB[k]==actual[offset+3*(N+32)+16+k]&&w.top[k]==actual[offset+4*(N+32)+16+k];}
  std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/bsw_trial"+std::to_string(trial)+".bin";std::ofstream o(path,std::ios::binary);o.write((char*)actual.data(),actual.size()*4);o.close();
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%s,\"max_rel_error\":null,\"jit_excluded\":true,\"timing_region\":\"named_source_its1_hostpacking_H2D_forward_host_end_feedback_reverse_D2H\",\"internal_repeats\":1,\"iterations\":1,\"forward_threads\":%d,\"reverse_threads\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,sec,ok?"true":"false",actual.size(),ok?"0":"null",FORWARD,w.reverse,path.c_str(),actual.size());fflush(stdout);if(!ok)return 3;
 }
}

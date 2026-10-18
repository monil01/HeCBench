// Original NVIDIA-independent MILC SU3 benchmark kernel; LICENSE.source.
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
typedef std::chrono::system_clock Clock;
int verbose=0,warmups=1;
#include "../su3-cuda/lattice.hpp"
#include "../su3-cuda/mat_nn_cuda.hpp"
#include "study_cases.h"
static_assert(sizeof(site)==320&&alignof(site)==16&&sizeof(su3_matrix)==72);
static_assert(offsetof(site,x)==288&&offsetof(site,parity)==308&&offsetof(site,pad)==312);
#define CK(x) do{if((x)!=cudaSuccess)throw std::runtime_error(#x);}while(0)
const int G=32,R=3;const unsigned GUARD=0xa5a5a5a5;
std::vector<unsigned>readv(const char*n){std::ifstream f(n,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("openfixture");size_t bytes=f.tellg();if(bytes%4)throw std::runtime_error("dtype");f.seekg(0);std::vector<unsigned>v(bytes/4);if(!f.read((char*)v.data(),bytes))throw std::runtime_error("shortfixture");return v;}
struct State{Case c;std::vector<unsigned>host[3];unsigned*dev[3];State(Case q,std::vector<unsigned>&f):c(q){int sizes[]={80*q.n,72,80*q.n};int off=q.io;for(int k=0;k<3;k++){host[k].assign(sizes[k]+2*G,GUARD);if(k<2){std::copy(f.begin()+off,f.begin()+off+sizes[k],host[k].begin()+G);off+=sizes[k];}CK(cudaMalloc(&dev[k],host[k].size()*4));CK(cudaMemcpy(dev[k],host[k].data(),host[k].size()*4,cudaMemcpyHostToDevice));}}std::vector<unsigned>get(int k){auto v=host[k];CK(cudaMemcpy(v.data(),dev[k],v.size()*4,cudaMemcpyDeviceToHost));return v;}void reset(){CK(cudaMemcpy(dev[2],host[2].data(),host[2].size()*4,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());}void launch(){k_mat_nn<<<c.n,c.threads>>>((const site*)(dev[0]+G),(const su3_matrix*)(dev[1]+G),(site*)(dev[2]+G),c.n);CK(cudaGetLastError());}};
void append(std::vector<unsigned>&o,std::vector<unsigned>v){o.insert(o.end(),v.begin(),v.end());}
void capture(State&s,std::vector<unsigned>&o){for(int k=0;k<3;k++)append(o,s.get(k));}
bool check_and_capture(State&s,std::vector<unsigned>&oracle,std::vector<unsigned>&out){bool ok=true;for(int k=0;k<3;k++){auto v=s.get(k);for(int j=0;j<(int)v.size();j++){unsigned want=k<2?s.host[k][j]:(j>=G&&j<(int)v.size()-G?oracle[s.c.oo+j-G]:GUARD);ok&=v[j]==want;}append(out,v);}return ok;}
int main(){auto f=readv("input_u32.bin"),o=readv("oracle_u32.bin");std::vector<State*>states;for(auto c:cases){auto*s=new State(c,f);s->launch();CK(cudaDeviceSynchronize());auto v=s->get(2);bool ok=true;int first=-1;for(int i=0;i<(int)v.size();i++){unsigned want=i>=G&&i<(int)v.size()-G?o[c.oo+i-G]:GUARD;if(v[i]!=want){ok=false;if(first<0)first=i;}}printf("STUDY_WARM sites=%d correct=%d\n",c.n,int(ok));fflush(stdout);if(!ok){std::string path="/home/imo/HeCBench_julia/.porting-state/portable_julia/su3_reference_review/native_warm_"+std::to_string(c.n)+".bin";std::ofstream(path,std::ios::binary).write((char*)v.data(),v.size()*4);printf("first_mismatch_word=%d actual=%u\n",first,v[first]);throw std::runtime_error("warmfullAoS_oracle");}states.push_back(s);}for(int trial=1;trial<=5;trial++){std::vector<unsigned>pre,post;bool ok=true;double sec=0;std::vector<double>times;for(auto*s:states){s->reset();capture(*s,pre);auto start=std::chrono::steady_clock::now();for(int r=0;r<R;r++)s->launch();CK(cudaDeviceSynchronize());double tm=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/R;times.push_back(tm);sec+=tm;ok&=check_and_capture(*s,o,post);}pre.insert(pre.end(),post.begin(),post.end());std::string path="su3_trial"+std::to_string(trial)+".bin";std::ofstream file(path,std::ios::binary);if(!file.write((char*)pre.data(),pre.size()*4))throw std::runtime_error("artifactwrite");file.close();printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"internal_repeats\":3,\"timing_region\":\"sum_seven_original_AoS_complex_matrix_batches_three_calls_completion\",\"max_abs_error\":null,\"max_rel_error\":null,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%zu,\"shape\":[%zu],\"byte_order\":\"little\"},\"case_times_seconds\":[",trial,ok?"true":"false",sec,path.c_str(),pre.size(),pre.size());for(size_t i=0;i<times.size();i++)printf("%s%.12g",i?",":"",times[i]);printf("]}\n");fflush(stdout);}}

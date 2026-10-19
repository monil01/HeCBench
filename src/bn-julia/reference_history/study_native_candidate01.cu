#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <cstdint>
#include <type_traits>
#include "bn_host_pipeline.h"
#include "kernels.h"
#ifndef FIXTURE_ROOT
#define FIXTURE_ROOT "."
#endif
static constexpr int G=32;
void ck(cudaError_t x){if(x!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(x));exit(2);}}
std::vector<double>load(const char*name){std::string path=std::string(FIXTURE_ROOT)+"/"+name;std::ifstream f(path,std::ios::binary|std::ios::ate);assert(f);size_t n=f.tellg();assert(n%8==0);std::vector<double>a(n/8);f.seekg(0);f.read((char*)a.data(),n);assert(f);return a;}
template<class T>struct Buf{T*base;std::vector<T>h;int n;Buf(int N,const T*src=nullptr):n(N){h.assign(N+64,T(-37));if constexpr(std::is_same_v<T,unsigned char>)std::fill(h.begin(),h.end(),173);for(int i=0;i<N;i++)h[G+i]=src?src[i]:T(0);ck(cudaMalloc(&base,h.size()*sizeof(T)));ck(cudaMemcpy(base,h.data(),h.size()*sizeof(T),cudaMemcpyHostToDevice));}~Buf(){ck(cudaFree(base));}T*p(){return base+G;}std::vector<T>read(){std::vector<T>a(h.size());ck(cudaMemcpy(a.data(),base,a.size()*sizeof(T),cudaMemcpyDeviceToHost));return a;}void append(std::vector<double>&out){auto a=read();out.insert(out.end(),a.begin(),a.end());}bool guards(std::vector<double>&out){auto a=read();out.insert(out.end(),a.begin(),a.end());return std::equal(a.begin(),a.begin()+G,h.begin())&&std::equal(a.end()-G,a.end(),h.end()-G);}};
int main(){auto fixture=load("fixture.bin"),cpu=load("score_oracle.bin"),mathgold=load("math_score_oracle.bin"),mathbound=load("math_score_budget.bin");
for(int trial=0;trial<=5;trial++){std::vector<double>prefix{3},computed,extra;bool ok=true;double secs=0,gens=0,search=0,maxerr=0,matherr=0;size_t off=1,co=0;
for(int ci=0;ci<3;ci++){bn_reset(ci);for(int i=0;i<4;i++)prefix.push_back(fixture[off+i]);off+=4;assert(fixture[off]==data[0]);off+=NODE_N*DATA_N+DATA_N+2;Buf<int>ddata(NODE_N*DATA_N,data),parents(4);Buf<unsigned char>mask(NODE_N);Buf<float>gamma(DATA_N+2,LG),local(NODE_N*sizepernode),dscore(1);
ddata.append(prefix);gamma.append(prefix);local.append(prefix);mask.append(prefix);dscore.append(prefix);parents.append(prefix);
ck(cudaDeviceSynchronize());auto begin=std::chrono::steady_clock::now();ck(cudaMemset(local.p(),0,NODE_N*sizepernode*4));genScoreKernel<<<sizepernode/256+1,256>>>(sizepernode,local.p(),ddata.p(),gamma.p());ck(cudaGetLastError());ck(cudaDeviceSynchronize());double t=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();gens+=t;secs+=t;
auto scores=local.read();for(int i=0;i<NODE_N*sizepernode;i++){double x=scores[G+i],e=std::abs(x-cpu[co+i]);maxerr=std::max(maxerr,e);matherr=std::max(matherr,std::abs(x-mathgold[co+i]));ok&=std::isfinite(x)&&e<=1e-3+2e-6*std::abs(cpu[co+i])&&std::abs(x-mathgold[co+i])<=mathbound[co+i];computed.push_back(x);}co+=NODE_N*sizepernode;
for(int iteration=0;iteration<ITER;iteration++){bn_prepare_iteration();for(int node=0;node<NODE_N;node++){int total=bn_total(node),blocks=total/(256*WORKLOAD)+1;assert(blocks==1);
ck(cudaDeviceSynchronize());begin=std::chrono::steady_clock::now();ck(cudaMemset(parents.p(),0,4*sizeof(int)));ck(cudaMemset(dscore.p(),int(-999999.f),sizeof(float)));ck(cudaMemcpy(mask.p(),bn_mask(node),NODE_N,cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());t=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();search+=t;secs+=t;
// Full actualmask/guards echo outside both explicitly segmented measuredregions.
auto actual=mask.read();for(int k=0;k<NODE_N;k++)ok&=actual[G+k]==bn_mask(node)[k];prefix.insert(prefix.end(),actual.begin(),actual.end());
ck(cudaDeviceSynchronize());begin=std::chrono::steady_clock::now();computeKernel<<<blocks,256,256*sizeof(float)>>>(WORKLOAD,sizepernode,local.p(),(bool*)mask.p(),node,total,dscore.p(),parents.p());ck(cudaGetLastError());float sc;int par[4];ck(cudaMemcpy(&sc,dscore.p(),4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(par,parents.p(),16,cudaMemcpyDeviceToHost));bn_accept_node(node,blocks,&sc,par);ck(cudaDeviceSynchronize());t=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();search+=t;secs+=t;
computed.push_back(sc);for(int x:par)computed.push_back(x);for(int i=0;i<4;i++){if(par[i]<0)break;ok&=par[i]>=1&&par[i]<NODE_N;}
}
bn_finish_iteration();double snap[573];bn_snapshot(snap);computed.insert(computed.end(),snap,snap+573);for(int node=0;node<NODE_N;node++)ok&=!graph[node][node];}
ok&=ddata.guards(extra)&&gamma.guards(extra)&&local.guards(extra)&&mask.guards(extra)&&dscore.guards(extra)&&parents.guards(extra);ok&=ddata.read()==ddata.h&&gamma.read()==gamma.h;bn_release();}
assert(off==fixture.size()&&co==cpu.size());if(trial==0){if(!ok){fprintf(stderr,"BN warmupfailed cpuerr=%.17g matherr=%.17g\n",maxerr,matherr);return 3;}continue;}
prefix.insert(prefix.end(),computed.begin(),computed.end());prefix.insert(prefix.end(),extra.begin(),extra.end());const char*dir=getenv("STUDY_OUTPUT_DIR");std::string path=std::string(dir?dir:".")+"/output_trial"+std::to_string(trial)+".bin";std::ofstream out(path,std::ios::binary);out.write((char*)prefix.data(),prefix.size()*8);assert(out);out.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"genScore_seconds\":%.17g,\"search_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"independent_math_max_abs_error\":%.17g,\"timing_region\":\"sum3cases_source_scoreplus_segmented100order_search_resets_transfers_GPUreduction_CPUselection_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"native_reference_repaired\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,gens,search,ok?"true":"false",prefix.size(),maxerr,matherr,path.c_str(),prefix.size());fflush(stdout);if(!ok)return 4;
}return 0;}

// Shared original CPU EM/model-order controller; target callbacks own GPU work.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>
#include "configured_native/gaussian_kernel.cu"
#include "independent_stage_oracle.h"
using Clock=std::chrono::steady_clock;
static void C(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"GMM_CUDA %s\n",cudaGetErrorString(e));exit(7);}}
struct Alloc {void*base;void*ptr;size_t bytes;int id;};
static std::map<void*,Alloc>allocations;static int alloc_id=0,capacity=0;
static std::vector<double>prefix,trace;static double excluded=0;static size_t stages=0;
static constexpr uint32_t GUARD=0xa5a5a5a5;
static cudaError_t guarded_malloc(void**out,size_t bytes){
 void*base;C(cudaMalloc(&base,bytes+128));C(cudaMemset(base,0xa5,bytes+128));void*p=(char*)base+64;C(cudaMemset(p,0,bytes));*out=p;allocations[p]={base,p,bytes,alloc_id++};return cudaSuccess;
}
static std::vector<uint32_t>guarded_words(void*p){
 auto it=allocations.find(p);if(it==allocations.end()){fprintf(stderr,"GMM_UNBOUND_POINTER\n");exit(7);}auto&a=it->second;
 std::vector<uint32_t>v(a.bytes/4+32);C(cudaMemcpy(v.data(),a.base,a.bytes+128,cudaMemcpyDeviceToHost));
 for(size_t i=0;i<v.size();i++)if(i<16||i>=v.size()-16)if(v[i]!=GUARD){fprintf(stderr,"GMM_GUARD alloc%d word%zu\n",a.id,i);exit(7);}return v;
}
static cudaError_t guarded_free(void*p){if(!p)return cudaSuccess;auto a=allocations.at(p);guarded_words(p);allocations.erase(p);return cudaFree(a.base);}
static std::vector<float>float_values(float*p){auto w=guarded_words(p);std::vector<float>x(w.size()-32);memcpy(x.data(),w.data()+16,x.size()*4);return x;}
static GMMState get_state(const clusters_t&c){return {float_values(c.N),float_values(c.pi),float_values(c.constant),float_values(c.avgvar),float_values(c.means),float_values(c.R),float_values(c.Rinv),float_values(c.memberships)};}
static void append_state(const clusters_t&c,float*like){
 for(auto*p:{c.N,c.pi,c.constant,c.avgvar,c.means,c.R,c.Rinv,c.memberships}){
  auto w=guarded_words(p);trace.push_back(w.size());for(auto q:w){float v;memcpy(&v,&q,4);if(!std::isfinite(v)){fprintf(stderr,"GMM_NONFINITE_STATE\n");exit(8);}trace.push_back(v);}
 }
 if(like){auto w=guarded_words(like);trace.push_back(w.size());for(auto q:w){float v;memcpy(&v,&q,4);if(!std::isfinite(v))exit(8);trace.push_back(v);}}else trace.push_back(0);
}
using Callback=int(*)(int,int,int,int,const float*,const clusters_t*,int,int,int,int,float*);
static Callback target_callback=nullptr;
static void gmm_dispatch(int stage,int gx,int gy,int threads,float*data,clusters_t*device_clusters,int d,int m,int n,float*like){
 auto observation_start=Clock::now();C(cudaDeviceSynchronize());clusters_t fields;C(cudaMemcpy(&fields,device_clusters,sizeof(fields),cudaMemcpyDeviceToHost));
 if(stages==0){
  prefix={double(d),double(n),double(capacity)};
  std::vector<Alloc>ordered;for(auto &p:allocations)ordered.push_back(p.second);std::sort(ordered.begin(),ordered.end(),[](auto&a,auto&b){return a.id<b.id;});
  for(auto&a:ordered){if(a.ptr==device_clusters)continue;auto w=guarded_words(a.ptr);prefix.push_back(a.id);prefix.push_back(w.size());for(auto q:w)prefix.push_back(double(q));}
 }
 auto before=get_state(fields);auto input=float_values(data);trace.insert(trace.end(),{double(stage),double(gx),double(gy),double(threads),double(d),double(m),double(n)});append_state(fields,like);
 excluded+=std::chrono::duration<double>(Clock::now()-observation_start).count();
 if(target_callback){if(target_callback(stage,gx,gy,threads,data,&fields,d,m,n,capacity,like)!=0){fprintf(stderr,"TARGET_CALLBACK_REJECTED stage%d\n",stage);exit(9);}}
 else{
  switch(stage){
   case 1:seed_clusters_kernel<<<gx,threads>>>(data,device_clusters,d,m,n);break;
   case 2:constants_kernel<<<gx,threads>>>(device_clusters,m,d);break;
   case 3:estep1<<<dim3(gx,gy),threads>>>(data,device_clusters,d,n);break;
   case 4:estep2<<<gx,threads>>>(data,device_clusters,d,m,n,like);break;
   case 5:mstep_N<<<gx,threads>>>(device_clusters,d,m,n);break;
   case 6:mstep_means<<<dim3(gx,gy),threads>>>(data,device_clusters,d,m,n);break;
   case 7:mstep_covariance2<<<dim3(gx,gy),threads>>>(data,device_clusters,d,m,n);break;
   default:exit(9);
  }
 }
 C(cudaDeviceSynchronize());observation_start=Clock::now();
 auto after=get_state(fields);std::vector<float>ll;if(like)ll=float_values(like);gmm_scientific_stage(stage,m,d,n,gx,gy,input,before,after,ll);append_state(fields,like);
 // Each selected kernel treats the input dataset as read-only.
 if(float_values(data)!=input){fprintf(stderr,"DATASET_READONLY_FAIL\n");exit(8);}stages++;
 excluded+=std::chrono::duration<double>(Clock::now()-observation_start).count();
}
#define cudaMalloc guarded_malloc
#define cudaFree guarded_free
#include "controller_adapted.inc"
#undef cudaMalloc
#undef cudaFree
#include "configured_native/cpu_reference.h"
#include "configured_native/readData.cu"
extern "C" int gmm_run(Callback callback,int trial,const char*path,int initial,int desired){
 if(!allocations.empty()){fprintf(stderr,"GPUALLOCATION_LEAK\n");return 1;}
 alloc_id=0;capacity=initial;prefix.clear();trace.clear();excluded=0;stages=0;oracle_checked=0;oracle_max_abs=0;oracle_max_rel=0;target_callback=callback;
 int d,n;float*events=readData(const_cast<char*>(path),&d,&n);if(!events||d!=2||n!=992)return 2;
 C(cudaDeviceSynchronize());auto start=Clock::now();int ideal=0;auto*saved=cluster(initial,desired,&ideal,d,n,events);C(cudaDeviceSynchronize());double seconds=std::chrono::duration<double>(Clock::now()-start).count()-excluded;
 if(!saved||ideal<1||ideal>initial||!allocations.empty())return 3;
 trace.insert(trace.end(),{double(ideal),double(stages)});
 for(auto spec:std::vector<std::pair<float*,int>>{{saved->N,ideal},{saved->pi,ideal},{saved->constant,ideal},{saved->avgvar,ideal},{saved->means,ideal*d},{saved->R,ideal*d*d},{saved->Rinv,ideal*d*d},{saved->memberships,ideal*n}})
  for(int i=0;i<spec.second;i++){if(!std::isfinite(spec.first[i]))return 4;trace.push_back(spec.first[i]);}
 if(trial>0){
  std::vector<double>out=prefix;out.insert(out.end(),trace.begin(),trace.end());std::string name="output_trial"+std::to_string(trial)+".bin";std::ofstream f(name,std::ios::binary);f.write((char*)out.data(),out.size()*8);f.close();if(!f)return 5;
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":true,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"metric_scope\":\"IndependentFloat64fullstageGaussianProbabilityMomentsAnalytic2x2inverse_maxrelusesmax1absoracle\",\"jit_excluded\":true,\"internal_repeats\":1,\"input_prefix_count\":%zu,\"timing_region\":\"FullsourceCPUcontroller_allGPUstages_allocation_transfers_convergence_merges_minus_explicit_poststage_observation_intervals\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"},\"active_final_clusters\":%d,\"selected_GPU_stages\":%zu,\"excluded_observation_seconds\":%.17g,\"whole_host_controller_seconds\":%.17g}\n",trial,seconds,oracle_checked,oracle_max_abs,oracle_max_rel,prefix.size(),name.c_str(),out.size(),ideal,stages,excluded,seconds+excluded);fflush(stdout);
 }
 freeCluster(saved);free(saved);free(events);return 0;
}
#ifndef GMM_LIBRARY
int main(int argc,char**argv){if(argc!=2)return 1;int rc=gmm_run(nullptr,0,argv[1],6,0);if(rc)return rc;for(int trial=1;trial<=5;trial++){rc=gmm_run(nullptr,trial,argv[1],6,0);if(rc)return rc;}return 0;}
#endif

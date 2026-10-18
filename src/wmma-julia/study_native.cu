#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <algorithm>
#define main original_wmma_main
#include "../wmma-cuda/main.cu"
#undef main
#include "study_cases.h"
static constexpr int G=32;static constexpr uint32_t SENT=0x4b123456;static constexpr uint16_t HSENT=0x3555;
static void ck(cudaError_t e){CHECK_CUDA_ERROR(e);}
static uint32_t bits(float f){uint32_t a;memcpy(&a,&f,4);return a;}
static float fl(uint32_t a){float f;memcpy(&f,&a,4);return f;}
template<class T>static std::vector<T>read(const char*name){FILE*f=fopen(name,"rb");if(!f)abort();fseek(f,0,SEEK_END);size_t n=ftell(f);rewind(f);std::vector<T>a(n/sizeof(T));if(fread(a.data(),1,n,f)!=n)abort();fclose(f);return a;}
struct State{Case c;std::vector<half>a,b;std::vector<float>cc,d;half*A,*B;float*C,*D;};
static void launch(State&s){auto c=s.c;if(c.impl==0)gemm_impl0<<<dim3(c.m/16,c.n/16),32>>>(c.m,c.n,c.k,s.A,s.B,s.C,s.D,c.k,c.k,c.n,c.n,.5f,2.f);else gemm_impl1<<<dim3(c.m/64,c.n/64),dim3(128,4)>>>(c.m,c.n,c.k,s.A,s.B,s.C,s.D,c.k,c.k,c.n,c.n,.5f,2.f);ck(cudaGetLastError());}
static void geometry_diagnostic(const char*outdir){
 State s;s.c={1,96,80,16,0,0,1};int na=96*16,nb=80*16,nc=96*80;s.a.resize(na);s.b.resize(nb);s.cc.resize(nc);s.d.assign(nc,fl(0x7fc12345));fill(s.a.data(),96,16);fill(s.b.data(),16,80);fill(s.cc.data(),96,80);
 ck(cudaMalloc(&s.A,na*2));ck(cudaMalloc(&s.B,nb*2));ck(cudaMalloc(&s.C,nc*4));ck(cudaMalloc(&s.D,nc*4));ck(cudaMemcpy(s.A,s.a.data(),na*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(s.B,s.b.data(),nb*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(s.C,s.cc.data(),nc*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(s.D,s.d.data(),nc*4,cudaMemcpyHostToDevice));launch(s);ck(cudaDeviceSynchronize());ck(cudaMemcpy(s.d.data(),s.D,nc*4,cudaMemcpyDeviceToHost));int finite=0;for(auto v:s.d)finite+=std::isfinite(v);if(finite!=4096)abort();
 const char*base="/home/imo/HeCBench_julia/.porting-state/portable_julia/diagnostics/wmma";std::string path=std::string(base)+"/impl1_96x80_output_f32.bin";FILE*f=fopen(path.c_str(),"wb");if(!f)abort();fwrite(s.d.data(),4,nc,f);fclose(f);f=fopen((std::string(base)+"/native_geometry.json").c_str(),"w");fprintf(f,"{\"original_kernel_unchanged\":true,\"source_host_bounds_admits\":true,\"impl\":1,\"m\":96,\"n\":80,\"k\":16,\"grid\":[1,1],\"block\":[128,4],\"finite_output_values\":%d,\"unwritten_NaNs\":%d,\"total_values\":%d,\"in_primary_measured_scope\":false,\"diagnostic_only\":true,\"cause\":\"sourceboundschecks16multiples,but64tilefloorgrid leavesrows/columnsunwritten\",\"output_path\":\"%s\"}\n",finite,nc-finite,nc,path.c_str());fclose(f);cudaFree(s.A);cudaFree(s.B);cudaFree(s.C);cudaFree(s.D);
}
int main(){const char*outdir=getenv("STUDY_OUTPUT_DIR");if(!outdir)outdir=".";geometry_diagnostic(outdir);auto input=read<uint32_t>("input_u32.bin");auto oracle=read<float>("oracle_f32.bin");std::vector<State>states;
 for(auto c:cases){State s;s.c=c;int na=c.m*c.k,nb=c.n*c.k,nc=c.m*c.n;s.a.resize(na+G);s.b.resize(nb+G);s.cc.resize(nc+G);s.d.assign(nc+G,fl(SENT));for(int i=0;i<na+G;i++){uint16_t v=i<na?input[c.io+i]:HSENT;memcpy(&s.a[i],&v,2);}for(int i=0;i<nb+G;i++){uint16_t v=i<nb?input[c.io+na+i]:HSENT;memcpy(&s.b[i],&v,2);}for(int i=0;i<nc+G;i++)s.cc[i]=fl(i<nc?input[c.io+na+nb+i]:SENT);
  if(states.size()<8){std::vector<half>a(na),b(nb);std::vector<float>cc(nc);fill(a.data(),c.m,c.k);fill(b.data(),c.k,c.n);fill(cc.data(),c.m,c.n);if(memcmp(a.data(),s.a.data(),na*2)||memcmp(b.data(),s.b.data(),nb*2)||memcmp(cc.data(),s.cc.data(),nc*4))abort();}
  ck(cudaMalloc(&s.A,(na+G)*2));ck(cudaMalloc(&s.B,(nb+G)*2));ck(cudaMalloc(&s.C,(nc+G)*4));ck(cudaMalloc(&s.D,(nc+G)*4));ck(cudaMemcpy(s.A,s.a.data(),(na+G)*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(s.B,s.b.data(),(nb+G)*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(s.C,s.cc.data(),(nc+G)*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(s.D,s.d.data(),(nc+G)*4,cudaMemcpyHostToDevice));for(int j=0;j<30;j++)launch(s);ck(cudaDeviceSynchronize());states.push_back(s);
 }
 for(int trial=1;trial<=5;trial++){std::vector<uint32_t>prefix,values,guards;std::vector<double>times;bool ok=true;long mismatches=0;
  for(auto&s:states){auto c=s.c;int na=c.m*c.k,nb=c.n*c.k,nc=c.m*c.n;std::vector<half>aa(na+G),bb(nb+G);std::vector<float>cc(nc+G);ck(cudaMemcpy(aa.data(),s.A,(na+G)*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(bb.data(),s.B,(nb+G)*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(cc.data(),s.C,(nc+G)*4,cudaMemcpyDeviceToHost));for(int i=0;i<na;i++){uint16_t v;memcpy(&v,&aa[i],2);prefix.push_back(v);ok &=v==input[c.io+i];}for(int i=0;i<nb;i++){uint16_t v;memcpy(&v,&bb[i],2);prefix.push_back(v);ok &=v==input[c.io+na+i];}for(int i=0;i<nc;i++){prefix.push_back(bits(cc[i]));ok &=bits(cc[i])==input[c.io+na+nb+i];}for(uint32_t v:{uint32_t(c.impl),uint32_t(c.m),uint32_t(c.n),uint32_t(c.k),uint32_t(c.repeats),uint32_t(c.impl==0?32:128),uint32_t(c.impl==0?1:4),16u,16u,16u})prefix.push_back(v);
   std::fill(s.d.begin(),s.d.end(),fl(SENT));std::fill(s.d.begin(),s.d.begin()+nc,fl(0x7fc12345));ck(cudaMemcpy(s.D,s.d.data(),(nc+G)*4,cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int j=0;j<c.repeats;j++)launch(s);ck(cudaDeviceSynchronize());times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());ck(cudaMemcpy(s.d.data(),s.D,(nc+G)*4,cudaMemcpyDeviceToHost));
   for(int i=0;i<nc;i++){uint32_t v=bits(s.d[i]);values.push_back(v);bool match=std::isfinite(s.d[i])&&v==bits(oracle[c.oo+i]);mismatches+=!match;ok &=match;}
   ck(cudaMemcpy(aa.data(),s.A,(na+G)*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(bb.data(),s.B,(nb+G)*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(cc.data(),s.C,(nc+G)*4,cudaMemcpyDeviceToHost));ok &=!memcmp(aa.data(),s.a.data(),(na+G)*2)&&!memcmp(bb.data(),s.b.data(),(nb+G)*2)&&!memcmp(cc.data(),s.cc.data(),(nc+G)*4);
   for(int i=na;i<na+G;i++){uint16_t v;memcpy(&v,&aa[i],2);guards.push_back(v);ok &=v==HSENT;}for(int i=nb;i<nb+G;i++){uint16_t v;memcpy(&v,&bb[i],2);guards.push_back(v);ok &=v==HSENT;}for(int i=nc;i<nc+G;i++){guards.push_back(bits(cc[i]));ok &=bits(cc[i])==SENT;}for(int i=nc;i<nc+G;i++){guards.push_back(bits(s.d[i]));ok &=bits(s.d[i])==SENT;}
  }
  std::vector<uint32_t>art=prefix;art.insert(art.end(),values.begin(),values.end());art.insert(art.end(),guards.begin(),guards.end());std::string path=std::string(outdir)+"/trial"+std::to_string(trial)+".bin";FILE*f=fopen(path.c_str(),"wb");fwrite(art.data(),4,art.size(),f);fclose(f);double total=0;std::string timesjson;for(auto t:times){total+=t;if(!timesjson.empty())timesjson+=",";char b[64];snprintf(b,sizeof(b),"%.17g",t);timesjson+=b;}
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":null,\"max_rel_error\":null,\"exact_raw_bit_mismatches\":%ld,\"timing_region\":\"sum11cases_R100_sourceWMMA16fragments_FP16_F32_devicecomplete\",\"jit_excluded\":true,\"internal_repeats\":100,\"case_count\":11,\"component_time_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,total,ok?"true":"false",values.size()+guards.size(),mismatches,timesjson.c_str(),path.c_str(),art.size());fflush(stdout);if(!ok)return 1;
 }
 for(auto&s:states){cudaFree(s.A);cudaFree(s.B);cudaFree(s.C);cudaFree(s.D);}return 0;
}

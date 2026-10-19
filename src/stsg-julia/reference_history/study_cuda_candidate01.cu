#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "native_reference_repaired_Filter.cu"
#ifndef FIXTURE
#define FIXTURE "study_fixture.bin"
#endif
#ifndef ORACLE
#define ORACLE "study_oracle.bin"
#endif
static constexpr int G=32;
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}
std::vector<double>readf(const char*path){std::ifstream f(path,std::ios::binary|std::ios::ate);assert(f);size_t n=f.tellg();assert(n%8==0);std::vector<double>a(n/8);f.seekg(0);f.read((char*)a.data(),n);assert(f);return a;}
template<class T>struct Buf{std::vector<T>initial;T*base=nullptr;int n;Buf(int N,const std::vector<double>*source=nullptr,size_t off=0):n(N){initial.assign(N+2*G,T(-37));if constexpr(std::is_same_v<T,unsigned char>)std::fill(initial.begin(),initial.end(),173);for(int i=0;i<N;i++)initial[G+i]=source?T((*source)[off+i]):T(0);ck(cudaMalloc(&base,initial.size()*sizeof(T)));ck(cudaMemcpy(base,initial.data(),initial.size()*sizeof(T),cudaMemcpyHostToDevice));}~Buf(){ck(cudaFree(base));}T*p(){return base+G;}std::vector<T>read(){std::vector<T>a(initial.size());ck(cudaMemcpy(a.data(),base,a.size()*sizeof(T),cudaMemcpyDeviceToHost));return a;}void append(std::vector<double>&a,bool full=false){auto v=read();a.insert(a.end(),v.begin()+(full?0:G),v.end()-(full?0:G));}bool guards(std::vector<double>&extra){auto a=read();extra.insert(extra.end(),a.begin(),a.begin()+G);extra.insert(extra.end(),a.end()-G,a.end());return std::equal(a.begin(),a.begin()+G,initial.begin())&&std::equal(a.end()-G,a.end(),initial.end()-G);}};
int main(){auto fixture=readf(FIXTURE),oracle=readf(ORACLE);for(int trial=0;trial<=5;trial++){size_t off=1;std::vector<double>prefix{fixture[0]},got,extra;bool ok=true;double secs=0,inderror=0;
for(int ci=0;ci<int(fixture[0]);ci++){std::vector<double>head(fixture.begin()+off,fixture.begin()+off+8);off+=8;int X=head[0],Y=head[1],B=head[2],Z=head[3],w=head[4],N=X*Y,Q=N*B*Z,W=(2*w+1)*(2*w+1);size_t roff=off,qoff=off+Q;off+=2*Q;std::vector<double>snap;
for(bool observer:{true,false}){Buf<short>raw(Q,&fixture,roff);Buf<unsigned char>rawqa(Q,&fixture,qoff);Buf<float>nd(Q),qa(Q),ref(N*B),res(N*W*4),r3(N*W*3),out(Q),vin(N*B);Buf<int>res1(N*B),index(N*W*3);
if(!observer){prefix.insert(prefix.end(),head.begin(),head.end());raw.append(prefix,true);rawqa.append(prefix,true);for(auto*v:{&nd,&qa,&ref,&res,&r3,&out,&vin})v->append(prefix,true);res1.append(prefix,true);index.append(prefix,true);ok&=raw.read()==raw.initial&&rawqa.read()==rawqa.initial;}
dim3 block(16,16),grid((X+15)/16,(Y+15)/16);ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();
Short_to_Float<<<grid,block>>>(raw.p(),rawqa.p(),X,Y,B,Z,nd.p(),qa.p());
if(observer){ck(cudaDeviceSynchronize());nd.append(snap);qa.append(snap);auto a=nd.read(),q=qa.read();for(int i=0;i<Q;i++){float want=float(raw.initial[G+i])/10000.f;float wantq=(want<-.2f||want>1.f||rawqa.initial[G+i]>3)?-1.f:float(rawqa.initial[G+i]);inderror=std::max(inderror,std::abs(double(a[G+i])-want));ok&=std::abs(double(a[G+i])-want)<=2e-7&&q[G+i]==wantq;}}
Generate_NDVI_reference<<<grid,block>>>(float(head[5]),2,nd.p(),qa.p(),X,Y,B,Z,ref.p(),r3.p(),res1.p());
if(observer){ck(cudaDeviceSynchronize());for(auto*v:{&nd,&qa,&ref,&r3})v->append(snap);res1.append(snap);}
ck(cudaMemset(r3.p(),0,N*W*3*sizeof(float)));
Compute_d_res<<<grid,block>>>(nd.p(),qa.p(),ref.p(),0,Y,0,0,X,Y,B,Z,w,res.p());if(observer){ck(cudaDeviceSynchronize());res.append(snap);}
STSG_filter<<<grid,block>>>(nd.p(),qa.p(),ref.p(),0,Y,0,0,X,Y,B,Z,w,float(head[6]),int(head[7]),out.p(),vin.p(),res.p(),r3.p(),index.p());ck(cudaGetLastError());ck(cudaDeviceSynchronize());
if(!observer){secs+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();got.insert(got.end(),snap.begin(),snap.end());for(auto*v:{&nd,&qa,&ref,&res,&r3,&out,&vin})v->append(got);res1.append(got);index.append(got);if(X==7){auto a=out.read();for(int i=0;i<Q;i++)ok&=a[G+i]==0.f;}
ok&=raw.guards(extra);ok&=rawqa.guards(extra);for(auto*v:{&nd,&qa,&ref,&res,&r3,&out,&vin})ok&=v->guards(extra);ok&=res1.guards(extra);ok&=index.guards(extra);ok&=raw.read()==raw.initial&&rawqa.read()==rawqa.initial;raw.append(extra,true);rawqa.append(extra,true);
}
}}
assert(off==fixture.size());assert(got.size()==oracle.size());double err=0,rel=0;for(size_t i=0;i<got.size();i++){double e=std::abs(got[i]-oracle[i]);err=std::max(err,e);rel=std::max(rel,e/std::max(std::abs(oracle[i]),1e-30));ok&=std::isfinite(got[i])&&e<=2e-5+2e-5*std::abs(oracle[i]);}
if(trial==0){if(!ok){fprintf(stderr,"WarmupOracleMismatch maxabs=%.17g independent=%.17g\n",err,inderror);return 3;}continue;}
prefix.insert(prefix.end(),got.begin(),got.end());prefix.insert(prefix.end(),extra.begin(),extra.end());const char*dir=getenv("STUDY_OUTPUT_DIR");std::string path=std::string(dir?dir:".")+"/output_trial"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)prefix.data(),prefix.size()*8);assert(f);f.close();
printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"independent_conversion_max_abs_error\":%.17g,\"timing_region\":\"sum3cases_full4stage_pipeline_source_scratch_memset_and_context_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"native_reference_repaired\":true,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,secs,ok?"true":"false",prefix.size(),err,rel,inderror,path.c_str(),prefix.size());fflush(stdout);if(!ok)return 4;
}return 0;}

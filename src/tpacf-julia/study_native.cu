#include <cuda_runtime.h>
#include <filesystem>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <cstring>
#include "study_observer.h"
#include "study_cases.h"
#include "corrected_native/kernel.h"
static double measured=0,paused=0;static bool guarded=false;
struct Buffer{void*p;unsigned char*raw;size_t n;};static std::vector<Buffer>buffers;
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static cudaError_t study_alloc(void**out,size_t n){unsigned char*raw;auto e=cudaMalloc(&raw,n+512);if(e!=cudaSuccess)return e;ck(cudaMemset(raw,0xa5,n+512));ck(cudaMemset(raw+256,0,n));*out=raw+256;buffers.push_back({*out,raw,n});return cudaSuccess;}
static cudaError_t study_free(void*p){if(!p)return cudaSuccess;if(!guarded){for(size_t i=0;i<buffers.size();i++){auto&b=buffers[i];unsigned char g[512];ck(cudaMemcpy(g,b.raw,256,cudaMemcpyDeviceToHost));ck(cudaMemcpy(g+256,b.raw+256+b.n,256,cudaMemcpyDeviceToHost));tp_guard(i,g);}guarded=true;}
for(auto&b:buffers)if(b.p==p){auto e=cudaFree(b.raw);b.raw=nullptr;return e;}throw std::runtime_error("unknown source GPU allocation");}
static cudaError_t study_copy(void*dst,const void*src,size_t n,cudaMemcpyKind kind,cudaStream_t stream){auto e=cudaMemcpyAsync(dst,src,n,kind,stream);if(e!=cudaSuccess)return e;if(kind==cudaMemcpyHostToDevice){ck(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();std::vector<unsigned char>actual(n);ck(cudaMemcpy(actual.data(),dst,n,cudaMemcpyDeviceToHost));tp_input(actual.data(),n);paused+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();}return cudaSuccess;}
void tileComputeSymm(int,int,int,int*,int,long long*,cudaStream_t&);
void tileCompute(int,int,int,int*,int,long long*,cudaStream_t&);
#define cudaMalloc study_alloc
#define cudaFree study_free
#define cudaMemcpyAsync study_copy
#include "native_instrumentation.cu"
#undef cudaMalloc
#undef cudaFree
#undef cudaMemcpyAsync
void tileComputeSymm(int type,int size,int njk,int*jk,int bins,long long*h,cudaStream_t&s){auto start=std::chrono::steady_clock::now();auto before=paused;source_tileComputeSymm(type,size,njk,jk,bins,h,s);measured+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()-(paused-before);}
void tileCompute(int dn,int rn,int njk,int*jk,int bins,long long*h,cudaStream_t&s){auto start=std::chrono::steady_clock::now();auto before=paused;source_tileCompute(dn,rn,njk,jk,bins,h,s);measured+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()-(paused-before);}
#define main original_tpacf_main
#include "corrected_native/main.cu"
#undef main
int main(){try{auto root=std::filesystem::path(__FILE__).parent_path();auto captures=root/"../../.porting-state/portable_julia/tpacf_primary";tp_start("cuda",root.c_str(),captures.c_str());for(int trial=0;trial<=5;trial++){measured=paused=0;tp_begin(trial);for(int ci=0;ci<3;ci++){const auto&c=CASES[ci];buffers.clear();guarded=false;int*jks=nullptr;calculatejkSizes(c.njk,c.n,&jks);int nb=0;double*b=init_bins(5,1.f,10000.f,1,&nb);writeBoundaries(b);double actual[63];ck(cudaDeviceSynchronize());ck(cudaMemcpyFromSymbol(actual,binbounds,504));int zero=0;for(int k=30;k>=0;k--)if(0.<b[k]){zero=k+1;break;}tp_case(ci,c.n,c.nr,c.njk,jks,zero,actual);
auto stem=std::filesystem::current_path()/("case"+std::to_string(ci)+"_random");for(int r=1;r<=c.nr;r++){auto link=stem.string()+"."+std::to_string(r);std::filesystem::remove(link);std::filesystem::create_symlink(root/"data"/("case"+std::to_string(ci)+"_catalog"+std::to_string(r)+".txt"),link);}auto data=(root/"data"/("case"+std::to_string(ci)+"_catalog0.txt")).string();auto random=stem.string();long long *DDs,*DRs,*RRs;doComputeGPU(data.data(),random.data(),c.nr,c.n,c.n,c.njk,jks,64,zero,&DDs,&DRs,&RRs);options a{};a.njk=c.njk;a.random_count=c.nr;long long **DD,**DR,**RR;compileHistograms(DDs,DRs,RRs,&DD,&DR,&RR,&a);auto name="case"+std::to_string(ci)+"_trial"+std::to_string(trial)+".txt";write_results(DD,RR,DR,c.njk,nb,5,1.f,c.nr,name.data());std::vector<int64_t>raw,compiled;raw.insert(raw.end(),DDs,DDs+c.njk*64);raw.insert(raw.end(),DRs,DRs+c.nr*c.njk*64);raw.insert(raw.end(),RRs,RRs+c.nr*64);for(int k=0;k<=c.njk;k++)compiled.insert(compiled.end(),DD[k],DD[k]+64);for(int k=0;k<=c.njk;k++)compiled.insert(compiled.end(),DR[k],DR[k]+64);compiled.insert(compiled.end(),RR[0],RR[0]+64);tp_output(raw.data(),compiled.data(),name.c_str());for(int k=0;k<=c.njk;k++){free(DD[k]);free(DR[k]);}free(DD);free(DR);free(RR[0]);free(RR);free(DDs);free(DRs);free(RRs);free(jks);free(b);}
if(trial)tp_write(trial,measured);}tp_finish();return 0;}catch(const std::exception&e){fprintf(stderr,"TPACF_NATIVE %s\n",e.what());return 2;}}

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include "cpu_bridge.h"
#include "../pad-cuda/kernel.h"
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(10);}}while(0)
struct Case{int m,n,p,size,nt,cut,cap,offset;float alpha;void*ctx;double*dev;int*flags;std::vector<uint64_t>gold;};
uint64_t bits(double x){uint64_t u;memcpy(&u,&x,8);return u;}
int main(){FILE*f=fopen("/home/imo/HeCBench_julia/src/pad-julia/fixture.bin","rb");if(!f)return 4;std::vector<uint64_t>prefix(163948);if(fread(prefix.data(),8,prefix.size(),f)!=prefix.size())return 5;fclose(f);std::vector<Case>cases;
for(int shape=0;shape<2;shape++)for(float alpha:{0.f,.1f,.25f,.5f,.75f,1.f}){Case c;c.m=256;c.n=shape?503:127;c.p=shape?9:1;c.size=c.m*(c.n+c.p);c.nt=c.size/8192;c.cut=int(c.nt*alpha);c.cap=(c.nt-c.cut)*8192;c.offset=shape?32768:0;c.alpha=alpha;c.ctx=pad_create(c.m,c.n,c.p,alpha,prefix.data()+c.offset);CHECK(cudaMalloc(&c.dev,(c.cap+16)*8));CHECK(cudaMalloc(&c.flags,(c.nt+17)*4));std::vector<double>guard(c.cap+16,-12345.);CHECK(cudaMemcpy(c.dev,guard.data(),guard.size()*8,cudaMemcpyHostToDevice));std::vector<int>fg(c.nt+17,-991827);CHECK(cudaMemcpy(c.flags,fg.data(),fg.size()*4,cudaMemcpyHostToDevice));c.gold.assign(c.size+16,bits(-12345.));for(int row=0;row<c.m;row++)for(int col=0;col<c.n+c.p;col++)c.gold[8+row*(c.n+c.p)+col]=col<c.n?prefix[c.offset+row*c.n+col]:bits(0.);cases.push_back(c);}
auto step=[](Case&c){pad_reset(c.ctx);CHECK(cudaMemcpy(c.dev+8,pad_data(c.ctx),c.cap*8,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(c.flags+8,pad_flags(c.ctx),(c.nt+1)*4,cudaMemcpyHostToDevice));call_Padding_kernel(8,256,c.n,c.m,c.p,c.nt,c.alpha,c.dev+8,c.dev+8,c.flags+8);CHECK(cudaGetLastError());pad_start(c.ctx);CHECK(cudaDeviceSynchronize());pad_join(c.ctx);};
for(auto&c:cases)step(c);
for(int trial=1;trial<=5;trial++){double seconds=0;bool ok=true;std::vector<uint64_t>art=prefix;for(auto&c:cases){auto t=std::chrono::steady_clock::now();for(int rep=0;rep<3;rep++)step(c);seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/3;CHECK(cudaMemcpy(pad_data(c.ctx),c.dev+8,c.cap*8,cudaMemcpyDeviceToHost));for(int i=0;i<c.size+16;i++){auto w=bits(pad_all_data(c.ctx)[i]);ok&=w==c.gold[i];art.push_back(w);}std::vector<double>dg(c.cap+16);CHECK(cudaMemcpy(dg.data(),c.dev,dg.size()*8,cudaMemcpyDeviceToHost));for(int k=0;k<16;k++){int idx=k<8?k:c.cap+k;ok&=bits(dg[idx])==bits(-12345.);art.push_back(bits(dg[idx]));}for(int i=0;i<=c.nt;i++){int want=i==0||i==c.cut?1:0;if(i>=1&&i<=c.cut)want++;int got=pad_flags(c.ctx)[i];ok&=got==want;art.push_back(uint32_t(got));}std::vector<int>gf(c.nt+17);CHECK(cudaMemcpy(gf.data(),c.flags,gf.size()*4,cudaMemcpyDeviceToHost));for(int i=0;i<c.nt+17;i++){int k=i-8;int want=-991827;if(k>=0&&k<=c.nt)want=(k==0||k==c.cut||k>c.cut)?1:0;ok&=gf[i]==want;art.push_back(uint32_t(gf[i]));}}
std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(trial)+".bin";FILE*out=fopen(path.c_str(),"wb");fwrite(art.data(),8,art.size(),out);fclose(out);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":0,\"max_rel_error\":0,\"timing_region\":\"sum12_static_hybrid_reset_copy_launch_cpuworkers_join_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"uint64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",art.size()-prefix.size(),path.c_str(),art.size());}
for(auto&c:cases){pad_destroy(c.ctx);cudaFree(c.dev);cudaFree(c.flags);}}

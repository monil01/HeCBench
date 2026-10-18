#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../lut-gemm-cuda/nQWeight_fp16.h"
#include "../lut-gemm-cuda/mv_fp16.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <string>
#include <cstring>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
std::vector<double> rd(const char*n){FILE*f=fopen(n,"rb");if(!f)abort();fseek(f,0,SEEK_END);size_t z=ftell(f)/8;rewind(f);std::vector<double>a(z);fread(a.data(),8,z,f);fclose(f);return a;}
struct Case{int M,K,nb,G,bi,T;std::vector<unsigned>hw;std::vector<__half>ha,hb,hx;unsigned*wo;__half*ao,*bo,*xo,*yo;};
void launch(Case&c){CK(cudaMemset(c.yo+8,0,c.M*2));dim3 g((c.M+2047)/2048,c.K/c.T);if(c.bi)_nqmv_bias<32><<<g,256>>>(c.wo+8,c.ao+8,c.bo+8,c.xo+8,c.yo+8,c.M,c.K,c.nb,2048,c.G);else _nqmv<64><<<g,256>>>(c.wo+8,c.ao+8,c.xo+8,c.yo+8,c.M,c.K,c.nb,2048,c.G);CK(cudaGetLastError());}
int main(){auto pre=rd(FIXTURE),gold=rd(ORACLE);size_t pos=1;std::vector<Case>cs;
for(int ci=0;ci<(int)pre[0];ci++){Case c;c.M=pre[pos++];c.K=pre[pos++];c.nb=pre[pos++];c.G=pre[pos++];c.bi=pre[pos++];c.T=pre[pos++];int ns[]={c.K/32*c.nb*c.M,c.K/c.G*c.nb*c.M,c.bi?c.K/c.G*c.M:0,c.K};c.hw.assign(ns[0]+16,1515870810);for(int j=0;j<ns[0];j++)c.hw[j+8]=pre[pos++];CK(cudaMalloc(&c.wo,c.hw.size()*4));CK(cudaMemcpy(c.wo,c.hw.data(),c.hw.size()*4,cudaMemcpyHostToDevice));std::vector<__half>*hs[]={&c.ha,&c.hb,&c.hx};__half**ds[]={&c.ao,&c.bo,&c.xo};for(int i=0;i<3;i++){auto&h=*hs[i];h.assign(ns[i+1]+16,__float2half(-123));for(int j=0;j<ns[i+1];j++)h[j+8]=__double2half(pre[pos++]);CK(cudaMalloc(ds[i],h.size()*2));CK(cudaMemcpy(*ds[i],h.data(),h.size()*2,cudaMemcpyHostToDevice));}CK(cudaMalloc(&c.yo,(c.M+16)*2));std::vector<__half>h(c.M+16,__float2half(-123));CK(cudaMemcpy(c.yo,h.data(),h.size()*2,cudaMemcpyHostToDevice));cs.push_back(c);}
if(pos!=pre.size())abort();for(auto&c:cs){launch(c);CK(cudaDeviceSynchronize());}
for(int tr=1;tr<=5;tr++){auto art=pre;size_t go=0;bool ok=true;double sum=0;std::vector<double>times;
auto append=[&](const std::vector<double>&v){for(double x:v){if(go>=gold.size()||!std::isfinite(x)||x!=gold[go])ok=false;go++;art.push_back(x);}};
for(auto&c:cs){std::vector<__half>can(c.M,__float2half(.5));CK(cudaMemcpy(c.yo+8,can.data(),c.M*2,cudaMemcpyHostToDevice));CK(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();launch(c);CK(cudaDeviceSynchronize());double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();times.push_back(dt);sum+=dt;std::vector<__half>h(c.M+16);CK(cudaMemcpy(h.data(),c.yo,h.size()*2,cudaMemcpyDeviceToHost));std::vector<double>v;for(auto x:h)v.push_back(__half2float(x));append(v);std::vector<unsigned>w(c.hw.size());CK(cudaMemcpy(w.data(),c.wo,w.size()*4,cudaMemcpyDeviceToHost));v.assign(w.begin(),w.end());ok&=w==c.hw;append(v);std::vector<__half>*hs[]={&c.ha,&c.hb,&c.hx};__half*ds[]={c.ao,c.bo,c.xo};for(int i=0;i<3;i++){h.resize(hs[i]->size());CK(cudaMemcpy(h.data(),ds[i],h.size()*2,cudaMemcpyDeviceToHost));v.clear();for(size_t j=0;j<h.size();j++){v.push_back(__half2float(h[j]));ok&=memcmp(&h[j],&(*hs[i])[j],sizeof(__half))==0;}append(v);}}
if(go!=gold.size())abort();std::string file=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/output_trial"+std::to_string(tr)+".bin";FILE*f=fopen(file.c_str(),"wb");fwrite(art.data(),8,art.size(),f);fclose(f);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":0.0,\"max_rel_error\":0.0,\"timing_region\":\"sum7_clear_plus_packed_half_LUT_GEMV\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"},\"path_times_seconds\":[",tr,sum,ok?"true":"false",go,file.c_str(),art.size());for(size_t i=0;i<times.size();i++)printf("%s%.17g",i?",":"",times[i]);printf("]}\n");}
}

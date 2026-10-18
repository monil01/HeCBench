#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#define Tmax 768
#define BF 8
#define BB 2
#include "../dwconv1d-cuda/src/timex.cu"
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
template<class T>std::vector<T>load(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);size_t n=f.tellg();f.seekg(0);std::vector<T>v(n/sizeof(T));f.read((char*)v.data(),n);return v;}
constexpr float S=-8192.25f;
struct Buf{std::vector<float>h;float*d;Buf(const std::vector<float>&v){h.assign(v.size()+8,S);std::copy(v.begin(),v.end(),h.begin()+4);ck(cudaMalloc(&d,h.size()*4));ck(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice));}float*ptr(){return d+4;}std::vector<float>get(){std::vector<float>r(h.size());ck(cudaMemcpy(r.data(),d,r.size()*4,cudaMemcpyDeviceToHost));return r;}};
struct Case{int b,c,t;float eps;Buf*w,*k,*g,*x,*gw,*gk;};
int main(){auto f=load<float>(DW_FIXTURE);auto oracle=load<double>(DW_ORACLE);std::vector<Case>cs;size_t i=0;while(i<f.size()){int b=f[i],c=f[i+1],t=f[i+2];float eps=f[i+4];i+=5;auto take=[&](size_t n){std::vector<float>r(f.begin()+i,f.begin()+i+n);i+=n;return r;};auto*w=new Buf(take(c*t));auto*k=new Buf(take(b*c*t));auto*g=new Buf(take(b*c*t));auto*x=new Buf(std::vector<float>(b*c*t,S));auto*gw=new Buf(std::vector<float>(b*c*t,S));auto*gk=new Buf(std::vector<float>(b*c*t,S));cs.push_back({b,c,t,eps,w,k,g,x,gw,gk});}
 auto run=[](Case&c){cuda_forward(c.w->ptr(),c.k->ptr(),c.x->ptr(),c.eps,c.b,c.c,c.t);cuda_backward(c.w->ptr(),c.k->ptr(),c.g->ptr(),c.gw->ptr(),c.gk->ptr(),c.b,c.c,c.t);ck(cudaDeviceSynchronize());};for(auto&c:cs)run(c);
 for(int trial=1;trial<=5;trial++){auto out=f;bool ok=true;double tm=0,err=0;size_t oi=0;for(auto&c:cs){ck(cudaDeviceSynchronize());auto st=std::chrono::steady_clock::now();run(c);tm+=std::chrono::duration<double>(std::chrono::steady_clock::now()-st).count();for(auto*b:{c.w,c.k,c.g}){auto r=b->get();ok&=r==b->h;for(int j=0;j<4;j++)out.push_back(r[j]);for(size_t j=r.size()-4;j<r.size();j++)out.push_back(r[j]);}for(auto*b:{c.x,c.gw,c.gk}){auto r=b->get();for(size_t j=4;j<r.size()-4;j++){double e=fabs(double(r[j])-oracle[oi]);err=fmax(err,e);ok&=e<=2e-5+2e-5*fabs(oracle[oi++]);out.push_back(r[j]);}for(int j=0;j<4;j++){ok&=r[j]==S;out.push_back(r[j]);}for(size_t j=r.size()-4;j<r.size();j++){ok&=r[j]==S;out.push_back(r[j]);}}}std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream z(path,std::ios::binary);z.write((char*)out.data(),out.size()*4);z.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"timing_region\":\"sum3_forward_backward_preallocated_kernel_pipeline_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,tm,ok?"true":"false",out.size(),err,path.c_str(),out.size());if(!ok)return 3;}}

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>
#include "../tsa-cuda/kernels.h"
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
std::vector<double> load(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)exit(3);size_t n=f.tellg();f.seekg(0);std::vector<double>v(n/8);f.read((char*)v.data(),n);return v;}
template<class T>void append(std::vector<double>&v,T*p,int n){std::vector<T>h(n);CK(cudaMemcpy(h.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));for(auto x:h)v.push_back(x);}
template<class T>struct Case{int w,h,steps;T a,b;std::vector<T>real,imag;T*r[2],*im[2];
Case(const std::vector<double>&input,size_t&at,int W,int H,int S):w(W),h(H),steps(S){int n=w*h;a=input[at++];b=input[at++];real.resize(n);imag.resize(n);for(auto&x:real)x=input[at++];for(auto&x:imag)x=input[at++];for(int i=0;i<2;i++){CK(cudaMalloc(r+i,n*sizeof(T)));CK(cudaMalloc(im+i,n*sizeof(T)));}reset();}
void reset(){int n=w*h;CK(cudaMemcpy(r[0],real.data(),n*sizeof(T),cudaMemcpyHostToDevice));CK(cudaMemcpy(im[0],imag.data(),n*sizeof(T),cudaMemcpyHostToDevice));CK(cudaMemset(r[1],0,n*sizeof(T)));CK(cudaMemset(im[1],0,n*sizeof(T)));}
void run(){constexpr int BY=sizeof(T)==8?32:96;dim3 grid((w+9)/10,(h+BY-9)/(BY-8)),block(16,16);for(int s=0;s<steps;s++){int i=s%2;kernel<T,1,16,BY,3,4,16><<<grid,block>>>(a,b,w,h,r[i],im[i],r[1-i],im[1-i]);}CK(cudaGetLastError());}
double trial(std::vector<double>&out){reset();CK(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();run();CK(cudaDeviceSynchronize());double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/steps;int active=steps%2;append(out,r[active],w*h);append(out,im[active],w*h);append(out,r[1-active],w*h);append(out,im[1-active],w*h);return elapsed;}
void prefix(std::vector<double>&v){v.push_back(a);v.push_back(b);append(v,r[0],w*h);append(v,im[0],w*h);}};
int main(){auto input=load("fixture_f64.bin"),expected=load("expected_f64.bin");size_t at=0;
Case<float>f0(input,at,64,96,3),f1(input,at,70,98,4),f2(input,at,34,50,1);
Case<double>d0(input,at,64,96,3),d1(input,at,70,98,4),d2(input,at,34,50,1);
std::vector<double>prefix;f0.prefix(prefix);f1.prefix(prefix);f2.prefix(prefix);d0.prefix(prefix);d1.prefix(prefix);d2.prefix(prefix);if(prefix!=input)return 4;
f0.run();f1.run();f2.run();d0.run();d1.run();d2.run();CK(cudaDeviceSynchronize());bool all=true;
for(int trial=1;trial<=5;trial++){std::vector<double>result;double times[6]={f0.trial(result),f1.trial(result),f2.trial(result),d0.trial(result),d1.trial(result),d2.trial(result)};bool ok=true;double err=0,scaled=0;for(size_t k=0;k<result.size();k++){double e=fabs(result[k]-expected[k]);double tol=k<58816?1e-5:1e-11;ok &=std::isfinite(result[k])&&e<=tol;err=std::max(err,e);scaled=std::max(scaled,e/tol);}all &=ok;auto out=prefix;out.insert(out.end(),result.begin(),result.end());const char*dir=getenv("STUDY_OUTPUT_DIR");char file[4096];snprintf(file,sizeof(file),"%s/tsa_trial%d.bin",dir?dir:".",trial);std::ofstream f(file,std::ios::binary);f.write((char*)out.data(),out.size()*8);f.close();double time=0;for(double t:times)time+=t;
printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"component_seconds\":[%.12g,%.12g,%.12g,%.12g,%.12g,%.12g],\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"max_scaled_absolute_oracle_error\":%.12g,\"internal_repeats\":1,\"jit_excluded\":true,\"timing_region\":\"sum_six_evolution_perstep_means_pairedrotations_devicecompletion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,time,times[0],times[1],times[2],times[3],times[4],times[5],ok?"true":"false",result.size(),err,scaled,file,out.size());fflush(stdout);}
puts(all?"PASS":"FAIL");return all?0:1;}

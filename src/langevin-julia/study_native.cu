
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <chrono>
#define main original_langevin_main
#include "/home/imo/HeCBench_julia/src/langevin-cuda/main.cu"
#undef main
const int N=4096,G=32,R=20;const char*BASE="/home/imo/HeCBench_julia/src/langevin-julia/";
template<class T>std::vector<T>readv(std::string name,int n){std::vector<T>v(n);std::ifstream f(std::string(BASE)+name,std::ios::binary);f.read((char*)v.data(),n*sizeof(T));if(!f)exit(3);return v;}
void launch(int m,float*a,float*o){if(m==0)k0<<<N/256,256>>>(a,o);else if(m==1)k1<<<N/256,256>>>(a,o);else k2<<<N/256,256>>>(a,o);}
struct Case{std::vector<float>input;std::vector<double>oracle;float*a,*out[3];};
int main(){
 std::vector<Case>cases(2);for(int c=0;c<2;c++){auto&v=cases[c];v.input=readv<float>("study_case"+std::to_string(c+1)+"_f32.bin",N);v.oracle=readv<double>("oracle_case"+std::to_string(c+1)+"_f64.bin",5*N);std::vector<float>h=v.input;h.resize(N+G,12345.5f);cudaMalloc(&v.a,(N+G)*4);cudaMemcpy(v.a,h.data(),h.size()*4,cudaMemcpyHostToDevice);h.assign(N+G,12345.5f);for(int m=0;m<3;m++){cudaMalloc(&v.out[m],h.size()*4);cudaMemcpy(v.out[m],h.data(),h.size()*4,cudaMemcpyHostToDevice);launch(m,v.a,v.out[m]);}}
 cudaDeviceSynchronize();
 for(int trial=1;trial<=5;trial++){
  std::vector<double>prefix,values,guards,times;double ma=0,mr=0,normalized=0;bool ok=true;
  for(int c=0;c<2;c++){auto&v=cases[c];
   std::vector<float>in(N+G);cudaMemcpy(in.data(),v.a,in.size()*4,cudaMemcpyDeviceToHost);for(int i=0;i<N;i++){unsigned int bits;memcpy(&bits,&in[i],4);prefix.push_back(bits);}for(int p:{N,R,256,16,c+1,3,0,0})prefix.push_back(p);for(int i=N;i<N+G;i++)guards.push_back(in[i]);
   for(int m=0;m<3;m++){
    cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int r=0;r<R;r++)launch(m,v.a,v.out[m]);cudaDeviceSynchronize();times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/R);
    std::vector<float>o(N+G);cudaMemcpy(o.data(),v.out[m],o.size()*4,cudaMemcpyDeviceToHost);
    for(int i=0;i<N;i++){double diff=fabs(o[i]-v.oracle[m*N+i]);double budget=m==2?2e-7:fmax(2e-6,8*0x1p-23/fabs(v.input[i]));ma=fmax(ma,diff);mr=fmax(mr,diff/fmax(fabs(v.oracle[m*N+i]),1e-30));normalized=fmax(normalized,diff/budget);if(!std::isfinite(o[i])||diff>budget)ok=false;values.push_back(o[i]);}
    for(int i=N;i<N+G;i++)guards.push_back(o[i]);
   }
  }
  for(auto g:guards)if(g!=12345.5)ok=false;
  auto all=prefix;all.insert(all.end(),values.begin(),values.end());all.insert(all.end(),guards.begin(),guards.end());std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/trial"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)all.data(),all.size()*8);f.close();double total=0;for(auto t:times)total+=t;
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":24832,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"max_policy_budget_ratio\":%.17g,\"timing_region\":\"sum6_langevin_k0_k1_k2_kernel_loop_means\",\"jit_excluded\":true,\"internal_repeats\":20,\"component_time_seconds\":[",trial,total,ok?"true":"false",ma,mr,normalized);for(int i=0;i<6;i++)printf("%s%.17g",i?",":"",times[i]);printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",path.c_str(),all.size());fflush(stdout);if(!ok)return 1;
 }
 for(auto&v:cases){cudaFree(v.a);for(auto o:v.out)cudaFree(o);}
}

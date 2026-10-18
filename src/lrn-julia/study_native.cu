
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#ifdef STUDY_ORIGINAL
#include "/home/imo/HeCBench_julia/src/lrn-cuda/kernels.h"
#else
#include "/home/imo/HeCBench_julia/src/lrn-julia/native_corrected/kernels.h"
#endif
const char*BASE="/home/imo/HeCBench_julia/src/lrn-julia/";const int G=32,R=10;const float A=.000122f,B=.75f,K=1.f;
template<class T>std::vector<T>readv(std::string s,int n){std::vector<T>x(n);std::ifstream f(std::string(BASE)+s,std::ios::binary);f.read((char*)x.data(),n*sizeof(T));if(!f)exit(3);return x;}
struct Case{int64_t N,C,D,H,W,count;std::vector<float>src,g;std::vector<double>ref;float*x,*up,*fw,*bw;};
void launch(Case&v,int wg,int mode){int64_t blocks=(v.count+wg-1)/wg;if(mode==0)lrn_fwd_kernel<<<blocks,wg>>>(v.x,v.fw,v.N,v.C,v.D,v.H,v.W,v.C*v.D*v.H*v.W,5,v.count,5,A,B,K);else lrn_bwd_kernel<<<blocks,wg>>>(v.x,v.up,v.bw,v.N,v.C,v.D,v.H,v.W,v.C*v.D*v.H*v.W,5,v.count,5,A,B,K);}
int main(){
 std::vector<Case>v(2);v[0].N=2;v[0].C=7;v[0].D=3;v[0].H=4;v[0].W=5;v[1].N=1;v[1].C=4;v[1].D=1;v[1].H=3;v[1].W=7;
 for(int c=0;c<2;c++){auto&q=v[c];q.count=q.N*q.C*q.D*q.H*q.W;auto in=readv<float>("study_case"+std::to_string(c+1)+"_f32.bin",2*q.count);q.src.assign(in.begin(),in.begin()+q.count);q.g.assign(in.begin()+q.count,in.end());q.ref=readv<double>("oracle_case"+std::to_string(c+1)+"_f64.bin",2*q.count);
  for(int b=0;b<4;b++){std::vector<float>h(q.count+G,12345.5f);if(b<2)std::copy(b==0?q.src.begin():q.g.begin(),b==0?q.src.end():q.g.end(),h.begin());else std::fill(h.begin(),h.begin()+q.count,0.f);float**ptr=b==0?&q.x:b==1?&q.up:b==2?&q.fw:&q.bw;cudaMalloc(ptr,h.size()*4);cudaMemcpy(*ptr,h.data(),h.size()*4,cudaMemcpyHostToDevice);}
  for(int wg:{64,128,256,512})for(int m=0;m<2;m++)launch(q,wg,m);
 }cudaDeviceSynchronize();
 for(int trial=1;trial<=5;trial++){
  std::vector<double>prefix,values,guards,times;double ma=0,mr=0;bool ok=true;int mismatch=0;
  for(int c=0;c<2;c++){auto&q=v[c];
   for(auto p:{q.x,q.up}){std::vector<float>h(q.count+G);cudaMemcpy(h.data(),p,h.size()*4,cudaMemcpyDeviceToHost);for(int i=0;i<q.count;i++){unsigned int bits;memcpy(&bits,&h[i],4);prefix.push_back(bits);}for(int i=q.count;i<q.count+G;i++)guards.push_back(h[i]);}
   for(int64_t p:{q.N,q.C,q.D,q.H,q.W,int64_t(5),int64_t(R),int64_t(c+1),int64_t(5)})prefix.push_back(p);for(float p:{A,B,K}){unsigned int bits;memcpy(&bits,&p,4);prefix.push_back(bits);}
   for(int wg:{64,128,256,512})for(int m=0;m<2;m++){
    cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();for(int i=0;i<R;i++)launch(q,wg,m);cudaDeviceSynchronize();times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/R);
    std::vector<float>h(q.count+G);cudaMemcpy(h.data(),m==0?q.fw:q.bw,h.size()*4,cudaMemcpyDeviceToHost);
    for(int i=0;i<q.count;i++){double d=fabs(h[i]-q.ref[m*q.count+i]);ma=fmax(ma,d);mr=fmax(mr,d/fmax(fabs(q.ref[m*q.count+i]),1e-30));bool pass=std::isfinite(h[i])&&d<=8e-7+4e-6*fabs(q.ref[m*q.count+i]);if(!pass)mismatch++;ok&=pass;values.push_back(h[i]);}
   }
   for(auto p:{q.fw,q.bw}){std::vector<float>h(q.count+G);cudaMemcpy(h.data(),p,h.size()*4,cudaMemcpyDeviceToHost);for(int i=q.count;i<q.count+G;i++)guards.push_back(h[i]);}
  }
  for(auto g:guards)if(g!=12345.5)ok=false;auto all=prefix;all.insert(all.end(),values.begin(),values.end());all.insert(all.end(),guards.begin(),guards.end());std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/trial"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)all.data(),all.size()*8);f.close();double total=0;for(auto x:times)total+=x;
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":7648,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"mismatch_values\":%d,\"timing_region\":\"sum16_lrn_fwd_bwd_four_workgroups_two_cases_kernel_loop_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"component_time_seconds\":[",trial,total,ok?"true":"false",ma,mr,mismatch);for(int i=0;i<16;i++)printf("%s%.17g",i?",":"",times[i]);printf("],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",path.c_str(),all.size());fflush(stdout);if(!ok)return 1;
 }for(auto&q:v)for(auto p:{q.x,q.up,q.fw,q.bw})cudaFree(p);
}

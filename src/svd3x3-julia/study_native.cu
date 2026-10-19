#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <unistd.h>
#include "corrected_kernels.cuh"
#include "study_check.h"
#include "study_params.h"
#define C(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s line%d\n",cudaGetErrorString(e),__LINE__);exit(7);}}while(0)
template<class T>std::vector<T>read_exact(std::string name,size_t n){std::ifstream f(name,std::ios::binary);std::vector<T>a(n);f.read((char*)a.data(),n*sizeof(T));if(!f||f.peek()!=EOF){fprintf(stderr,"bad asset %s\n",name.c_str());exit(2);}return a;}
template<class T>void save(std::string name,const std::vector<T>&a){std::ofstream f(name,std::ios::binary);f.write((const char*)a.data(),a.size()*sizeof(T));if(!f)exit(8);}
void append(std::vector<double>&a,const std::vector<float>&b){for(float x:b)a.push_back(x);}
struct Work{
 int id,n;float *di,*dout;std::vector<float>in,initial,before,init_echo,after,output;std::vector<double>sigma;
 Work(int k):id(k),n(NS[k]){
  auto v=read_exact<float>("case"+std::to_string(k)+"_input_f32.bin",9*n);in.assign(9*n+32,91.25f);std::copy(v.begin(),v.end(),in.begin()+16);initial.assign(21*n+32,91.25f);std::fill(initial.begin()+16,initial.end()-16,-777.f);sigma=read_exact<double>("case"+std::to_string(k)+"_sigma_f64.bin",3*n);C(cudaMalloc(&di,in.size()*4));C(cudaMalloc(&dout,initial.size()*4));reset();
 }
 std::vector<float>read(float*p,size_t n){std::vector<float>x(n);C(cudaMemcpy(x.data(),p,n*4,cudaMemcpyDeviceToHost));return x;}
 void reset(){C(cudaMemcpy(di,in.data(),in.size()*4,cudaMemcpyHostToDevice));C(cudaMemcpy(dout,initial.data(),initial.size()*4,cudaMemcpyHostToDevice));before=read(di,in.size());init_echo=read(dout,initial.size());if(before!=in||init_echo!=initial)exit(5);}
 double run(){C(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();for(int r=0;r<R;r++)svd3_SOA<<<n/256+1,256>>>(di+16,dout+16,n);C(cudaGetLastError());C(cudaDeviceSynchronize());return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/R;}
 void capture(std::vector<double>&prefix){after=read(di,in.size());output=read(dout,initial.size());prefix.insert(prefix.end(),{double(n),double(R),256.,double(n/256+1)});append(prefix,before);append(prefix,init_echo);append(prefix,after);}
 bool tail(std::vector<double>&out,int trial,double*maxima){std::vector<double>scores(3*n);int good=svd_check(after.data()+16,output.data()+16,sigma.data(),n,scores.data(),maxima);good&=after==in;bool guards=true;for(int i=0;i<16;i++)guards&=output[i]==91.25f&&output[21*n+16+i]==91.25f;good&=guards;out.insert(out.end(),scores.begin(),scores.end());for(int i=0;i<16;i++)out.push_back(output[i]);for(int i=0;i<16;i++)out.push_back(output[21*n+16+i]);for(int i=0;i<6;i++)out.push_back(good?1.:0.);
  std::string base=std::string(getenv("STUDY_RAW_DIR"))+"/native_pid"+std::to_string((long long)getpid())+"_trial"+std::to_string(trial)+"_case"+std::to_string(id);save(base+"_input_before.bin",before);save(base+"_output_initial.bin",init_echo);save(base+"_input_after.bin",after);save(base+"_output_final.bin",output);return good;
 }
};
int main(){std::vector<Work*>w;for(int k=0;k<6;k++)w.push_back(new Work(k));for(auto x:w)x->run();for(int trial=1;trial<=5;trial++){
 for(auto x:w)x->reset();double seconds=0.;for(auto x:w)seconds+=x->run();std::vector<double>out;for(auto x:w)x->capture(out);if(out.size()!=PREFIX)return 3;bool good=true;double maxima[5]={};for(auto x:w){double m[5];good&=x->tail(out,trial,m);for(int k=0;k<5;k++)maxima[k]=std::max(maxima[k],m[k]);}if(out.size()!=COUNT)return 4;
 std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/svd_trial"+std::to_string(trial)+".bin";save(path,out);
 printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":null,\"max_rel_error\":null,\"jit_excluded\":true,\"timing_region\":\"sum_sixcase_source_R7_regionmeans_contextcompletion\",\"internal_repeats\":7,\"kernel_launches\":42,\"numerical_agreement_claim\":false,\"raw_output_identity_claim\":false,\"property_maxima\":[%.17g,%.17g,%.17g,%.17g,%.17g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,good?"true":"false",out.size(),maxima[0],maxima[1],maxima[2],maxima[3],maxima[4],path.c_str(),out.size());fflush(stdout);if(!good)return 6;
 }return 0;}

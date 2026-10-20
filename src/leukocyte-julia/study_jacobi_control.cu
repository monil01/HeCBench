#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define LOCAL_WORK_SIZE 256
#define FP_TYPE float
#define FP_CONST(num) num##f
#define PI_FP32 FP_CONST(3.14159)
#define ONE_OVER_PI (FP_CONST(1.0) / PI_FP32)
#define MU FP_CONST(0.5)
#define LAMBDA (FP_CONST(8.0) * MU + FP_CONST(1.0))
#define NEXT_LOWEST_POWER_OF_TWO 256
#define heaviside old_heaviside
#define kernel_IMGVF old_IMGVF
#include "reference_history/corrected_native_three_inherited_repairs/kernel_IMGVF.h"
#undef heaviside
#undef kernel_IMGVF
#define heaviside corrected_heaviside
#define kernel_IMGVF corrected_IMGVF
#include "corrected_native/kernel_IMGVF.h"
#undef heaviside
#undef kernel_IMGVF

#define CK(call) do { auto error=(call); if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); } while(0)
template<class T> struct Buffer {
  T *base=nullptr,*data=nullptr; size_t count;
  Buffer(size_t n): count(n) { CK(cudaMalloc(&base,(n+64)*sizeof(T))); data=base+32; CK(cudaMemset(base,165,(n+64)*sizeof(T))); }
  ~Buffer() { cudaFree(base); }
  void upload(const std::vector<T>& input) { CK(cudaMemcpy(data,input.data(),count*sizeof(T),cudaMemcpyHostToDevice)); }
  std::vector<T> host() const { std::vector<T> result(count+64); CK(cudaMemcpy(result.data(),base,result.size()*sizeof(T),cudaMemcpyDeviceToHost)); return result; }
};
struct Case { int m,n; float e; bool constant; };
std::vector<double> independent_step(const std::vector<float>& input,int m,int n,float epsilon) {
  std::vector<double> output(input.size());
  const int di[]={-1,1,0,0,-1,1,-1,1},dj[]={0,0,-1,1,1,1,-1,-1};
  const double direction[]={-1,1,-1,1,0,2,-2,0};
  const double inverse_e=double(1.0f/epsilon),inverse_pi=double(ONE_OVER_PI);
  const double diffusion=double(MU/LAMBDA),reaction=1.0/double(LAMBDA);
  for(int row=0;row<m;row++) for(int col=0;col<n;col++) {
    double old=input[row*n+col],sum=0;
    for(int k=0;k<8;k++) {
      int r=std::max(0,std::min(m-1,row+di[k]));
      int c=std::max(0,std::min(n-1,col+dj[k]));
      double difference=double(input[r*n+c])-old;
      double h=std::atan(difference*direction[k]*inverse_e)*inverse_pi+.5;
      sum+=h*difference;
    }
    double value=old+diffusion*sum;
    output[row*n+col]=value-reaction*old*(value-old);
  }
  return output;
}
void write_file(const std::string& path,const void* bytes,size_t count) {
  std::ofstream output(path,std::ios::binary); output.write((const char*)bytes,count);
  if(!output) throw std::runtime_error("control artifact write failed");
}
int main(int argc,char**argv) {
 try {
  if(argc!=2) throw std::runtime_error("persistent evidence directory required");
  std::string folder=argv[1];
  std::vector<Case> cases={{7,9,1e-10f,false},{41,41,1e-10f,false},{41,47,1e-10f,false},
      {41,55,1e-10f,false},{41,61,1e-10f,false},{41,81,1e-10f,false},
      {41,61,.125f,false},{41,61,.125f,true}};
  for(int trial=1;trial<=5;trial++) {
    std::vector<double> prefix,values; std::vector<unsigned char> guards;
    double max_error=0; int old_wrong=0; bool pass=true;
    for(size_t ci=0;ci<cases.size();ci++) {
      auto c=cases[ci]; size_t size=c.m*c.n;
      std::vector<float> input(size);
      for(int r=0;r<c.m;r++) for(int col=0;col<c.n;col++)
        input[r*c.n+col]=c.constant ? .375f : float((17*r+13*col)%257)/256.0f;
      auto oracle=independent_step(input,c.m,c.n,c.e);
      Buffer<float> initial(size),old(size),fixed(size);
      Buffer<int> offsets(1),rows(1),columns(1);
      initial.upload(input); old.upload(input); fixed.upload(input);
      offsets.upload({0}); rows.upload({c.m}); columns.upload({c.n});
      CK(cudaDeviceSynchronize());
      old_IMGVF<<<1,256>>>(old.data,initial.data,offsets.data,rows.data,columns.data,1.0f,1.0f,c.e,0.0f,1);
      CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      corrected_IMGVF<<<1,256>>>(fixed.data,initial.data,offsets.data,rows.data,columns.data,1.0f,1.0f,c.e,0.0f,1);
      CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      auto actual_input=initial.host(),actual_old=old.host(),actual_new=fixed.host();
      auto actual_offset=offsets.host(),actual_m=rows.host(),actual_n=columns.host();
      prefix.insert(prefix.end(),{double(actual_m[32]),double(actual_n[32]),double(actual_offset[32]),1.,1.,double(c.e),0.,1.});
      for(size_t i=0;i<size;i++) prefix.push_back(actual_input[32+i]);
      double case_error=0; int case_old_wrong=0;
      for(size_t i=0;i<size;i++) {
        double error=std::abs(double(actual_new[32+i])-oracle[i]);
        case_error=std::max(case_error,error); pass &= std::isfinite(actual_new[32+i]) && error<=5e-6;
        if(std::abs(double(actual_old[32+i])-oracle[i])>5e-6) case_old_wrong++;
        if(c.constant) pass &= actual_new[32+i]==.375f;
        values.push_back(actual_old[32+i]); values.push_back(actual_new[32+i]); values.push_back(oracle[i]);
      }
      max_error=std::max(max_error,case_error); old_wrong+=case_old_wrong;
      fprintf(stderr,"trial%d case%zu m%d n%d e%.9g constant%d corrected_maxabs%.17g old_wrong%d\n",
          trial,ci,c.m,c.n,c.e,c.constant,case_error,case_old_wrong);
      auto capture_guard=[&](const auto& buffer) {
        using T=typename std::decay_t<decltype(buffer)>::value_type;
        const auto* bytes=(const unsigned char*)buffer.data();
        guards.insert(guards.end(),bytes,bytes+32*sizeof(T));
        guards.insert(guards.end(),bytes+(buffer.size()-32)*sizeof(T),bytes+buffer.size()*sizeof(T));
      };
      capture_guard(actual_input); capture_guard(actual_old); capture_guard(actual_new);
      capture_guard(actual_offset); capture_guard(actual_m); capture_guard(actual_n);
    }
    pass &= old_wrong>0 && std::all_of(guards.begin(),guards.end(),[](auto x){return x==165;});
    prefix.insert(prefix.end(),values.begin(),values.end());
    std::string output="trial"+std::to_string(trial)+".bin";
    write_file(output,prefix.data(),prefix.size()*8);
    write_file(folder+"/trial"+std::to_string(trial)+".guards.u8",guards.data(),guards.size());
    printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":0.0,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"old_wrong_values\":%d,\"timing_region\":\"diagnostic_only_one_step_Jacobi_no_performance_claim\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",
        trial,pass?"true":"false",prefix.size(),max_error,old_wrong,output.c_str(),prefix.size());
    fflush(stdout);
  }
  return 0;
 } catch(const std::exception& error) { fprintf(stderr,"%s\n",error.what()); return 1; }
}

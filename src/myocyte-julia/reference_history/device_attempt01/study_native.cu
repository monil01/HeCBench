
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define fp float
#include "/home/imo/HeCBench_julia/src/myocyte-cuda/kernel_ecc.cu"
#include "/home/imo/HeCBench_julia/src/myocyte-cuda/kernel_cam.cu"
#include "/home/imo/HeCBench_julia/src/myocyte-cuda/kernel.cu"
extern "C" double study_solve(void*,const float*,const float*,int);
extern "C" size_t study_count();
extern "C" void study_copy(double*);
extern "C" void study_check(const double*,size_t,int,const double*,double*);
const char*BASE="/home/imo/HeCBench_julia/src/myocyte-julia/";
float *dy,*dp,*df,*dc;const int G=16;
double callback(float t,const float*y,const float*p,float*out,float*com){
 cudaMemcpy(dy,y,91*4,cudaMemcpyHostToDevice);cudaMemcpy(dp,p,18*4,cudaMemcpyHostToDevice);
 cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();
 kernel<<<2,32>>>((int)t,dy,df,dp,dc);cudaDeviceSynchronize();
 double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 cudaMemcpy(out,df,91*4,cudaMemcpyDeviceToHost);cudaMemcpy(com,dc,3*4,cudaMemcpyDeviceToHost);
 auto e=cudaGetLastError();if(e!=cudaSuccess){fprintf(stderr,"CUDA%s\n",cudaGetErrorString(e));exit(3);}return dt;
}
std::vector<float>readtxt(const char*n){std::ifstream f(std::string(BASE)+n);std::vector<float>v;float x;while(f>>x)v.push_back(x);return v;}
int main(){
 auto y=readtxt("study_y.txt"),p=readtxt("study_params.txt");if(y.size()!=91||p.size()!=18)return 3;
 std::vector<double>pd(p.begin(),p.end());float**a[]={&dy,&dp,&df,&dc};int lens[]={91,18,91,3};
 for(int i=0;i<4;i++){std::vector<float>h(lens[i]+G,12345.5f);cudaMalloc(a[i],h.size()*4);cudaMemcpy(*a[i],h.data(),h.size()*4,cudaMemcpyHostToDevice);}
 study_solve((void*)callback,y.data(),p.data(),2); // all solverstage/JIT warmup outside trials.
 for(int trial=1;trial<=5;trial++){
  std::vector<double>v(y.begin(),y.end());v.insert(v.end(),p.begin(),p.end());for(int i:{1,2,2,32,91,18})v.push_back(i);
  double seconds=0,ma=0,mr=0;int ok=1;size_t checked=0;std::vector<double>components;std::vector<int>calls;
  for(int xmax:{1,2}){double dt=study_solve((void*)callback,y.data(),p.data(),xmax);components.push_back(dt);seconds+=dt;
   std::vector<double>result(study_count());study_copy(result.data());double metrics[6];study_check(result.data(),result.size(),xmax,pd.data(),metrics);
   ok&=metrics[0]==0;checked+=metrics[1];ma=fmax(ma,metrics[2]);mr=fmax(mr,metrics[3]);calls.push_back(metrics[4]);v.insert(v.end(),result.begin(),result.end());
  }
  for(int i=0;i<4;i++){std::vector<float>h(lens[i]+G);cudaMemcpy(h.data(),*a[i],h.size()*4,cudaMemcpyDeviceToHost);for(int j=lens[i];j<lens[i]+G;j++){v.push_back(h[j]);ok&=h[j]==12345.5f;checked++;}}
  std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/trial"+std::to_string(trial)+".bin";std::ofstream file(path,std::ios::binary);file.write((char*)v.data(),v.size()*8);file.close();
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"relative_error_denominator_floor\":1e-12,\"timing_region\":\"sum_original_gpu_evaluator_times_full_RKF_xmax1_and2\",\"jit_excluded\":true,\"internal_repeats\":1,\"evaluator_calls\":[%d,%d],\"component_time_seconds\":[%.17g,%.17g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",checked,ma,mr,calls[0],calls[1],components[0],components[1],path.c_str(),v.size());fflush(stdout);if(!ok)return 1;
 }
 for(auto a:a)cudaFree(*a);
}

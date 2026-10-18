
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
#include "/home/imo/HeCBench_julia/src/myocyte-julia/native_corrected/kernel_cam.cu"
#include "/home/imo/HeCBench_julia/src/myocyte-cuda/kernel.cu"
extern "C" void study_couple32(float*,float*,float*,float*);
extern "C" void study_probe_check(double,const float*,const float*,const float*,const float*,const float*,double*);
const char*BASE="/home/imo/HeCBench_julia/src/myocyte-julia/";const int G=16,R=20;
int main(){
 std::vector<float>input(444);std::ifstream file(std::string(BASE)+"study_probes_f32.bin",std::ios::binary);file.read((char*)input.data(),input.size()*4);if(!file)return 3;
 float*arrays[4];int lens[]={91,18,91,3};
 for(int i=0;i<4;i++){std::vector<float>h(lens[i]+G,12345.5f);cudaMalloc(&arrays[i],h.size()*4);cudaMemcpy(arrays[i],h.data(),h.size()*4,cudaMemcpyHostToDevice);}
 // Warm all declared branch/state specializations outside five trials.
 for(int c=0;c<4;c++){auto in=input.data()+c*111;cudaMemcpy(arrays[0],in,91*4,cudaMemcpyHostToDevice);cudaMemcpy(arrays[1],in+91,18*4,cudaMemcpyHostToDevice);kernel<<<2,32>>>((int)in[109],arrays[0],arrays[2],arrays[1],arrays[3]);cudaDeviceSynchronize();}
 for(int trial=1;trial<=5;trial++){
  std::vector<double>prefix,values,guards,times;bool ok=true;double ma=0,mr=0,balance=0;int checked=0;
  for(int c=0;c<4;c++){auto in=input.data()+c*111;cudaMemcpy(arrays[0],in,91*4,cudaMemcpyHostToDevice);cudaMemcpy(arrays[1],in+91,18*4,cudaMemcpyHostToDevice);
   double dt=0;
   for(int repeat=0;repeat<R;repeat++){cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();kernel<<<2,32>>>((int)in[109],arrays[0],arrays[2],arrays[1],arrays[3]);cudaDeviceSynchronize();dt+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
   times.push_back(dt/R);
   std::vector<float>actual[4];for(int i=0;i<4;i++){actual[i].resize(lens[i]+G);cudaMemcpy(actual[i].data(),arrays[i],actual[i].size()*4,cudaMemcpyDeviceToHost);for(int j=lens[i];j<lens[i]+G;j++)guards.push_back(actual[i][j]);}
   prefix.insert(prefix.end(),actual[0].begin(),actual[0].begin()+91);prefix.insert(prefix.end(),actual[1].begin(),actual[1].begin()+18);prefix.push_back(in[109]);prefix.push_back((int)in[109]);
   std::vector<float>coupled(actual[2].begin(),actual[2].begin()+91);study_couple32(actual[0].data(),actual[1].data(),coupled.data(),actual[3].data());
   double metrics[6];study_probe_check(in[109],actual[0].data(),actual[1].data(),actual[2].data(),actual[3].data(),coupled.data(),metrics);
   ok&=metrics[0]==0;checked+=metrics[1];ma=fmax(ma,metrics[2]);mr=fmax(mr,metrics[3]);balance=fmax(balance,metrics[5]);
   values.insert(values.end(),actual[2].begin(),actual[2].begin()+91);values.insert(values.end(),actual[3].begin(),actual[3].begin()+3);values.insert(values.end(),coupled.begin(),coupled.end());
  }
  for(auto g:guards)ok&=g==12345.5;checked+=guards.size();auto e=cudaGetLastError();ok&=e==cudaSuccess;
  auto out=prefix;out.insert(out.end(),values.begin(),values.end());out.insert(out.end(),guards.begin(),guards.end());std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/trial"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)out.data(),out.size()*8);f.close();
  double seconds=0;for(auto t:times)seconds+=t;
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"relative_error_denominator_floor\":1e-12,\"diffusion_balance_roundoff_budget_ratio\":%.17g,\"timing_region\":\"sum4_mean_source91RHS_device_evaluator_complete\",\"jit_excluded\":true,\"internal_repeats\":20,\"full_solver_trajectory_verified\":false,\"component_time_seconds\":[%.17g,%.17g,%.17g,%.17g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,ok?"true":"false",checked,ma,mr,balance,times[0],times[1],times[2],times[3],path.c_str(),out.size());fflush(stdout);if(!ok)return 1;
 }
 for(auto a:arrays)cudaFree(a);
}

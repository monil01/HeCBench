
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define main original_minisweep_main
#include "/home/imo/HeCBench_julia/src/minisweep-cuda/main.cu"
#undef main
const char*BASE="/home/imo/HeCBench_julia/src/minisweep-julia/";const int G=32,REPEATS=3;
std::vector<double>readv(std::string name,size_t n){std::vector<double>v(n);std::ifstream f(std::string(BASE)+name,std::ios::binary);f.read((char*)v.data(),n*8);if(!f)exit(3);return v;}
struct Case{Dimensions d;std::vector<size_t>lens;std::vector<double*>arrays;std::vector<double>oracle;StepInfoAll info;};
void launch(Case&c){
 auto d=c.d;auto&a=c.arrays;dim3 th(8,8,4),xy((d.ncell_x+7)/8,(d.ncell_y+7)/8,2),xz((d.ncell_x+7)/8,(d.ncell_z+7)/8,2),yz((d.ncell_y+7)/8,(d.ncell_z+7)/8,2);
 init_facexy<<<xy,th>>>(0,0,d.ne,d.na,d.ncell_x,d.ncell_y,d.ncell_z,d.ncell_z,a[4]);
 init_facexz<<<xz,th>>>(0,0,d.ne,d.na,d.ncell_x,d.ncell_y,d.ncell_z,1,1,c.info,a[5]);
 init_faceyz<<<yz,th>>>(0,0,d.ne,d.na,d.ncell_x,d.ncell_y,d.ncell_z,1,1,c.info,a[6]);
 wavefronts<<<dim3(2,(d.ne+63)/64),dim3(4,64)>>>(d.ncell_x+d.ncell_y+d.ncell_z-2,0,0,(int)c.lens[2],8,d,c.info,a[4],a[5],a[6],a[0],a[1],a[2],a[3],a[7]);
}
double execute(Case&c,int iterations,std::vector<double>&captured){
 captured.resize(c.lens[3]*iterations);double elapsed=0;
 for(int r=0;r<iterations;r++){
  cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();cudaMemset(c.arrays[3],0,c.lens[3]*8);launch(c);cudaDeviceSynchronize();elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  cudaMemcpy(captured.data()+r*c.lens[3],c.arrays[3],c.lens[3]*8,cudaMemcpyDeviceToHost);
 }
 auto status=cudaGetLastError();if(status!=cudaSuccess){fprintf(stderr,"CUDAerror%s\n",cudaGetErrorString(status));exit(1);}return elapsed/iterations;
}
int main(){
 std::vector<Case>cases(2);cases[0].d={5,5,5,30,4,33};cases[1].d={2,3,2,2,4,5};
 for(int caseid=0;caseid<2;caseid++){
  auto&c=cases[caseid];auto d=c.d;size_t am=d.nm*d.na*8,ns=Dimensions_size_state(d,4),fxy=Dimensions_size_facexy(d,4,8),fxz=Dimensions_size_facexz(d,4,8),fyz=Dimensions_size_faceyz(d,4,8);c.lens={am,am,ns,ns,fxy,fxz,fyz,fxy};c.arrays.resize(8);
  auto input=readv("study_case"+std::to_string(caseid+1)+"_f64.bin",am*2+ns);c.oracle=readv("oracle_case"+std::to_string(caseid+1)+"_f64.bin",ns*REPEATS+2*fxy+fxz+fyz);
  size_t k=0;for(int idx=0;idx<8;idx++){std::vector<double>h(c.lens[idx]+G,12345.5);if(idx<3){std::copy(input.begin()+k,input.begin()+k+c.lens[idx],h.begin());k+=c.lens[idx];}cudaMalloc(&c.arrays[idx],h.size()*8);cudaMemcpy(c.arrays[idx],h.data(),h.size()*8,cudaMemcpyHostToDevice);}
  if(caseid==0){std::vector<double>init(ns);initialize_input_state(init.data(),d,4);if(memcmp(init.data(),input.data()+2*am,ns*8)!=0){fprintf(stderr,"Sourceinitializerbytesdiffer\n");return 1;}}
  StepScheduler scheduler{1,1,1,1,8};if(StepScheduler_nstep(&scheduler)!=1)return 1;for(int o=0;o<8;o++){c.info.stepinfo[o]=StepScheduler_stepinfo(&scheduler,0,o,0,0);auto v=c.info.stepinfo[o];if(v.block_z!=0||v.octant!=o||!v.is_active)return 1;}
  std::vector<double>captured;execute(c,1,captured);
 }
 for(int trial=1;trial<=5;trial++){
  std::vector<double>prefix,values,guards,times;bool ok=true;double ma=0,mr=0;int mismatches=0;
  for(int caseid=0;caseid<2;caseid++){
   auto&c=cases[caseid];auto d=c.d;std::vector<double>v;times.push_back(execute(c,REPEATS,v));std::vector<std::vector<double>>actual(8);
   for(int idx=0;idx<8;idx++){actual[idx].resize(c.lens[idx]+G);cudaMemcpy(actual[idx].data(),c.arrays[idx],actual[idx].size()*8,cudaMemcpyDeviceToHost);if(idx<3)prefix.insert(prefix.end(),actual[idx].begin(),actual[idx].begin()+c.lens[idx]);guards.insert(guards.end(),actual[idx].begin()+c.lens[idx],actual[idx].end());}
   for(int p:{d.ncell_x,d.ncell_y,d.ncell_z,d.ne,d.na,4,4,8,REPEATS,1,1,1,1,caseid+1,1,0})prefix.push_back(p);
   for(int idx=4;idx<8;idx++)v.insert(v.end(),actual[idx].begin(),actual[idx].begin()+c.lens[idx]);
   for(size_t i=0;i<v.size();i++){double diff=fabs(v[i]-c.oracle[i]);ma=fmax(ma,diff);mr=fmax(mr,diff/fmax(fabs(c.oracle[i]),1e-12));bool pass=std::isfinite(v[i])&&diff<=1e-8+1e-10*fabs(c.oracle[i]);ok&=pass;mismatches+=!pass;}
   values.insert(values.end(),v.begin(),v.end());
  }
  for(auto g:guards)if(g!=12345.5)ok=false;
  auto all=prefix;all.insert(all.end(),values.begin(),values.end());all.insert(all.end(),guards.begin(),guards.end());std::string path=std::string(getenv("STUDY_OUTPUT_DIR"))+"/trial"+std::to_string(trial)+".bin";std::ofstream file(path,std::ios::binary);file.write((char*)all.data(),all.size()*8);file.close();
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":3356704,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"relative_error_denominator_floor\":1e-12,\"mismatch_values\":%d,\"timing_region\":\"sum2_mean_memset_three_face_inits_persistent_sweep_complete\",\"jit_excluded\":true,\"internal_repeats\":3,\"untimed_state_readbacks_per_case\":3,\"component_time_seconds\":[%.17g,%.17g],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,times[0]+times[1],ok?"true":"false",ma,mr,mismatches,times[0],times[1],path.c_str(),all.size());fflush(stdout);if(!ok)return 1;
 }
 for(auto&c:cases)for(auto a:c.arrays)cudaFree(a);
}

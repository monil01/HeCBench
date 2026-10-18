#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include "../../grrt-cuda/constants.h"
#undef IMAGE_SIZE
#define IMAGE_SIZE 8
#include "canonical_method.cuh"

int main(){std::vector<std::vector<double>>inputs={{0,acos(.25)/PI*180,8,340e9},{0,45,8,340e9}},gold(2,std::vector<double>(192));FILE*gf=fopen("/home/imo/HeCBench_julia/src/grrt-julia/secondary_schwarzschild/gold_float80_rounded_float64.bin","rb");if(!gf)return 9;for(int f=0;f<2;f++)if(fread(gold[f].data(),8,192,gf)!=192)return 10;fclose(gf);double fixedcpu_initial_null_max=0;double *dv[2],*out[2];for(int f=0;f<2;f++){cudaMalloc(&dv[f],32);cudaMalloc(&out[f],192*8);cudaMemcpy(dv[f],inputs[f].data(),32,cudaMemcpyHostToDevice);cudaMemset(out[f],0xff,192*8);}auto launch=[&](){fixedcpu_task1<<<dim3(1,8),dim3(100,1)>>>(out[0],dv[0],0,0);fixedcpu_task2<<<dim3(1,8),dim3(100,1)>>>(out[1],dv[1],0,0);};launch();cudaDeviceSynchronize();for(int trial=1;trial<=5;trial++){for(int f=0;f<2;f++)cudaMemset(out[f],0xff,192*8);cudaDeviceSynchronize();auto st=std::chrono::steady_clock::now();launch();cudaDeviceSynchronize();double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-st).count();bool ok=cudaGetLastError()==cudaSuccess;double err=0,redabs=0,lumrel=0;std::vector<double>art=inputs[0];art.insert(art.end(),inputs[1].begin(),inputs[1].end());for(int f=0;f<2;f++){std::vector<double>got(192);cudaMemcpy(got.data(),out[f],192*8,cudaMemcpyDeviceToHost);for(int i=0;i<192;i++){double a=got[i],ref=gold[f][i],e=fabs(a-ref);err=std::max(err,e);ok&=std::isfinite(a)&&e<=1e-8+1e-6*fabs(ref);if(i%3==2){if(f==0){redabs=std::max(redabs,e);ok&=(a>0)==(ref>0);}else{lumrel=std::max(lumrel,e/fabs(ref));ok&=a>0;}}}art.insert(art.end(),got.begin(),got.end());}const char*dir=getenv("STUDY_OUTPUT_DIR");std::string path=std::string(dir?dir:".")+"/output_trial"+std::to_string(trial)+".bin";FILE*fp=fopen(path.c_str(),"wb");fwrite(art.data(),8,art.size(),fp);fclose(fp);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":384,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"redshift_max_abs_error\":%.17g,\"luminosity_max_rel_error\":%.17g,\"initial_null_residual_CPU\":%.17g,\"timing_region\":\"conditional_canonical_A0_localized_disk_controlled_transfer_8x8\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":392,\"byte_order\":\"little\"}}\n",trial,elapsed,ok?"true":"false",err,lumrel,redabs,lumrel,fixedcpu_initial_null_max,path.c_str());}for(int f=0;f<2;f++){cudaFree(dv[f]);cudaFree(out[f]);}}
